import multiprocessing
import os

import click

from tracestorm.logger import init_logger
from tracestorm.process_datasets import get_datasets
from tracestorm.request_generator import generate_request
from tracestorm.result_analyzer import ResultAnalyzer
from tracestorm.trace_generator import generate_trace
from tracestorm.trace_player import play
from tracestorm.utils import round_robin_shard

logger = init_logger(__name__)


@click.command()
@click.option("--model", required=True, help="Model name")
@click.option("--rps", type=int, default=1, help="Requests per second")
@click.option(
    "--pattern", default="uniform", help="Pattern for generating trace"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility of trace patterns",
)
@click.option("--duration", type=int, default=10, help="Duration in seconds")
@click.option(
    "--subprocesses", type=int, default=1, help="Number of subprocesses"
)
@click.option(
    "--base-url",
    default=lambda: os.environ.get(
        "OPENAI_BASE_URL", "http://localhost:8000/v1"
    ),
    help="OpenAI Base URL",
)
@click.option(
    "--api-key",
    default=lambda: os.environ.get("OPENAI_API_KEY", "none"),
    help="OpenAI API Key",
)
@click.option("--datasets", default=None, help="Config file for datasets")
def main(
    model,
    rps,
    pattern,
    seed,
    duration,
    subprocesses,
    base_url,
    api_key,
    datasets,
):
    raw_trace = generate_trace(rps, pattern, duration)
    total_requests = len(raw_trace)
    logger.debug(f"Raw trace: {raw_trace}")

    sort_strategy = None
    if datasets:
        datasets, sort_strategy = get_datasets(datasets)
        logger.info(f"Loaded datasets with sort strategy: {sort_strategy}")

    requests = generate_request(model, total_requests, datasets, sort_strategy)
    logger.debug(f"Requests: {requests}")

    ipc_queue = multiprocessing.Queue()
    processes = []

    if total_requests == 0:
        logger.warning("No requests to process. Trace is empty.")
        return

    for i, (partial_trace, partial_requests) in enumerate(
        round_robin_shard(raw_trace, requests, subprocesses), start=1
    ):
        p = multiprocessing.Process(
            target=play,
            args=(
                f"TracePlayer-{i}",
                partial_trace,
                partial_requests,
                base_url,
                api_key,
                ipc_queue,
            ),
        )
        p.start()
        processes.append(p)

    results_collected = 0
    aggregated_results = []
    while results_collected < total_requests:
        try:
            name, timestamp, resp = ipc_queue.get(timeout=30)
            results_collected += 1
            logger.info(
                f"Received result from {name} for timestamp {timestamp}: {resp['token_count']} tokens"
            )
            aggregated_results.append((name, timestamp, resp))
        except Exception as e:
            logger.error(
                f"Timeout or error reading from IPC queue: {e}", exc_info=True
            )
            break

    for p in processes:
        p.join()

    logger.info("All subprocesses have finished.")
    logger.debug(f"Aggregated results: {aggregated_results}")

    result_analyzer = ResultAnalyzer()
    result_analyzer.store_raw_results(aggregated_results)
    print(result_analyzer)
    result_analyzer.plot_cdf()


if __name__ == "__main__":
    main()
