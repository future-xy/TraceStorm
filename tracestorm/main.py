import argparse
import multiprocessing
import os

from tracestorm.logger import init_logger
from tracestorm.process_datasets import get_datasets
from tracestorm.request_generator import generate_request
from tracestorm.result_analyzer import ResultAnalyzer
from tracestorm.trace_generator import generate_trace
from tracestorm.trace_player import play
from tracestorm.utils import round_robin_shard

logger = init_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run a replay of OpenAI requests."
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--rps", type=int, default=1, help="Requests per second"
    )
    parser.add_argument(
        "--pattern",
        choices=["uniform", "poisson", "random"],
        default="uniform",
        help="Pattern for generating trace",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for trace reproducibility",
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="Duration in seconds"
    )
    parser.add_argument(
        "--subprocesses", type=int, default=1, help="Number of subprocesses"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI Base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "none"),
        help="OpenAI API Key",
    )
    parser.add_argument(
        "--datasets", default=None, help="Config file for datasets"
    )

    return parser.parse_args()


def main():
    args = get_args()

    raw_trace = generate_trace(args.rps, args.pattern, args.duration, args.seed)
    total_requests = len(raw_trace)
    logger.debug(f"Raw trace: {raw_trace}")

    datasets, sort_strategy = (None, None)
    if args.datasets:
        datasets, sort_strategy = get_datasets(args.datasets)
        logger.info(f"Loaded datasets with sort strategy: {sort_strategy}")

    requests = generate_request(
        args.model, total_requests, datasets, sort_strategy
    )
    logger.debug(f"Requests: {requests}")

    ipc_queue = multiprocessing.Queue()
    processes = []

    if total_requests == 0:
        logger.warning("No requests to process. Trace is empty.")
        return

    # Launch subprocesses
    for i, (partial_trace, partial_requests) in enumerate(
        round_robin_shard(raw_trace, requests, args.subprocesses), start=1
    ):
        p = multiprocessing.Process(
            target=play,
            args=(
                f"TracePlayer-{i}",
                partial_trace,
                partial_requests,
                args.base_url,
                args.api_key,
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
