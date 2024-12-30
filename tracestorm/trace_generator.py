import math
import random
from typing import List, Optional

import numpy as np


def generate_trace(
    rps: int, pattern: str, duration: int, seed: Optional[int] = None
) -> List[int]:
    """
    Generates a list of timestamps (ms) at which requests should be sent.

    Args:
        rps (int): Requests per second. Must be non-negative.
        pattern (str): Distribution pattern ('uniform', 'random', 'poisson', etc.).
        duration (int): Total duration in seconds. Must be non-negative.
        seed (int): seed for reproducibility of 'poisson' and 'random' patterns

    Returns:
        List[int]: Sorted list of timestamps in milliseconds.

    Raises:
        ValueError: If an unknown pattern is provided or if inputs are invalid.
    """
    if not isinstance(rps, int) or rps < 0:
        raise ValueError("rps must be a non-negative integer")

    if not isinstance(duration, int) or duration < 0:
        raise ValueError("duration must be a non-negative integer")

    total_requests = rps * duration
    total_duration_ms = duration * 1000
    timestamps = []

    if total_requests == 0:
        return timestamps

    if seed is not None:
        np.random.seed(seed)

    if pattern == "uniform":
        # Distribute requests evenly across the duration
        interval = total_duration_ms / total_requests
        current_time = 0.0
        for _ in range(total_requests):
            timestamp = int(round(current_time))
            timestamp = min(timestamp, total_duration_ms - 1)
            timestamps.append(timestamp)
            current_time += interval
    elif pattern == "poisson":
        # Exponential distribution for intervals
        rate_ms = rps / 1000
        intervals = np.random.exponential(1 / rate_ms, total_requests)
        current_time = 0.0
        for i in range(total_requests):
            timestamp = int(round(current_time))
            timestamps.append(timestamp)
            current_time += intervals[i]
    elif pattern == "random":
        timestamps = np.random.randint(
            0, total_duration_ms, size=total_requests
        ).tolist()
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return sorted(timestamps)
