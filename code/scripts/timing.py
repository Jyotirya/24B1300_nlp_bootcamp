"""Small timing and resource-statistics helpers for retrieval evaluation."""

from __future__ import annotations

import math
import importlib
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def percentile(values: Sequence[float], percentage: float) -> float:
    """Return a linearly interpolated percentile in milliseconds or seconds."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_statistics(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Calculate mean and p95 latency from measured query latencies."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    values = [float(value) for value in latencies_ms]
    return {
        "mean_query_latency_ms": sum(values) / len(values),
        "p95_query_latency_ms": percentile(values, 95),
    }


def timed_call(function: Callable[[], T]) -> tuple[T, float]:
    """Run a callable and return its value with elapsed wall-clock milliseconds."""
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000


def process_rss_mb() -> float:
    """Return the current process RSS in megabytes."""
    try:
        psutil = importlib.import_module("psutil")
    except ImportError as exc:
        raise RuntimeError("psutil is required for memory measurements") from exc
    return float(psutil.Process().memory_info().rss / (1024 * 1024))
