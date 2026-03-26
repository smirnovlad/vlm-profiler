"""Latency measurement with warmup, statistics, and per-component hooks."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyResult:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    all_times_ms: list[float]


def measure_latency(
    run_fn,
    warmup_runs: int = 3,
    timed_runs: int = 10,
    device: str = "cuda",
) -> LatencyResult:
    """Measure latency of a callable with warmup and statistics.

    Args:
        run_fn: Callable that performs one inference (no args).
        warmup_runs: Number of warmup iterations (not counted).
        timed_runs: Number of timed iterations.
        device: 'cuda' or 'cpu' — controls sync behavior.

    Returns:
        LatencyResult with timing statistics.
    """
    # Warmup
    for _ in range(warmup_runs):
        run_fn()
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times_ms = []
    for _ in range(timed_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        run_fn()

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    arr = np.array(times_ms)
    return LatencyResult(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        all_times_ms=times_ms,
    )
