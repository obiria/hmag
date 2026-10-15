"""
Concurrent throughput measurement.

`c` client threads share a query list; each sends its next request as soon as
its previous one is answered. Throughput is completed requests per second of
wall-clock time, and latency runs from sending a request to receiving its
answer. The inverse of the mean latency is recorded separately for comparison.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyLevelResult:
    concurrency: int
    num_queries: int
    wall_clock_s: float
    throughput_qps: float          # completed requests / wall-clock time
    inverse_latency_qps: float     # 1000 / mean latency in ms
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    errors: int


def _timed_call(fn: Callable[[str], Any], query: str) -> Tuple[float, bool]:
    """Returns (latency_ms, is_error); a failed generation is reported in the metadata."""
    start = time.perf_counter()
    result = fn(query)
    latency_ms = (time.perf_counter() - start) * 1000.0
    metadata = getattr(result, 'metadata', None) or {}
    return latency_ms, bool(metadata.get('generation_error'))


def run_concurrency_sweep(process_fn_factory: Callable[[], Callable[[str], Any]],
                           queries: Sequence[str],
                           concurrency_levels: Sequence[int] = (1, 2, 4, 8, 16),
                           queries_per_level: int = 20) -> List[ConcurrencyLevelResult]:
    """Runs `queries_per_level` queries at each concurrency level with a thread pool.
    `process_fn_factory()` builds a fresh system for every level, so no cache state
    carries over between levels."""
    results = []
    pool = list(queries)
    if not pool:
        raise ValueError("queries must be non-empty")

    for concurrency in concurrency_levels:
        process_fn = process_fn_factory()
        batch = [pool[i % len(pool)] for i in range(queries_per_level)]
        latencies: List[float] = []
        errors = 0

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_timed_call, process_fn, q) for q in batch]
            for future in as_completed(futures):
                try:
                    latency_ms, is_error = future.result()
                    latencies.append(latency_ms)
                    if is_error:
                        errors += 1
                except Exception as e:
                    logger.warning(f"Query failed during load test: {e}")
                    errors += 1
        wall_clock = time.perf_counter() - start

        completed = len(latencies)
        throughput = completed / wall_clock if wall_clock > 0 else 0.0
        inverse_latency_throughput = (1000.0 / np.mean(latencies)) if latencies else 0.0

        results.append(ConcurrencyLevelResult(
            concurrency=concurrency,
            num_queries=completed,
            wall_clock_s=wall_clock,
            throughput_qps=throughput,
            inverse_latency_qps=inverse_latency_throughput,
            latency_mean_ms=float(np.mean(latencies)) if latencies else 0.0,
            latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
            latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            errors=errors,
        ))
        logger.info(f"Concurrency={concurrency}: throughput={throughput:.3f} QPS "
                    f"(1/mean-latency={inverse_latency_throughput:.3f} QPS), errors={errors}")

    return results
