"""
Resource monitoring (GPU utilization, power, energy and memory sampled with
NVML; process memory with psutil) and percentile-bootstrap confidence intervals.
"""

import threading
import time
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

_NVML_INIT_DONE = False
_NVML_INIT_FAILED = False


def _get_nvml_handle(gpu_index: int):
    """NVML handle of a GPU; NVML is initialized once per process. Returns None when
    NVML is unavailable."""
    global _NVML_INIT_DONE, _NVML_INIT_FAILED
    if not _NVML_AVAILABLE or _NVML_INIT_FAILED:
        return None
    if not _NVML_INIT_DONE:
        try:
            pynvml.nvmlInit()
            _NVML_INIT_DONE = True
        except Exception as e:
            logger.warning(f"NVML init failed, GPU metrics disabled: {e}")
            _NVML_INIT_FAILED = True
            return None
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception as e:
        logger.warning(f"Could not get NVML handle for GPU {gpu_index}: {e}")
        return None


@dataclass
class ResourceSample:
    timestamp: float
    gpu_util_pct: Optional[float]
    gpu_memory_mb: Optional[float]
    gpu_power_w: Optional[float]
    cpu_util_pct: float
    rss_memory_mb: float


@dataclass
class ResourceSummary:
    duration_s: float
    avg_gpu_util_pct: Optional[float]
    peak_gpu_memory_mb: Optional[float]
    avg_gpu_power_w: Optional[float]
    energy_joules: Optional[float]
    avg_cpu_util_pct: float
    peak_rss_memory_mb: float
    num_samples: int


class ResourceMonitor:
    """Background sampler of GPU utilization, memory and power (NVML) and of process
    CPU and memory use (psutil), every `sample_interval_s` seconds."""

    def __init__(self, gpu_index: int = 0, sample_interval_s: float = 0.2):
        self.sample_interval_s = sample_interval_s
        self.samples: List[ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
        self._nvml_handle = _get_nvml_handle(gpu_index)

    def _sample_once(self) -> ResourceSample:
        gpu_util = gpu_mem = gpu_power = None
        if self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)  # milliwatts
                gpu_util = float(util.gpu)
                gpu_mem = float(mem.used) / (1024 ** 2)
                gpu_power = float(power) / 1000.0
            except Exception as e:
                logger.debug(f"NVML sample failed: {e}")

        return ResourceSample(
            timestamp=time.time(),
            gpu_util_pct=gpu_util,
            gpu_memory_mb=gpu_mem,
            gpu_power_w=gpu_power,
            cpu_util_pct=self._process.cpu_percent(interval=None),
            rss_memory_mb=self._process.memory_info().rss / (1024 ** 2),
        )

    def _run(self):
        while not self._stop_event.is_set():
            self.samples.append(self._sample_once())
            time.sleep(self.sample_interval_s)

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._process.cpu_percent(interval=None)  # prime the counter
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_s * 5)

        if not self.samples:
            return ResourceSummary(0.0, None, None, None, None, 0.0, 0.0, 0)

        duration = self.samples[-1].timestamp - self.samples[0].timestamp
        gpu_utils = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        gpu_mems = [s.gpu_memory_mb for s in self.samples if s.gpu_memory_mb is not None]
        gpu_powers = [s.gpu_power_w for s in self.samples if s.gpu_power_w is not None]

        energy_j = None
        if len(gpu_powers) >= 2:
            # energy: trapezoidal integral of the sampled power over time
            energy_j = 0.0
            for i in range(1, len(self.samples)):
                p0, p1 = self.samples[i - 1].gpu_power_w, self.samples[i].gpu_power_w
                if p0 is None or p1 is None:
                    continue
                dt = self.samples[i].timestamp - self.samples[i - 1].timestamp
                energy_j += 0.5 * (p0 + p1) * dt

        return ResourceSummary(
            duration_s=duration,
            avg_gpu_util_pct=float(np.mean(gpu_utils)) if gpu_utils else None,
            peak_gpu_memory_mb=float(np.max(gpu_mems)) if gpu_mems else None,
            avg_gpu_power_w=float(np.mean(gpu_powers)) if gpu_powers else None,
            energy_joules=energy_j,
            avg_cpu_util_pct=float(np.mean([s.cpu_util_pct for s in self.samples])),
            peak_rss_memory_mb=float(np.max([s.rss_memory_mb for s in self.samples])),
            num_samples=len(self.samples),
        )


def bootstrap_ci(data: List[float], num_resamples: int = 10000, ci: float = 0.95,
                  statistic: Callable[[np.ndarray], float] = np.mean,
                  seed: int = 42) -> Tuple[float, float, float]:
    """Percentile-bootstrap confidence interval; returns (estimate, lower, upper)."""
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan')
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    point_estimate = float(statistic(arr))
    resample_stats = np.empty(num_resamples)
    n = arr.size
    for i in range(num_resamples):
        resample = arr[rng.integers(0, n, size=n)]
        resample_stats[i] = statistic(resample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(resample_stats, 100 * alpha))
    upper = float(np.percentile(resample_stats, 100 * (1 - alpha)))
    return point_estimate, lower, upper
