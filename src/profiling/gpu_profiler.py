"""GPU utilization and memory tracking (simulated, stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPUStats:
    device_id: int
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float = 0.0

    def memory_free_mb(self) -> float:
        return self.memory_total_mb - self.memory_used_mb

    def memory_utilization_pct(self) -> float:
        if self.memory_total_mb > 0:
            return self.memory_used_mb / self.memory_total_mb * 100
        return 0.0


class GPUProfiler:
    def __init__(self, num_devices: int = 1) -> None:
        self.num_devices = num_devices
        self._history: dict[int, list[GPUStats]] = {
            i: [] for i in range(num_devices)
        }

    def record(self, stats: GPUStats) -> None:
        device_id = stats.device_id
        if device_id not in self._history:
            self._history[device_id] = []
        self._history[device_id].append(stats)

    def latest(self, device_id: int = 0) -> Optional[GPUStats]:
        hist = self._history.get(device_id, [])
        return hist[-1] if hist else None

    def history(self, device_id: int = 0) -> list[GPUStats]:
        return list(self._history.get(device_id, []))

    def peak_utilization(self, device_id: int = 0) -> float:
        hist = self._history.get(device_id, [])
        if not hist:
            return 0.0
        return max(s.utilization_pct for s in hist)

    def peak_memory_mb(self, device_id: int = 0) -> float:
        hist = self._history.get(device_id, [])
        if not hist:
            return 0.0
        return max(s.memory_used_mb for s in hist)

    def summary(self, device_id: int = 0) -> dict:
        hist = self._history.get(device_id, [])
        return {
            "device": device_id,
            "peak_util_pct": self.peak_utilization(device_id),
            "peak_mem_mb": self.peak_memory_mb(device_id),
            "samples": len(hist),
        }


GPU_PROFILER_REGISTRY = {"default": GPUProfiler}
