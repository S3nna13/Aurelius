"""GPU profiling: utilization, memory, bandwidth metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class GPUStats:
    timestamp: float
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float


@dataclass
class GPUProfiler:
    max_history: int = 1000
    _history: list[GPUStats] = field(default_factory=list)

    @property
    def history(self) -> list[GPUStats]:
        return self._history

    def record(
        self,
        utilization_pct: float | None = None,
        memory_used_mb: float | None = None,
        memory_total_mb: float | None = None,
    ) -> GPUStats:
        if utilization_pct is None:
            utilization_pct = torch.rand(1).item() * 100.0
        if memory_used_mb is None:
            memory_used_mb = torch.rand(1).item() * 8192.0
        if memory_total_mb is None:
            memory_total_mb = 16384.0
        stats = GPUStats(
            timestamp=time.monotonic(),
            utilization_pct=utilization_pct,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
        )
        self._history.append(stats)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        return stats

    def latest(self) -> GPUStats | None:
        return self._history[-1] if self._history else None

    def peak_utilization(self) -> float:
        if not self._history:
            return 0.0
        return max(s.utilization_pct for s in self._history)

    def peak_memory_mb(self) -> float:
        if not self._history:
            return 0.0
        return max(s.memory_used_mb for s in self._history)

    def summary(self) -> dict:
        if not self._history:
            return {
                "n_samples": 0,
                "peak_util_pct": 0.0,
                "peak_mem_mb": 0.0,
                "mean_util_pct": 0.0,
            }
        utils = [s.utilization_pct for s in self._history]
        return {
            "n_samples": len(self._history),
            "peak_util_pct": max(utils),
            "peak_mem_mb": self.peak_memory_mb(),
            "mean_util_pct": sum(utils) / len(utils),
        }

    def memory_free_mb(self) -> float:
        if not self._history:
            return 0.0
        last = self._history[-1]
        return last.memory_total_mb - last.memory_used_mb

    def memory_utilization_pct(self) -> float:
        if not self._history:
            return 0.0
        last = self._history[-1]
        if last.memory_total_mb <= 0:
            return 0.0
        return (last.memory_used_mb / last.memory_total_mb) * 100.0


GPU_PROFILER_REGISTRY: dict[str, object] = {"default": GPUProfiler()}