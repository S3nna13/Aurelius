"""CPU utilization and per-core stats tracking."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CPUSample:
    timestamp: float
    overall_pct: float
    per_core_pct: list[float]
    num_cores: int

    def max_core_pct(self) -> float:
        return max(self.per_core_pct) if self.per_core_pct else 0.0

    def mean_core_pct(self) -> float:
        return statistics.mean(self.per_core_pct) if self.per_core_pct else 0.0


class CPUProfiler:
    def __init__(self, max_samples: int = 1000) -> None:
        self.max_samples = max_samples
        self._samples: list[CPUSample] = []

    def record(
        self, overall_pct: float, per_core_pct: list[float] | None = None
    ) -> CPUSample:
        sample = CPUSample(
            timestamp=time.monotonic(),
            overall_pct=overall_pct,
            per_core_pct=list(per_core_pct or []),
            num_cores=len(per_core_pct or []),
        )
        if len(self._samples) >= self.max_samples:
            self._samples.pop(0)
        self._samples.append(sample)
        return sample

    def samples(self) -> list[CPUSample]:
        return list(self._samples)

    def p99_utilization(self) -> float:
        if not self._samples:
            return 0.0
        vals = sorted(s.overall_pct for s in self._samples)
        n = len(vals)
        idx = int(0.99 * n) - 1
        idx = max(0, min(idx, n - 1))
        return vals[idx]

    def mean_utilization(self) -> float:
        if not self._samples:
            return 0.0
        return statistics.mean(s.overall_pct for s in self._samples)

    def hottest_core(self) -> int | None:
        if not self._samples:
            return None
        max_cores = max(s.num_cores for s in self._samples)
        if max_cores == 0:
            return None
        core_totals: list[float] = [0.0] * max_cores
        core_counts: list[int] = [0] * max_cores
        for sample in self._samples:
            for i, val in enumerate(sample.per_core_pct):
                core_totals[i] += val
                core_counts[i] += 1
        means = [
            core_totals[i] / core_counts[i] if core_counts[i] > 0 else 0.0
            for i in range(max_cores)
        ]
        return means.index(max(means)) if means else None

    def reset(self) -> None:
        self._samples.clear()


CPU_PROFILER_REGISTRY: dict[str, object] = {"default": CPUProfiler}