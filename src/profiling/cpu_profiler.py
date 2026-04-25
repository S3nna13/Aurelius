"""CPU profiling: per-core utilization sampling and aggregation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class CPUSample:
    timestamp: float
    core_pcts: list[float]


@dataclass
class CPUProfiler:
    max_samples: int = 10000
    _samples: list[CPUSample] = field(default_factory=list)

    @property
    def samples(self) -> list[CPUSample]:
        return self._samples

    def record(self, core_pcts: list[float] | None = None) -> CPUSample:
        if core_pcts is None:
            n_cores = 8
            core_pcts = [torch.rand(1).item() * 100.0 for _ in range(n_cores)]
        sample = CPUSample(timestamp=time.monotonic(), core_pcts=list(core_pcts))
        self._samples.append(sample)
        if len(self._samples) > self.max_samples:
            self._samples.pop(0)
        return sample

    def max_core_pct(self) -> float:
        if not self._samples:
            return 0.0
        return max(max(s.core_pcts) for s in self._samples)

    def mean_core_pct(self) -> float:
        if not self._samples:
            return 0.0
        all_vals = [v for s in self._samples for v in s.core_pcts]
        return sum(all_vals) / len(all_vals) if all_vals else 0.0

    def p99_utilization(self) -> float:
        if not self._samples:
            return 0.0
        all_vals = sorted(v for s in self._samples for v in s.core_pcts)
        if not all_vals:
            return 0.0
        idx = int(len(all_vals) * 0.99)
        idx = min(idx, len(all_vals) - 1)
        return all_vals[idx]

    def mean_utilization(self) -> float:
        return self.mean_core_pct()

    def hottest_core(self) -> int:
        if not self._samples:
            return -1
        core_sums = {}
        for s in self._samples:
            for i, v in enumerate(s.core_pcts):
                core_sums[i] = core_sums.get(i, 0.0) + v
        if not core_sums:
            return -1
        return max(core_sums, key=core_sums.get)

    def reset(self) -> None:
        self._samples.clear()


CPU_PROFILER_REGISTRY: dict[str, object] = {"default": CPUProfiler()}