from __future__ import annotations

import time
from typing import Any


class TimerContext:
    _counter: int = 0

    def __init__(self, profiler: LatencyProfiler, name: str | None = None) -> None:
        if name is None:
            name = f"block_{TimerContext._counter}"
            TimerContext._counter += 1
        self.profiler = profiler
        self.name = name

    def __enter__(self) -> str:
        self.start = time.perf_counter()
        return self.name

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.perf_counter() - self.start) * 1000
        self.profiler.record(self.name, elapsed)


class LatencyProfiler:
    def __init__(self) -> None:
        self._records: dict[str, list[float]] = {}

    def record(self, name: str, ms: float) -> None:
        if name not in self._records:
            self._records[name] = []
        self._records[name].append(ms)

    def get_stats(self, name: str) -> dict[str, float] | None:
        samples = self._records.get(name)
        if not samples:
            return None
        n = len(samples)
        total = sum(samples)
        sorted_s = sorted(samples)
        return {
            "count": n,
            "total_ms": total,
            "avg_ms": total / n,
            "min_ms": sorted_s[0],
            "max_ms": sorted_s[-1],
            "p50_ms": sorted_s[n // 2],
            "p95_ms": sorted_s[int(n * 0.95)],
        }

    def summary(self) -> dict[str, dict[str, float]]:
        return {name: self.get_stats(name) for name in self._records if self.get_stats(name)}

    def reset(self) -> None:
        self._records.clear()


LATENCY_PROFILER = LatencyProfiler()
