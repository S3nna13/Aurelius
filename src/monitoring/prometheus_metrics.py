from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class MetricSample:
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramBucket:
    upper_bound: float
    count: int = 0


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def increment_counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            key = self._key(name, labels or {})
            self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            key = self._key(name, labels or {})
            self._gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            key = self._key(name, labels or {})
            self._histograms[key].append(value)

    def read_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        return self._counters.get(self._key(name, labels or {}), 0.0)

    def read_gauge(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        return self._gauges.get(self._key(name, labels or {}))

    def read_histogram(self, name: str, labels: dict[str, str] | None = None) -> list[float]:
        return self._histograms.get(self._key(name, labels or {}), [])

    def histogram_summary(self, name: str, labels: dict[str, str] | None = None) -> dict[str, float]:
        samples = self.read_histogram(name, labels)
        if not samples:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_s = sorted(samples)
        n = len(sorted_s)
        return {
            "count": n,
            "sum": sum(sorted_s),
            "avg": sum(sorted_s) / n,
            "min": sorted_s[0],
            "max": sorted_s[-1],
            "p50": sorted_s[n // 2],
            "p95": sorted_s[int(n * 0.95)],
            "p99": sorted_s[int(n * 0.99)],
        }

    def export_text(self) -> str:
        lines: list[str] = []
        with self._lock:
            for key, val in sorted(self._counters.items()):
                lines.append(f"# TYPE {key} counter")
                lines.append(f"{key} {val}")
            for key, val in sorted(self._gauges.items()):
                lines.append(f"# TYPE {key} gauge")
                lines.append(f"{key} {val}")
            for key, samples in sorted(self._histograms.items()):
                lines.append(f"# TYPE {key} histogram")
                for s in samples:
                    lines.append(f"{key} {s}")
        return "\n".join(lines)

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


METRICS_COLLECTOR = MetricsCollector()
