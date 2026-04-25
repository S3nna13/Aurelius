"""Metrics collector: counters, gauges, histograms with sliding windows."""
from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricSample:
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.monotonic)
    labels: dict = field(default_factory=dict)


class MetricsCollector:
    def __init__(self, window_size: int = 1000) -> None:
        self._window_size = window_size
        self._buffers: dict[str, Deque[MetricSample]] = {}

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        **labels,
    ) -> MetricSample:
        sample = MetricSample(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.monotonic(),
            labels=dict(labels),
        )
        if name not in self._buffers:
            self._buffers[name] = deque(maxlen=self._window_size)
        self._buffers[name].append(sample)
        return sample

    def increment(self, name: str, by: float = 1.0, **labels) -> None:
        self.record(name, by, MetricType.COUNTER, **labels)

    def get_samples(self, name: str, last_n: int | None = None) -> list[MetricSample]:
        buf = self._buffers.get(name, deque())
        samples = list(buf)
        if last_n is not None:
            samples = samples[-last_n:]
        return samples

    def summary(self, name: str) -> dict:
        samples = self.get_samples(name)
        zero = {"count": 0, "sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0}
        if not samples:
            return zero
        values = [s.value for s in samples]
        n = len(values)
        total = math.fsum(values)
        sorted_vals = sorted(values)

        def percentile(p: float) -> float:
            if n == 1:
                return sorted_vals[0]
            # Use statistics.quantiles when n >= 2
            # quantiles returns n-1 cut points for n equal groups
            # For p95 and p99 we need 100 groups; for p50 we need 2 groups
            idx = (n - 1) * p
            lo = int(idx)
            hi = lo + 1
            if hi >= n:
                return sorted_vals[-1]
            frac = idx - lo
            return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

        return {
            "count": n,
            "sum": total,
            "mean": total / n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

    def metric_names(self) -> list[str]:
        return list(self._buffers.keys())


METRICS_COLLECTOR = MetricsCollector()
