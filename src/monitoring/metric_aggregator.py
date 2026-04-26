from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)


class AggregationWindow(StrEnum):
    LAST_1M = "1m"
    LAST_5M = "5m"
    LAST_15M = "15m"
    LAST_1H = "1h"


_WINDOW_SECONDS: dict[AggregationWindow, float] = {
    AggregationWindow.LAST_1M: 60.0,
    AggregationWindow.LAST_5M: 300.0,
    AggregationWindow.LAST_15M: 900.0,
    AggregationWindow.LAST_1H: 3600.0,
}


class MetricAggregator:
    """Ring-buffer metric aggregator with windowed statistics."""

    def __init__(self, max_points_per_metric: int = 10_000) -> None:
        self._max = max_points_per_metric
        self._store: dict[str, deque[MetricPoint]] = {}

    def record(self, name: str, value: float, labels: dict | None = None) -> MetricPoint:
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
        )
        if name not in self._store:
            self._store[name] = deque(maxlen=self._max)
        self._store[name].append(point)
        return point

    def get_window(self, name: str, window: AggregationWindow) -> list[MetricPoint]:
        cutoff = time.time() - _WINDOW_SECONDS[window]
        return [p for p in self._store.get(name, []) if p.timestamp >= cutoff]

    def stats(self, name: str, window: AggregationWindow) -> dict:
        points = self.get_window(name, window)
        if not points:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "stddev": 0.0,
            }
        values = np.array([p.value for p in points], dtype=float)
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "stddev": float(np.std(values)),
        }

    def rate(self, name: str, window: AggregationWindow) -> float:
        points = self.get_window(name, window)
        if not points:
            return 0.0
        return len(points) / _WINDOW_SECONDS[window]

    def list_metrics(self) -> list[str]:
        return list(self._store.keys())

    def flush(self, name: str) -> int:
        buf = self._store.pop(name, None)
        return len(buf) if buf is not None else 0


METRIC_AGGREGATOR = MetricAggregator()
