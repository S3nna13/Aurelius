"""Metrics collector with counters, histograms, gauges and time-windowed aggregation."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _HistogramState:
    values: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))


@dataclass(slots=True)
class _GaugeState:
    value: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass(slots=True)
class _CounterState:
    value: float = 0.0


class MetricsCollector:
    """Thread-safe metrics collector supporting counters, histograms, and gauges.

    Histograms store values in a bounded deque for time-windowed aggregation.
    All operations are non-blocking and lock-protected.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, _CounterState] = {}
        self._histograms: dict[str, _HistogramState] = {}
        self._gauges: dict[str, _GaugeState] = {}

    # ------------------------------------------------------------------ #
    # Counters
    # ------------------------------------------------------------------ #

    def increment(self, name: str, value: float = 1.0) -> float:
        """Increment a counter. Returns new value."""
        with self._lock:
            state = self._counters.get(name)
            if state is None:
                state = _CounterState()
                self._counters[name] = state
            state.value += value
            return state.value

    def counter_value(self, name: str) -> float:
        with self._lock:
            state = self._counters.get(name)
            return state.value if state else 0.0

    def reset_counter(self, name: str) -> float:
        """Reset counter to 0 and return previous value."""
        with self._lock:
            state = self._counters.get(name)
            if state is None:
                return 0.0
            prev = state.value
            state.value = 0.0
            return prev

    # ------------------------------------------------------------------ #
    # Histograms
    # ------------------------------------------------------------------ #

    def record(self, name: str, value: float) -> None:
        """Record a value into a histogram."""
        with self._lock:
            state = self._histograms.get(name)
            if state is None:
                state = _HistogramState()
                self._histograms[name] = state
            state.values.append(value)

    def histogram_summary(self, name: str) -> dict[str, Any]:
        """Return summary stats for a histogram."""
        with self._lock:
            state = self._histograms.get(name)
            if state is None or not state.values:
                return {"count": 0, "min": None, "max": None, "mean": None, "p99": None}
            vals = list(state.values)
        vals.sort()
        n = len(vals)
        total = sum(vals)
        mean = total / n
        p99_idx = min(len(vals) - 1, int(n * 0.99))
        return {
            "count": n,
            "min": vals[0],
            "max": vals[-1],
            "mean": mean,
            "p99": vals[p99_idx],
        }

    def histogram_values(self, name: str) -> list[float]:
        with self._lock:
            state = self._histograms.get(name)
            return list(state.values) if state else []

    def reset_histogram(self, name: str) -> list[float]:
        with self._lock:
            state = self._histograms.get(name)
            if state is None:
                return []
            prev = list(state.values)
            state.values.clear()
            return prev

    # ------------------------------------------------------------------ #
    # Gauges
    # ------------------------------------------------------------------ #

    def gauge(self, name: str, value: float) -> float:
        """Set a gauge. Returns the new value."""
        with self._lock:
            state = self._gauges.get(name)
            if state is None:
                state = _GaugeState()
                self._gauges[name] = state
            state.value = value
            state.last_update = time.time()
            return value

    def gauge_value(self, name: str) -> float | None:
        with self._lock:
            state = self._gauges.get(name)
            return state.value if state else None

    def gauge_age(self, name: str) -> float | None:
        """Return seconds since last gauge update."""
        with self._lock:
            state = self._gauges.get(name)
            if state is None:
                return None
            return time.time() - state.last_update

    # ------------------------------------------------------------------ #
    # Bulk operations
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all metrics."""
        with self._lock:
            counters = {k: v.value for k, v in self._counters.items()}
            gauges = {
                k: {"value": v.value, "last_update": v.last_update} for k, v in self._gauges.items()
            }
            histograms = {k: list(v.values) for k, v in self._histograms.items()}
        return {
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
        }

    def reset_all(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
