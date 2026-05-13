"""SRE Metrics Collector for golden signals monitoring.

This module provides the SREMetricsCollector class for tracking
the four golden signals of Site Reliability Engineering:
    - Latency: Response time distributions
    - Traffic: Request volume metrics
    - Errors: Error rates and counts
    - Saturation: Resource utilization levels

Features:
    - Counters for total requests, errors
    - Histograms for latency percentiles (p50, p90, p99, p99.9)
    - Health scores combining all signals
    - Thread-safe operations

Example:
    >>> collector = SREMetricsCollector()
    >>> collector.record_request(latency_ms=150)
    >>> collector.record_request(latency_ms=200)
    >>> collector.get_percentile(90)
    200.0
    >>> collector.get_health_score()
    0.95
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any


class Histogram:
    """Tracks a histogram of values for percentile calculations.

    Uses a simple sorted-sample approach suitable for low-to-medium
    volume metrics. For high-volume scenarios, consider implementing
    a t-digest or HDR histogram.
    """

    def __init__(self) -> None:
        """Initialize an empty histogram."""
        self._values: list[float] = []
        self._lock = threading.Lock()

    def record(self, value: float) -> None:
        """Record a single value.

        Args:
            value: The value to record.
        """
        with self._lock:
            self._values.append(value)

    def record_many(self, values: list[float]) -> None:
        """Record multiple values at once.

        Args:
            values: List of values to record.
        """
        with self._lock:
            self._values.extend(values)

    def get_percentile(self, percentile: float) -> float | None:
        """Calculate the given percentile of recorded values.

        Args:
            percentile: Percentile to calculate (0-100).

        Returns:
            The value at the given percentile, or None if no data.
        """
        with self._lock:
            if not self._values:
                return None
            sorted_values = sorted(self._values)
            n = len(sorted_values)
            idx = (percentile / 100) * (n - 1)
            lower = int(math.floor(idx))
            upper = int(math.ceil(idx))
            if lower == upper:
                return sorted_values[lower]
            # Linear interpolation
            fraction = idx - lower
            return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction

    def clear(self) -> None:
        """Clear all recorded values."""
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        """Return the number of recorded values.

        Returns:
            Count of recorded values.
        """
        with self._lock:
            return len(self._values)


class Counter:
    """Thread-safe counter for tracking cumulative metrics."""

    def __init__(self) -> None:
        """Initialize a counter at zero."""
        self._count = 0
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> None:
        """Increment the counter.

        Args:
            amount: Amount to add to the counter.
        """
        with self._lock:
            self._count += amount

    def get(self) -> int:
        """Get the current count.

        Returns:
            The current counter value.
        """
        with self._lock:
            return self._count

    def reset(self) -> None:
        """Reset the counter to zero."""
        with self._lock:
            self._count = 0


class SREMetricsCollector:
    """Collects and tracks SRE golden signals metrics.

    This class provides a unified interface for tracking latency,
    traffic, errors, and saturation metrics with health scoring.

    Attributes:
        name: Optional identifier for this collector.
    """

    # SLO thresholds
    LATENCY_SLO_MS = 200.0  # Target p90 latency
    ERROR_RATE_THRESHOLD = 0.05  # 5% error rate threshold
    SATURATION_THRESHOLD = 0.8  # 80% utilization threshold

    def __init__(self, name: str = "default") -> None:
        """Initialize the SRE metrics collector.

        Args:
            name: Identifier for this collector instance.
        """
        self.name = name
        self._created_at = time.time()

        # Counters
        self._request_counter = Counter()
        self._error_counter = Counter()
        self._traffic_counter = Counter()

        # Histograms
        self._latency_histogram = Histogram()
        self._saturation_histogram = Histogram()

        # Error tracking
        self._error_types: dict[str, int] = {}
        self._lock = threading.Lock()

    def record_request(self, latency_ms: float, success: bool = True) -> None:
        """Record a request with its latency.

        Args:
            latency_ms: Request latency in milliseconds.
            success: Whether the request succeeded.
        """
        self._request_counter.increment()
        self._latency_histogram.record(latency_ms)
        if not success:
            self._error_counter.increment()
            with self._lock:
                error_type = "generic"
                self._error_types[error_type] = self._error_types.get(error_type, 0) + 1

    def record_error(self, error_type: str = "generic") -> None:
        """Record an error occurrence.

        Args:
            error_type: Classification of the error.
        """
        self._error_counter.increment()
        with self._lock:
            self._error_types[error_type] = self._error_types.get(error_type, 0) + 1

    def record_traffic(self, count: int = 1) -> None:
        """Record traffic volume.

        Args:
            count: Number of requests/units to record.
        """
        self._traffic_counter.increment(count)

    def record_saturation(self, utilization: float) -> None:
        """Record saturation/utilization level.

        Args:
            utilization: Utilization value between 0.0 and 1.0.
        """
        self._saturation_histogram.record(utilization)

    def get_request_count(self) -> int:
        """Get total request count.

        Returns:
            Total number of recorded requests.
        """
        return self._request_counter.get()

    def get_error_count(self) -> int:
        """Get total error count.

        Returns:
            Total number of recorded errors.
        """
        return self._error_counter.get()

    def get_traffic_count(self) -> int:
        """Get total traffic count.

        Returns:
            Total traffic volume recorded.
        """
        return self._traffic_counter.get()

    def get_error_rate(self) -> float:
        """Calculate the current error rate.

        Returns:
            Error rate as a fraction (0.0 to 1.0).
        """
        total = self._request_counter.get()
        if total == 0:
            return 0.0
        return self._error_counter.get() / total

    def get_error_types(self) -> dict[str, int]:
        """Get a copy of error type counts.

        Returns:
            Dictionary mapping error types to their counts.
        """
        with self._lock:
            return dict(self._error_types)

    def get_percentile(self, percentile: float) -> float | None:
        """Get latency percentile.

        Args:
            percentile: Percentile to retrieve (0-100).

        Returns:
            Latency value at the given percentile in milliseconds.
        """
        return self._latency_histogram.get_percentile(percentile)

    def get_latency_stats(self) -> dict[str, float | None]:
        """Get comprehensive latency statistics.

        Returns:
            Dictionary with p50, p90, p95, p99, p99.9 latency values.
        """
        return {
            "p50": self.get_percentile(50),
            "p90": self.get_percentile(90),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "p99_9": self.get_percentile(99.9),
        }

    def get_saturation_stats(self) -> dict[str, float | None]:
        """Get saturation/utilization statistics.

        Returns:
            Dictionary with mean and peak saturation values.
        """
        with self._lock:
            values = list(self._saturation_histogram._values)  # noqa: SLF001
        if not values:
            return {"mean": None, "peak": None}
        return {
            "mean": sum(values) / len(values),
            "peak": max(values),
        }

    def get_latency_health_score(self) -> float:
        """Calculate health score based on latency.

        Returns:
            Score between 0.0 (unhealthy) and 1.0 (healthy).
        """
        p90 = self.get_percentile(90)
        if p90 is None:
            return 1.0  # No data = healthy
        if p90 <= self.LATENCY_SLO_MS:
            return 1.0
        # Score degrades as latency exceeds SLO
        ratio = p90 / self.LATENCY_SLO_MS
        return max(0.0, 1.0 - (ratio - 1) * 0.5)

    def get_error_health_score(self) -> float:
        """Calculate health score based on error rate.

        Returns:
            Score between 0.0 (unhealthy) and 1.0 (healthy).
        """
        error_rate = self.get_error_rate()
        if error_rate <= self.ERROR_RATE_THRESHOLD:
            return 1.0
        # Score degrades as error rate exceeds threshold
        ratio = error_rate / self.ERROR_RATE_THRESHOLD
        return max(0.0, 1.0 - (ratio - 1) * 0.5)

    def get_saturation_health_score(self) -> float:
        """Calculate health score based on saturation.

        Returns:
            Score between 0.0 (unhealthy) and 1.0 (healthy).
        """
        stats = self.get_saturation_stats()
        peak = stats.get("peak")
        if peak is None:
            return 1.0  # No data = healthy
        if peak <= self.SATURATION_THRESHOLD:
            return 1.0
        # Score degrades as saturation exceeds threshold
        ratio = peak / self.SATURATION_THRESHOLD
        return max(0.0, 1.0 - (ratio - 1) * 0.5)

    def get_health_score(self) -> float:
        """Calculate overall health score combining all signals.

        Uses a weighted average of individual signal scores.

        Returns:
            Overall health score between 0.0 and 1.0.
        """
        latency_score = self.get_latency_health_score()
        error_score = self.get_error_health_score()
        saturation_score = self.get_saturation_health_score()
        # Equal weights for all signals
        return (latency_score + error_score + saturation_score) / 3.0

    def get_summary(self) -> dict[str, Any]:
        """Get a complete metrics summary.

        Returns:
            Dictionary containing all metrics and health scores.
        """
        return {
            "name": self.name,
            "uptime_seconds": time.time() - self._created_at,
            "requests": self.get_request_count(),
            "errors": self.get_error_count(),
            "error_rate": self.get_error_rate(),
            "error_types": self.get_error_types(),
            "traffic": self.get_traffic_count(),
            "latency": self.get_latency_stats(),
            "saturation": self.get_saturation_stats(),
            "health_scores": {
                "latency": self.get_latency_health_score(),
                "error": self.get_error_health_score(),
                "saturation": self.get_saturation_health_score(),
                "overall": self.get_health_score(),
            },
        }

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self._request_counter.reset()
        self._error_counter.reset()
        self._traffic_counter.reset()
        self._latency_histogram.clear()
        self._saturation_histogram.clear()
        with self._lock:
            self._error_types.clear()
        self._created_at = time.time()

    def reset_counters(self) -> None:
        """Reset only counters, preserving histogram data."""
        self._request_counter.reset()
        self._error_counter.reset()
        self._traffic_counter.reset()
        with self._lock:
            self._error_types.clear()

    def __repr__(self) -> str:
        """Return string representation of the collector.

        Returns:
            A string showing the collector state.
        """
        return f"SREMetricsCollector(name={self.name!r}, health={self.get_health_score():.2f})"
