from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HistogramBucket:
    lower: float
    upper: float
    count: int


class LatencyHistogram:
    """Records latency samples and computes histogram statistics."""

    _DEFAULT_BUCKETS: list[float] = [1, 5, 10, 25, 50, 100, 250, 500, 1000, float("inf")]

    def __init__(self, buckets: list[float] | None = None) -> None:
        raw = buckets if buckets is not None else list(self._DEFAULT_BUCKETS)
        self._boundaries: list[float] = sorted(raw)
        # Ensure the last boundary is +inf so every sample lands somewhere
        if not math.isinf(self._boundaries[-1]):
            self._boundaries.append(float("inf"))
        self._counts: list[int] = [0] * len(self._boundaries)
        self._total_ms: float = 0.0
        self._n: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, latency_ms: float) -> None:
        """Increment the appropriate bucket count."""
        for i, upper in enumerate(self._boundaries):
            if latency_ms <= upper:
                self._counts[i] += 1
                break
        self._total_ms += latency_ms
        self._n += 1

    def total_count(self) -> int:
        return self._n

    def mean(self) -> float:
        if self._n == 0:
            return 0.0
        return self._total_ms / self._n

    def buckets(self) -> list[HistogramBucket]:
        """Return sorted list of HistogramBucket objects."""
        result: list[HistogramBucket] = []
        lower = 0.0
        for upper, count in zip(self._boundaries, self._counts):
            result.append(HistogramBucket(lower=lower, upper=upper, count=count))
            lower = upper
        return result

    def percentile(self, p: float) -> float:
        """Linear interpolation within bucket. p in [0, 100]. Returns 0.0 if no samples."""
        if self._n == 0:
            return 0.0

        target = (p / 100.0) * self._n  # rank (1-indexed count target)
        cumulative = 0
        lower = 0.0
        for upper, count in zip(self._boundaries, self._counts):
            if count == 0:
                lower = upper
                continue
            cumulative += count
            if cumulative >= target:
                # Interpolate within this bucket
                # How far into this bucket does our target rank fall?
                bucket_start_rank = cumulative - count  # rank before entering bucket
                fraction = (target - bucket_start_rank) / count
                bucket_lower = lower
                bucket_upper = upper if not math.isinf(upper) else lower
                return bucket_lower + fraction * (bucket_upper - bucket_lower)
            lower = upper

        # Should not reach here
        return lower

    def reset(self) -> None:
        self._counts = [0] * len(self._boundaries)
        self._total_ms = 0.0
        self._n = 0

    def to_dict(self) -> dict:
        return {
            "total": self._n,
            "mean_ms": self.mean(),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "buckets": [
                {"lower": b.lower, "upper": b.upper, "count": b.count} for b in self.buckets()
            ],
        }


LATENCY_HISTOGRAM_REGISTRY: dict[str, type[LatencyHistogram]] = {"default": LatencyHistogram}
