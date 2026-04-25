"""Aurelius metric aggregator: aggregates multiple metrics with weighting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricWeight:
    name: str
    weight: float


@dataclass(frozen=True)
class MetricScore:
    name: str
    value: float
    sample_count: int = 0


class MetricAggregator:
    """Aggregates metric scores with optional custom weights."""

    def __init__(self, weights: list[MetricWeight] | None = None) -> None:
        # Map from metric name -> weight value
        self._weights: dict[str, float] = (
            {w.name: w.weight for w in weights} if weights else {}
        )
        self._scores: dict[str, MetricScore] = {}

    def add_score(self, score: MetricScore) -> None:
        """Add or replace a metric score."""
        self._scores[score.name] = score

    def _get_weight(self, name: str) -> float:
        """Return weight for a metric, defaulting to 1.0 if not specified."""
        return self._weights.get(name, 1.0)

    def weighted_average(self) -> float:
        """Compute weighted average of all added scores. Returns 0.0 if no scores."""
        if not self._scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in self._scores.items():
            w = self._get_weight(name)
            weighted_sum += score.value * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    def report(self) -> dict:
        """Return a full report of scores, weighted average, and weights used."""
        scores_dict = {name: s.value for name, s in self._scores.items()}
        weights_used = {name: self._get_weight(name) for name in self._scores}
        return {
            "scores": scores_dict,
            "weighted_average": self.weighted_average(),
            "weights_used": weights_used,
        }

    def reset(self) -> None:
        """Clear all accumulated scores."""
        self._scores.clear()

    def best(self, n: int = 3) -> list[MetricScore]:
        """Return top-n scores by value descending."""
        sorted_scores = sorted(
            self._scores.values(), key=lambda s: s.value, reverse=True
        )
        return sorted_scores[:n]


METRIC_AGGREGATOR_REGISTRY = {"default": MetricAggregator}
