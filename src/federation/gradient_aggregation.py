"""Gradient aggregation: FedAvg, FedMedian, Krum, Trimmed Mean."""

from __future__ import annotations

import math
from enum import Enum


class AggregationStrategy(str, Enum):
    FEDAVG = "fedavg"
    FEDMEDIAN = "fedmedian"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"


class GradientAggregator:
    """Aggregates gradients from multiple clients using various strategies."""

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        trim_fraction: float = 0.1,
    ) -> None:
        self.strategy = strategy
        self.trim_fraction = trim_fraction

    def fedavg(
        self,
        gradients: list[list[float]],
        weights: list[float] | None = None,
    ) -> list[float]:
        """Weighted mean per position; equal weights if None."""
        if not gradients:
            return []

        n = len(gradients)
        n_params = len(gradients[0])

        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        result = []
        for i in range(n_params):
            val = sum(w * g[i] for w, g in zip(weights, gradients))
            result.append(val)

        return result

    def fedmedian(self, gradients: list[list[float]]) -> list[float]:
        """Median per position (sorted + middle element)."""
        if not gradients:
            return []

        n_params = len(gradients[0])
        result = []
        for i in range(n_params):
            col = sorted(g[i] for g in gradients)
            mid = len(col) // 2
            if len(col) % 2 == 1:
                median = col[mid]
            else:
                median = (col[mid - 1] + col[mid]) / 2.0
            result.append(median)

        return result

    def krum(self, gradients: list[list[float]], f: int = 1) -> list[float]:
        """Multi-Krum: return mean of n-f gradients with smallest scores."""
        n = len(gradients)
        if n == 0:
            return []

        n_params = len(gradients[0])

        def sq_dist(a: list[float], b: list[float]) -> float:
            return sum((x - y) ** 2 for x, y in zip(a, b))

        # Number of neighbours to consider per gradient: n - f - 2
        k = max(1, n - f - 2)

        scores = []
        for i in range(n):
            dists = sorted(
                sq_dist(gradients[i], gradients[j]) for j in range(n) if j != i
            )
            score = sum(dists[:k])
            scores.append((score, i))

        scores.sort(key=lambda x: x[0])
        # Select n - f gradients with smallest scores
        n_select = max(1, n - f)
        selected = [gradients[idx] for _, idx in scores[:n_select]]

        # Return mean of selected gradients
        result = []
        for i in range(n_params):
            result.append(sum(g[i] for g in selected) / len(selected))

        return result

    def trimmed_mean(
        self,
        gradients: list[list[float]],
        trim_fraction: float | None = None,
    ) -> list[float]:
        """Per position: sort, trim top and bottom trim_fraction, take mean of remainder."""
        if not gradients:
            return []

        if trim_fraction is None:
            trim_fraction = self.trim_fraction

        n = len(gradients)
        n_params = len(gradients[0])
        k = int(math.floor(n * trim_fraction))

        result = []
        for i in range(n_params):
            col = sorted(g[i] for g in gradients)
            if 2 * k < n:
                trimmed = col[k: n - k]
            else:
                trimmed = col
            result.append(sum(trimmed) / len(trimmed))

        return result

    def aggregate(
        self,
        gradients: list[list[float]],
        weights: list[float] | None = None,
    ) -> list[float]:
        """Dispatch to the appropriate aggregation method by strategy."""
        if self.strategy == AggregationStrategy.FEDAVG:
            return self.fedavg(gradients, weights)
        elif self.strategy == AggregationStrategy.FEDMEDIAN:
            return self.fedmedian(gradients)
        elif self.strategy == AggregationStrategy.KRUM:
            return self.krum(gradients)
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            return self.trimmed_mean(gradients)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


GRADIENT_AGGREGATOR = GradientAggregator()
