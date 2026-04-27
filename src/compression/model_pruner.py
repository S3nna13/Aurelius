"""Model pruner: magnitude, structured, and random weight pruning."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum


class PruningStrategy(StrEnum):
    MAGNITUDE = "magnitude"
    STRUCTURED = "structured"
    RANDOM = "random"


@dataclass
class PruningConfig:
    strategy: PruningStrategy = PruningStrategy.MAGNITUDE
    sparsity: float = 0.5
    target_modules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PruningStats:
    module_name: str
    original_params: int
    remaining_params: int
    sparsity_achieved: float


class ModelPruner:
    """Weight pruning utilities for model compression."""

    def __init__(self, config: PruningConfig | None = None, seed: int = 42) -> None:
        self.config = config or PruningConfig()
        self._rng = random.Random(seed)  # noqa: S311

    def compute_mask(
        self,
        weights: list[float],
        sparsity: float,
        strategy: PruningStrategy = PruningStrategy.MAGNITUDE,
    ) -> list[bool]:
        """Return a boolean mask where True means keep."""
        n = len(weights)
        if n == 0:
            return []
        sparsity = max(0.0, min(1.0, sparsity))
        keep_count = int(round(n * (1.0 - sparsity)))
        keep_count = max(0, min(n, keep_count))

        if strategy == PruningStrategy.MAGNITUDE:
            indexed = sorted(range(n), key=lambda i: abs(weights[i]), reverse=True)
            keep = set(indexed[:keep_count])
            return [i in keep for i in range(n)]

        if strategy == PruningStrategy.RANDOM:
            indices = list(range(n))
            self._rng.shuffle(indices)
            keep = set(indices[:keep_count])
            return [i in keep for i in range(n)]

        if strategy == PruningStrategy.STRUCTURED:
            return [i < keep_count for i in range(n)]

        return [True] * n

    def apply_mask(self, weights: list[float], mask: list[bool]) -> list[float]:
        """Zero out masked-out weights."""
        if len(weights) != len(mask):
            raise ValueError("weights and mask length mismatch")
        return [w if m else 0.0 for w, m in zip(weights, mask)]

    def prune(self, module_name: str, weights: list[float]) -> tuple[list[float], PruningStats]:
        mask = self.compute_mask(weights, self.config.sparsity, self.config.strategy)
        pruned = self.apply_mask(weights, mask)
        remaining = sum(1 for m in mask if m)
        n = len(weights)
        achieved = 0.0 if n == 0 else 1.0 - remaining / n
        stats = PruningStats(
            module_name=module_name,
            original_params=n,
            remaining_params=remaining,
            sparsity_achieved=achieved,
        )
        return pruned, stats

    def achieved_sparsity(self, weights: list[float]) -> float:
        """Fraction of zero weights."""
        n = len(weights)
        if n == 0:
            return 0.0
        zeros = sum(1 for w in weights if w == 0.0)
        return zeros / n

    def can_prune_further(
        self,
        current_sparsity: float,
        target_sparsity: float,
        tolerance: float = 0.01,
    ) -> bool:
        return current_sparsity + tolerance < target_sparsity


MODEL_PRUNER_REGISTRY: dict[str, type[ModelPruner]] = {
    "default": ModelPruner,
}
