"""Model pruner: magnitude pruning, structured pruning, pruning schedule."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

_rng = random.Random(42)


class PruningMethod(str, Enum):
    MAGNITUDE = "magnitude"
    RANDOM = "random"
    STRUCTURED_HEAD = "structured_head"
    STRUCTURED_LAYER = "structured_layer"


@dataclass
class PruningMask:
    layer_name: str
    mask: list[float]  # 1.0 = keep, 0.0 = prune
    sparsity: float
    method: PruningMethod


class ModelPruner:
    def __init__(self, target_sparsity: float = 0.5) -> None:
        self.target_sparsity = target_sparsity

    def compute_mask(
        self,
        layer_name: str,
        weights: list[float],
        method: PruningMethod = PruningMethod.MAGNITUDE,
        structured_k: int = None,
    ) -> PruningMask:
        n = len(weights)
        if n == 0:
            return PruningMask(layer_name=layer_name, mask=[], sparsity=0.0, method=method)

        if method == PruningMethod.MAGNITUDE:
            keep_count = max(1, int(round(n * (1 - self.target_sparsity))))
            indexed = sorted(enumerate(weights), key=lambda x: abs(x[1]), reverse=True)
            keep_indices = set(idx for idx, _ in indexed[:keep_count])
            mask = [1.0 if i in keep_indices else 0.0 for i in range(n)]
            actual_sparsity = sum(1 for m in mask if m == 0.0) / n

        elif method == PruningMethod.RANDOM:
            indices = list(range(n))
            _rng.shuffle(indices)
            keep_count = max(1, int(round(n * (1 - self.target_sparsity))))
            keep_indices = set(indices[:keep_count])
            mask = [1.0 if i in keep_indices else 0.0 for i in range(n)]
            actual_sparsity = sum(1 for m in mask if m == 0.0) / n

        elif method == PruningMethod.STRUCTURED_HEAD:
            keep_k = structured_k if structured_k is not None else max(1, n // 2)
            keep_k = min(keep_k, n)
            mask = [1.0] * keep_k + [0.0] * (n - keep_k)
            actual_sparsity = sum(1 for m in mask if m == 0.0) / n

        elif method == PruningMethod.STRUCTURED_LAYER:
            if self.target_sparsity >= 0.8:
                mask = [0.0] * n
                actual_sparsity = 1.0
            else:
                mask = [1.0] * n
                actual_sparsity = 0.0

        else:
            mask = [1.0] * n
            actual_sparsity = 0.0

        return PruningMask(
            layer_name=layer_name,
            mask=mask,
            sparsity=actual_sparsity,
            method=method,
        )

    def apply_mask(self, weights: list[float], mask: list[float]) -> list[float]:
        return [w * m for w, m in zip(weights, mask)]

    def sparsity_schedule(
        self,
        current_step: int,
        total_steps: int,
        final_sparsity: float,
        warmup_steps: int = 100,
    ) -> float:
        if current_step < warmup_steps:
            return 0.0
        denom = total_steps - warmup_steps
        if denom <= 0:
            return final_sparsity
        progress = (current_step - warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))
        return final_sparsity * (1 - (1 - progress) ** 3)

    def global_sparsity(self, masks: list[PruningMask]) -> float:
        if not masks:
            return 0.0
        total_weights = sum(len(m.mask) for m in masks)
        if total_weights == 0:
            return 0.0
        weighted_sum = sum(m.sparsity * len(m.mask) for m in masks)
        return weighted_sum / total_weights


PRUNING_REGISTRY: dict[str, ModelPruner] = {
    "default": ModelPruner(0.5),
    "aggressive": ModelPruner(0.9),
    "light": ModelPruner(0.2),
}
