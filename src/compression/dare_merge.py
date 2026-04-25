"""DARE model merging: Drop And REscale (arXiv 2311.03099)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DAREConfig:
    drop_rate: float = 0.9
    rescale: bool = True
    seed: int = 42


@dataclass
class MergeResult:
    merged_state_dict: dict[str, torch.Tensor]
    n_params_merged: int
    drop_rate_applied: float


class DAREMerger:
    """DARE: random sparsification of task vectors before merging."""

    def __init__(self, config: DAREConfig | None = None) -> None:
        self.config = config or DAREConfig()

    def compute_task_vector(
        self,
        base: dict[str, torch.Tensor],
        finetuned: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {k: finetuned[k] - base[k] for k in base}

    def sparsify(
        self, task_vector: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self.config.seed)
        result: dict[str, torch.Tensor] = {}
        for k, t in task_vector.items():
            mask = torch.bernoulli(
                torch.full(t.shape, 1.0 - self.config.drop_rate), generator=rng
            ).to(dtype=t.dtype)
            sparse = t * mask
            if self.config.rescale and self.config.drop_rate < 1.0:
                sparse = sparse / (1.0 - self.config.drop_rate)
            result[k] = sparse
        return result

    def merge(
        self,
        base: dict[str, torch.Tensor],
        finetuned_models: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> MergeResult:
        if not finetuned_models:
            raise ValueError("finetuned_models must be non-empty")
        n = len(finetuned_models)
        if weights is None:
            weights = [1.0 / n] * n
        if len(weights) != n:
            raise ValueError("len(weights) must equal len(finetuned_models)")

        task_vectors = [
            self.compute_task_vector(base, ft) for ft in finetuned_models
        ]
        sparse_vectors = [self.sparsify(tv) for tv in task_vectors]

        merged: dict[str, torch.Tensor] = {}
        for k in base:
            combined = torch.zeros_like(base[k])
            for sv, w in zip(sparse_vectors, weights):
                combined = combined + w * sv[k]
            merged[k] = base[k] + combined

        n_params = sum(t.numel() for t in merged.values())
        return MergeResult(
            merged_state_dict=merged,
            n_params_merged=n_params,
            drop_rate_applied=self.config.drop_rate,
        )
