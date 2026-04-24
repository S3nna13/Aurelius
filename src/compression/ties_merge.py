"""TIES model merging: Trim, Elect Sign, Merge (arXiv 2306.01708)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .dare_merge import MergeResult


@dataclass
class TIESConfig:
    trim_ratio: float = 0.2
    elect_sign: str = "majority"


class TIESMerger:
    """TIES merging: trim small deltas, resolve sign conflicts, merge."""

    def __init__(self, config: TIESConfig | None = None) -> None:
        self.config = config or TIESConfig()
        if self.config.elect_sign not in ("majority", "magnitude_weighted"):
            raise ValueError("elect_sign must be 'majority' or 'magnitude_weighted'")

    def compute_task_vectors(
        self,
        base: dict[str, torch.Tensor],
        models: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        return [{k: m[k] - base[k] for k in base} for m in models]

    def trim(
        self, task_vectors: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        trimmed: list[dict[str, torch.Tensor]] = []
        for tv in task_vectors:
            new_tv: dict[str, torch.Tensor] = {}
            for k, t in tv.items():
                flat = t.abs().flatten()
                if flat.numel() == 0:
                    new_tv[k] = t.clone()
                    continue
                k_keep = max(1, int(flat.numel() * (1.0 - self.config.trim_ratio)))
                threshold = flat.kthvalue(flat.numel() - k_keep + 1).values
                mask = t.abs() >= threshold
                new_tv[k] = t * mask.to(dtype=t.dtype)
            trimmed.append(new_tv)
        return trimmed

    def elect_sign(
        self, trimmed_vectors: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        elected: dict[str, torch.Tensor] = {}
        for k in trimmed_vectors[0]:
            stacked = torch.stack([tv[k] for tv in trimmed_vectors], dim=0)
            if self.config.elect_sign == "majority":
                sign_sum = stacked.sign().sum(dim=0)
                s = sign_sum.sign()
                s[s == 0] = 1.0
                elected[k] = s
            else:
                weighted_sum = stacked.sum(dim=0)
                s = weighted_sum.sign()
                s[s == 0] = 1.0
                elected[k] = s
        return elected

    def disjoint_merge(
        self,
        trimmed_vectors: list[dict[str, torch.Tensor]],
        elected_signs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        for k in elected_signs:
            es = elected_signs[k]
            aligned_sum = torch.zeros_like(es)
            counts = torch.zeros_like(es)
            for tv in trimmed_vectors:
                t = tv[k]
                agrees = (t.sign() == es) | (t == 0)
                contributing = t * agrees.to(dtype=t.dtype)
                aligned_sum = aligned_sum + contributing
                counts = counts + agrees.to(dtype=t.dtype).abs()
            safe_counts = counts.clone()
            safe_counts[safe_counts == 0] = 1.0
            merged[k] = aligned_sum / safe_counts
        return merged

    def merge(
        self,
        base: dict[str, torch.Tensor],
        models: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> MergeResult:
        if not models:
            raise ValueError("models must be non-empty")

        task_vectors = self.compute_task_vectors(base, models)
        trimmed = self.trim(task_vectors)
        elected = self.elect_sign(trimmed)
        delta = self.disjoint_merge(trimmed, elected)

        merged_state: dict[str, torch.Tensor] = {
            k: base[k] + delta[k] for k in base
        }
        n_params = sum(t.numel() for t in merged_state.values())
        return MergeResult(
            merged_state_dict=merged_state,
            n_params_merged=n_params,
            drop_rate_applied=self.config.trim_ratio,
        )
