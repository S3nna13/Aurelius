from __future__ import annotations

from enum import Enum

import torch


class PruningStrategy(Enum):
    NEURONS = "neurons"
    HEADS = "heads"
    CHANNELS = "channels"


class StructuredPruner:
    def __init__(self, strategy: str = "neurons", amount: float = 0.2) -> None:
        try:
            self.strategy = PruningStrategy(strategy)
        except ValueError:
            raise ValueError(f"Unknown pruning strategy: {strategy}. Options: neurons, heads, channels")
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"amount must be in [0,1], got {amount}")
        self.amount = amount

    def prune(self, weight: torch.Tensor, dim: int = 0, groups: int | None = None, group_size: int | None = None) -> torch.Tensor:
        pruned = weight.clone()
        if self.amount <= 0.0:
            return pruned
        if self.amount >= 1.0:
            return torch.zeros_like(pruned)
        n = weight.shape[dim]
        k = max(1, int(n * self.amount))

        if self.strategy == PruningStrategy.HEADS and groups is not None and group_size is not None:
            views = weight.view(groups, group_size)
            norms = views.norm(dim=1)
            _, indices = torch.topk(norms, k=k, largest=False)
            mask = torch.ones(groups, dtype=torch.bool, device=weight.device)
            mask[indices] = False
            expanded = mask.repeat_interleave(group_size).reshape(weight.shape)
            pruned[~expanded] = 0.0
        elif self.strategy in (PruningStrategy.NEURONS, PruningStrategy.CHANNELS):
            reshaped = weight.view(n, -1)
            norms = reshaped.norm(dim=1)
            _, indices = torch.topk(norms, k=k, largest=False)
            slc = [slice(None)] * weight.dim()
            slc[dim] = indices
            pruned[tuple(slc)] = 0.0
        return pruned
