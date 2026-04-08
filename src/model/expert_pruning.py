"""Helpers for pruning under-utilized experts in MoE layers."""

from __future__ import annotations

import torch


def expert_importance(router_probs: torch.Tensor) -> torch.Tensor:
    """Average routing mass assigned to each expert."""
    if router_probs.dim() < 2:
        raise ValueError("router_probs must have an expert dimension")
    return router_probs.reshape(-1, router_probs.size(-1)).mean(dim=0)


def prune_mask(router_probs: torch.Tensor, keep_k: int) -> torch.Tensor:
    """Return a boolean mask keeping the top-k experts by importance."""
    importance = expert_importance(router_probs)
    if keep_k <= 0 or keep_k > importance.numel():
        raise ValueError(f"keep_k must be in [1, {importance.numel()}], got {keep_k}")
    topk = torch.topk(importance, k=keep_k).indices
    mask = torch.zeros_like(importance, dtype=torch.bool)
    mask[topk] = True
    return mask


def apply_expert_pruning(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero pruned experts in a leading expert dimension tensor."""
    if weights.size(0) != mask.numel():
        raise ValueError("weights leading dimension must match mask length")
    return weights * mask.view(-1, *([1] * (weights.dim() - 1))).to(dtype=weights.dtype)


def retained_fraction(mask: torch.Tensor) -> torch.Tensor:
    """Fraction of experts retained by pruning."""
    return mask.float().mean()
