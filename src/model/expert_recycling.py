"""Helpers for recycling inactive experts back into active use."""

from __future__ import annotations

import torch


def inactive_expert_mask(usage: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Mask experts whose usage is at or below a threshold."""
    if usage.dim() != 1:
        raise ValueError("usage must be 1D")
    return usage <= threshold


def recycle_expert_weights(
    expert_weights: torch.Tensor,
    inactive_mask: torch.Tensor,
    source_expert: int,
    noise_scale: float = 0.0,
) -> torch.Tensor:
    """Copy one source expert into inactive experts, with optional noise."""
    if expert_weights.size(0) != inactive_mask.numel():
        raise ValueError("expert_weights leading dimension must match inactive_mask")
    if source_expert < 0 or source_expert >= expert_weights.size(0):
        raise ValueError("source_expert out of range")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    updated = expert_weights.clone()
    source = expert_weights[source_expert]
    for idx, inactive in enumerate(inactive_mask.tolist()):
        if inactive:
            noise = noise_scale * torch.randn_like(source) if noise_scale > 0 else 0.0
            updated[idx] = source + noise
    return updated


def recycled_fraction(inactive_mask: torch.Tensor) -> torch.Tensor:
    """Fraction of experts selected for recycling."""
    if inactive_mask.numel() == 0:
        return torch.tensor(0.0)
    return inactive_mask.float().mean()
