"""Route-dropout helpers for expert routing regularization."""

from __future__ import annotations

import torch


def route_dropout_mask(shape: torch.Size | tuple[int, ...], drop_prob: float, device=None) -> torch.Tensor:
    """Sample a Bernoulli keep mask for router probabilities."""
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
    keep_prob = 1.0 - drop_prob
    return torch.bernoulli(torch.full(shape, keep_prob, device=device)).to(dtype=torch.bool)


def apply_route_dropout(router_probs: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Drop and renormalize router probabilities."""
    mask = route_dropout_mask(router_probs.shape, drop_prob, device=router_probs.device)
    dropped = router_probs * mask.to(dtype=router_probs.dtype)
    denom = dropped.sum(dim=-1, keepdim=True)
    fallback = torch.full_like(router_probs, 1.0 / router_probs.size(-1))
    return torch.where(denom > 0, dropped / denom.clamp_min(1e-8), fallback)


def route_survival_rate(mask: torch.Tensor) -> torch.Tensor:
    """Fraction of routes kept by dropout."""
    return mask.float().mean()

