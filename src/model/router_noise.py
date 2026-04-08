"""Noisy router helpers for mixture-of-experts gating."""

from __future__ import annotations

import torch


def add_gumbel_noise(logits: torch.Tensor) -> torch.Tensor:
    """Add standard Gumbel noise to router logits."""
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-8, 1.0 - 1e-8)))
    return logits + noise


def noisy_topk_routing(
    logits: torch.Tensor,
    k: int,
    noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k experts with optional Gaussian noise."""
    if logits.dim() < 2:
        raise ValueError("logits must have an expert dimension")
    if k <= 0 or k > logits.size(-1):
        raise ValueError(f"k must be in [1, {logits.size(-1)}], got {k}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    noisy_logits = logits if noise_std == 0.0 else logits + noise_std * torch.randn_like(logits)
    values, indices = torch.topk(noisy_logits, k=k, dim=-1)
    return indices, values


def routing_mask(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Convert top-k indices to a binary expert mask."""
    if torch.any(indices < 0) or torch.any(indices >= n_experts):
        raise ValueError("indices contain out-of-range expert ids")
    mask = torch.zeros(*indices.shape[:-1], n_experts, device=indices.device, dtype=torch.bool)
    return mask.scatter(-1, indices, True)


def noisy_router_probs(logits: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
    """Compute router probabilities after optional Gaussian noise."""
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    noisy_logits = logits if noise_std == 0.0 else logits + noise_std * torch.randn_like(logits)
    return torch.softmax(noisy_logits, dim=-1)
