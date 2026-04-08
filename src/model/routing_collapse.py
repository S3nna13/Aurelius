"""Regularizers for preventing MoE routing collapse."""

from __future__ import annotations

import math

import torch


def expert_load(router_probs: torch.Tensor) -> torch.Tensor:
    """Average probability mass assigned to each expert."""
    if router_probs.dim() < 2:
        raise ValueError("router_probs must have expert dimension")
    return router_probs.reshape(-1, router_probs.size(-1)).mean(dim=0)


def load_balance_loss(router_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Encourage balanced expert utilization."""
    load = expert_load(router_probs)
    n_experts = load.numel()
    uniform = torch.full_like(load, 1.0 / n_experts)
    return torch.sum(load * torch.log((load + eps) / (uniform + eps)))


def routing_entropy(router_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean token-level routing entropy."""
    probs = router_probs.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.mean()


def z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Switch Transformer-style log-sum-exp regularizer."""
    return torch.logsumexp(router_logits, dim=-1).pow(2).mean()


def collapse_score(router_probs: torch.Tensor) -> torch.Tensor:
    """Heuristic [0, 1] collapse score based on effective expert count."""
    load = expert_load(router_probs)
    effective = 1.0 / load.square().sum().clamp_min(1e-8)
    n_experts = float(load.numel())
    return 1.0 - (effective - 1.0) / max(n_experts - 1.0, 1.0)
