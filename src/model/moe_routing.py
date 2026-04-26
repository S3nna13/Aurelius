"""
moe_routing.py — Alternative MoE routing strategies beyond standard top-k.

Implements:
  - RoutingConfig dataclass
  - topk_routing
  - expert_choice_routing
  - compute_router_z_loss
  - compute_load_balance_loss
  - Router (nn.Module)
  - SwitchRouter (nn.Module)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RoutingConfig:
    n_experts: int = 8
    top_k: int = 2
    router_type: str = "topk"
    noise_std: float = 0.0
    capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def topk_routing(
    logits: Tensor,
    k: int,
    noise_std: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Top-k routing.

    Args:
        logits: (S, n_experts) where S = B*T
        k: number of experts per token
        noise_std: if > 0, add Gaussian noise to logits before selection

    Returns:
        expert_indices: (S, k) — selected expert indices
        gates: (S, k) — softmax weights over the selected top-k logits
    """
    if noise_std > 0.0:
        logits = logits + torch.randn_like(logits) * noise_std

    # Select top-k
    topk_logits, expert_indices = torch.topk(logits, k, dim=-1)  # (S, k)

    # Softmax over just the selected k logits
    gates = F.softmax(topk_logits, dim=-1)  # (S, k)

    return expert_indices, gates


def expert_choice_routing(
    logits: Tensor,
    capacity: int,
) -> tuple[Tensor, Tensor]:
    """Expert-choice routing: each expert independently selects its top-capacity tokens.

    Args:
        logits: (S, n_experts) where S = B*T
        capacity: number of tokens each expert selects

    Returns:
        token_indices: (n_experts, capacity) — indices of selected tokens per expert
        gates: (n_experts, capacity) — softmax gate values per expert
    """
    S, n_experts = logits.shape

    # Transpose so we operate per expert: (n_experts, S)
    expert_logits = logits.t()  # (n_experts, S)

    # Each expert picks its top-capacity tokens
    cap = min(capacity, S)
    topk_logits, token_indices = torch.topk(expert_logits, cap, dim=-1)  # (n_experts, cap)

    # Softmax over the selected tokens for each expert
    gates = F.softmax(topk_logits, dim=-1)  # (n_experts, cap)

    return token_indices, gates


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------


def compute_router_z_loss(logits: Tensor, coef: float = 1e-3) -> Tensor:
    """Z-loss to prevent router logit collapse.

    z_loss = coef * mean( log(sum(exp(logits)))^2 )

    Args:
        logits: (S, n_experts)
        coef: scaling coefficient

    Returns:
        scalar Tensor
    """
    # log(sum(exp(logits))) = logsumexp along expert dim
    log_z = torch.logsumexp(logits, dim=-1)  # (S,)
    loss = coef * (log_z**2).mean()
    return loss


def compute_load_balance_loss(
    gates: Tensor,
    n_experts: int,
    coef: float = 0.01,
) -> Tensor:
    """Load balancing auxiliary loss.

    loss = coef * n_experts * sum_i( f_i * p_i )

    where:
        f_i = fraction of tokens routed to expert i (based on argmax / top selection)
        p_i = mean gate probability for expert i across all tokens

    Args:
        gates: (S, n_experts) — full softmax probability distribution over experts
               (not just top-k; pass the full softmax of logits)
        n_experts: number of experts
        coef: scaling coefficient

    Returns:
        scalar Tensor
    """
    S = gates.shape[0]

    # p_i: mean gate probability per expert
    p = gates.mean(dim=0)  # (n_experts,)

    # f_i: fraction of tokens whose argmax selects expert i
    expert_ids = gates.argmax(dim=-1)  # (S,)
    counts = torch.zeros(n_experts, device=gates.device, dtype=gates.dtype)
    counts.scatter_add_(0, expert_ids, torch.ones(S, device=gates.device, dtype=gates.dtype))
    f = counts / S  # (n_experts,)

    loss = coef * n_experts * (f * p).sum()
    return loss


# ---------------------------------------------------------------------------
# Router module
# ---------------------------------------------------------------------------


class Router(nn.Module):
    """Standard top-k router with auxiliary losses.

    forward(x) returns:
        expert_indices: (B*T, top_k)
        gates: (B*T, top_k)
        aux_loss: scalar — router_z_loss + load_balance_loss
    """

    def __init__(self, d_model: int, config: RoutingConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(d_model, config.n_experts, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            expert_indices: (B*T, top_k)
            gates: (B*T, top_k)
            aux_loss: scalar Tensor
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)  # (S, D)

        logits = self.proj(x_flat)  # (S, n_experts)

        # Routing
        expert_indices, gates = topk_routing(
            logits,
            k=self.config.top_k,
            noise_std=self.config.noise_std,
        )

        # Auxiliary losses
        z_loss = compute_router_z_loss(logits, coef=self.config.aux_loss_coef)

        full_probs = F.softmax(logits, dim=-1)  # (S, n_experts)
        lb_loss = compute_load_balance_loss(
            full_probs,
            n_experts=self.config.n_experts,
            coef=self.config.aux_loss_coef,
        )

        aux_loss = z_loss + lb_loss

        return expert_indices, gates, aux_loss


# ---------------------------------------------------------------------------
# SwitchRouter module
# ---------------------------------------------------------------------------


class SwitchRouter(nn.Module):
    """Switch Transformer-style top-1 router with capacity-based token dropping.

    forward(x) returns:
        expert_idx: (B*T,) — selected expert index per token
        gates: (B*T,) — gate value per token (1.0 for kept tokens, 0.0 if dropped)
        aux_loss: scalar Tensor
    """

    def __init__(self, d_model: int, config: RoutingConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(d_model, config.n_experts, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            expert_idx: (B*T,)
            gates: (B*T,) — gate probabilities; 0 for capacity-dropped tokens
            aux_loss: scalar Tensor
        """
        B, T, D = x.shape
        S = B * T
        x_flat = x.view(S, D)

        logits = self.proj(x_flat)  # (S, n_experts)
        probs = F.softmax(logits, dim=-1)  # (S, n_experts)

        # Top-1 selection
        gate_vals, expert_idx = probs.max(dim=-1)  # (S,), (S,)

        # Capacity-based dropping: capacity = ceil(S / n_experts * capacity_factor)
        n_experts = self.config.n_experts
        capacity = math.ceil(S / n_experts * self.config.capacity_factor)

        # For each expert, keep only the first `capacity` tokens (by position order)
        # Tokens beyond capacity are dropped (gate set to 0)
        kept_gates = gate_vals.clone()
        for e in range(n_experts):
            mask = (expert_idx == e).nonzero(as_tuple=False).squeeze(1)
            if mask.numel() > capacity:
                dropped = mask[capacity:]
                kept_gates[dropped] = 0.0

        # Auxiliary losses
        z_loss = compute_router_z_loss(logits, coef=self.config.aux_loss_coef)
        lb_loss = compute_load_balance_loss(
            probs,
            n_experts=n_experts,
            coef=self.config.aux_loss_coef,
        )
        aux_loss = z_loss + lb_loss

        return expert_idx, kept_gates, aux_loss
