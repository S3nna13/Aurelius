"""Soft Mixture-of-Experts with continuous routing (no discrete token assignment).

Unlike sparse MoE (top-k hard routing) or expert-choice routing, Soft MoE uses
fully-differentiable slot-based dispatch: every token contributes to every expert
slot via a softmax over tokens, and every token aggregates from every slot via
a softmax over slots. There is no token dropping, no capacity buffers, and no
auxiliary load-balance loss required — load is enforced implicitly by the
softmax normalisation.

Reference: Puigcerver et al. (2023) "From Sparse to Soft Mixtures of Experts"
https://arxiv.org/abs/2308.00951
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SoftMoEConfig:
    """Configuration for a Soft MoE layer."""

    n_experts: int = 8
    n_slots: int = 1  # number of processing slots per expert
    d_model: int = 64
    d_ff: int = 256
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Expert FFN
# ---------------------------------------------------------------------------


class ExpertFFN(nn.Module):
    """Standard feed-forward expert: Linear → GELU → Dropout → Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., d_model) → (..., d_model)."""
        return self.fc2(self.drop(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# Soft Router
# ---------------------------------------------------------------------------


class SoftRouter(nn.Module):
    """Computes continuous dispatch and combine weights via learned slot embeddings.

    Each expert has *n_slots* slot embeddings; routing is fully soft:
    - dispatch weights  = softmax over *tokens*  (dim=1) — how much each token
      feeds into a given slot.
    - combine weights   = softmax over *slots*   (dim=2) — how much each slot
      contributes back to a given token position.
    """

    def __init__(self, d_model: int, n_experts: int, n_slots: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_slots = n_slots
        total_slots = n_experts * n_slots
        # Learnable slot embeddings — shape (total_slots, d_model)
        self.slots = nn.Parameter(torch.empty(total_slots, d_model))
        nn.init.normal_(self.slots, std=d_model**-0.5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, D)

        Returns:
            dispatch_weights: (B, T, E*S) — softmax over tokens (dim=1)
            combine_weights:  (B, T, E*S) — softmax over slots  (dim=2)
        """
        # logits: (B, T, E*S)
        logits = x @ self.slots.T  # (B, T, D) x (D, E*S) → (B, T, E*S)

        # Dispatch: how much does each *token* contribute to each slot?
        # Normalise over the token dimension so each slot's inputs sum to 1.
        dispatch_weights = F.softmax(logits, dim=1)

        # Combine: how much does each *slot* contribute back to each token?
        # Normalise over the slot dimension so each token's output sums to 1.
        combine_weights = F.softmax(logits, dim=2)

        return dispatch_weights, combine_weights


# ---------------------------------------------------------------------------
# Soft MoE Layer
# ---------------------------------------------------------------------------


class SoftMoELayer(nn.Module):
    """Full Soft MoE layer: route → process via experts → aggregate.

    All routing is differentiable; gradients flow through both the dispatch
    and combine weight paths as well as through the expert parameters.
    """

    def __init__(self, config: SoftMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                ExpertFFN(config.d_model, config.d_ff, config.dropout)
                for _ in range(config.n_experts)
            ]
        )
        self.router = SoftRouter(config.d_model, config.n_experts, config.n_slots)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D)

        Returns:
            out: (B, T, D) — weighted sum of all expert contributions
        """
        dispatch_weights, combine_weights = self.router(x)  # each (B, T, E*S)

        n_slots = self.config.n_slots
        out = torch.zeros_like(x)  # (B, T, D)

        for e, expert in enumerate(self.experts):
            s = e * n_slots
            t = s + n_slots

            # dispatch slice: (B, T, n_slots)
            d = dispatch_weights[:, :, s:t]
            c = combine_weights[:, :, s:t]

            # Slot inputs: aggregate tokens → slots
            # d.transpose(1,2): (B, n_slots, T)  @  x: (B, T, D) → (B, n_slots, D)
            slot_input = d.transpose(1, 2) @ x  # (B, n_slots, D)

            # Run each slot input through the expert
            slot_output = expert(slot_input)  # (B, n_slots, D)

            # Aggregate slots → tokens weighted by combine weights
            # c: (B, T, n_slots)  @  slot_output: (B, n_slots, D) → (B, T, D)
            out = out + c @ slot_output  # (B, T, D)

        return out


# ---------------------------------------------------------------------------
# Load statistics
# ---------------------------------------------------------------------------


def compute_load_stats(dispatch_weights: Tensor) -> dict:
    """Compute per-slot load statistics from dispatch weights.

    Args:
        dispatch_weights: (B, T, E*S) — softmax-normalised dispatch weights
            produced by SoftRouter (each column already sums to 1 over tokens).

    Returns:
        dict with keys:
            "expert_load"       — Tensor shape (E*S,): mean token mass per slot
            "load_balance_loss" — scalar Tensor: variance of loads (≥ 0)
            "max_load"          — float: maximum load across all slots
            "min_load"          — float: minimum load across all slots
    """
    # Sum dispatch weights over batch and token dimensions → (E*S,)
    # This gives total "mass" routed to each slot across the batch.
    expert_load = dispatch_weights.sum(dim=(0, 1))  # (E*S,)

    # Variance of loads as a differentiable balance penalty
    load_balance_loss = expert_load.var()

    return {
        "expert_load": expert_load,
        "load_balance_loss": load_balance_loss,
        "max_load": expert_load.max().item(),
        "min_load": expert_load.min().item(),
    }
