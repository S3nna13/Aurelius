"""Mixture-of-Experts (MoE) feedforward layer with top-k routing.

Implements sparse MoE with:
- Top-k token routing with load-balancing auxiliary loss
- Per-token expert assignment
- Expert capacity with overflow handling

References:
    Shazeer et al. 2017, "Outrageously Large Neural Networks: The Sparsely-Gated MoE"
    Fedus et al. 2021, "Switch Transformers"
    Lepikhin et al. 2021, "GShard"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MoEConfig:
    """Configuration for the Mixture-of-Experts layer."""

    d_model: int = 64
    d_ff: int = 256
    n_experts: int = 8
    top_k: int = 2  # tokens routed to top-k experts
    aux_loss_coef: float = 0.01  # load-balancing auxiliary loss weight
    expert_dropout: float = 0.0  # dropout within each expert
    capacity_factor: float = 1.25  # overflow buffer above token/expert


# ---------------------------------------------------------------------------
# Expert FFN
# ---------------------------------------------------------------------------


class ExpertFFN(nn.Module):
    """A single expert: two-layer FFN with SiLU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.silu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class TopKRouter(nn.Module):
    """Top-k gating router with load-balancing auxiliary loss.

    Computes routing probabilities and selects top-k experts per token.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute top-k routing.

        Args:
            x: (B, T, d_model) or (N, d_model) where N = B*T

        Returns:
            router_probs: (N, n_experts) — softmax probabilities
            top_k_indices: (N, top_k) — expert assignments
            top_k_weights: (N, top_k) — normalized weights for selected experts
        """
        # Flatten to (N, d_model) for routing
        shape = x.shape
        if x.dim() == 3:
            N = shape[0] * shape[1]
            x_flat = x.view(N, shape[2])
        else:
            N = shape[0]
            x_flat = x

        logits = self.gate(x_flat)  # (N, E)
        router_probs = F.softmax(logits, dim=-1)  # (N, E)

        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        return router_probs, top_k_indices, top_k_weights


# ---------------------------------------------------------------------------
# Load-balancing auxiliary loss
# ---------------------------------------------------------------------------


def compute_load_balancing_loss(
    router_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    n_experts: int,
    aux_loss_coef: float = 0.01,
) -> torch.Tensor:
    """Compute load-balancing auxiliary loss.

    L_aux = aux_loss_coef * n_experts * sum_i(f_i * P_i)

    where:
        f_i = fraction of tokens routed to expert i (using top-1 for f)
        P_i = mean router probability for expert i

    Args:
        router_probs: (N, E) softmax probabilities.
        top_k_indices: (N, top_k) expert indices.
        n_experts: Total number of experts.
        aux_loss_coef: Scaling coefficient.

    Returns:
        Scalar auxiliary loss.
    """
    router_probs.shape[0]

    # f_i: fraction of tokens assigned to expert i (use top-1 assignment)
    top1_indices = top_k_indices[:, 0]  # (N,)
    one_hot = F.one_hot(top1_indices, num_classes=n_experts).float()  # (N, E)
    f = one_hot.mean(dim=0)  # (E,)

    # P_i: mean router probability per expert
    P = router_probs.mean(dim=0)  # (E,)

    # Auxiliary loss
    aux_loss = aux_loss_coef * n_experts * (f * P).sum()
    return aux_loss


# ---------------------------------------------------------------------------
# Sparse MoE Layer
# ---------------------------------------------------------------------------


class SparseMoELayer(nn.Module):
    """Sparse Top-k Mixture-of-Experts feedforward layer.

    Replaces a standard FFN with E parallel expert FFNs,
    routing each token to its top-k experts.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.router = TopKRouter(config.d_model, config.n_experts, config.top_k)
        self.experts = nn.ModuleList(
            [
                ExpertFFN(config.d_model, config.d_ff, config.expert_dropout)
                for _ in range(config.n_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts and aggregate outputs.

        Args:
            x: (B, T, d_model) hidden states.

        Returns:
            (output, aux_loss):
                output: (B, T, d_model)
                aux_loss: scalar load-balancing loss
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Route
        router_probs, top_k_indices, top_k_weights = self.router(x)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Dispatch: for each expert, collect its tokens and process
        for expert_idx in range(self.config.n_experts):
            # Find which tokens are routed to this expert (in any top-k slot)
            expert_mask = top_k_indices == expert_idx  # (N, top_k) bool
            token_mask = expert_mask.any(dim=-1)  # (N,)

            if not token_mask.any():
                continue

            # Tokens for this expert
            expert_tokens = x_flat[token_mask]  # (n_e, D)

            # Expert forward
            expert_out = self.experts[expert_idx](expert_tokens)  # (n_e, D)

            # Weights: sum over top-k slots where this expert is selected
            weights = (expert_mask[token_mask].float() * top_k_weights[token_mask]).sum(
                dim=-1, keepdim=True
            )
            # (n_e, top_k) * (n_e, top_k) → sum → (n_e, 1)

            output[token_mask] = output[token_mask] + weights * expert_out

        output = output.view(B, T, D)

        # Compute load-balancing auxiliary loss
        aux_loss = compute_load_balancing_loss(
            router_probs, top_k_indices, self.config.n_experts, self.config.aux_loss_coef
        )

        return output, aux_loss


# ---------------------------------------------------------------------------
# MoE Transformer Block (drop-in replacement for FFN sublayer)
# ---------------------------------------------------------------------------


class MoEBlock(nn.Module):
    """Transformer block with MoE FFN + RMSNorm + residual."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.moe = SparseMoELayer(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (x + moe_output, aux_loss)
        """
        normed = self.norm(x)
        moe_out, aux_loss = self.moe(normed)
        return x + moe_out, aux_loss
