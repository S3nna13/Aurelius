"""Expert Choice routing for MoE (Zhou et al., 2022) — experts select top-k tokens."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ExpertChoiceConfig:
    n_experts: int = 4
    capacity_factor: float = 1.0    # tokens per expert = capacity_factor * T / n_experts
    d_model: int = 64
    d_expert: int = 128             # expert hidden dim (typically 2-4x d_model)
    use_aux_loss: bool = True
    aux_loss_coeff: float = 0.01


def compute_expert_capacity(
    n_tokens: int,
    n_experts: int,
    capacity_factor: float,
) -> int:
    """Capacity = ceil(capacity_factor * n_tokens / n_experts). Minimum 1."""
    return max(1, math.ceil(capacity_factor * n_tokens / n_experts))


def expert_choice_routing(
    router_logits: Tensor,      # (T, n_experts) — token scores per expert
    capacity: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Expert-choice routing: each expert picks top-capacity tokens.

    1. router_probs = softmax(router_logits, dim=0) — normalize over tokens
    2. For each expert e: select top-capacity tokens by router_probs[:, e]
    3. Returns:
       - expert_mask: (T, n_experts) binary, 1 if expert e chose token t
       - token_indices: (n_experts, capacity) indices of chosen tokens per expert
       - expert_weights: (n_experts, capacity) softmax weights for chosen tokens
    """
    T, n_experts = router_logits.shape

    # Normalize over tokens (dim=0), not over experts
    router_probs = F.softmax(router_logits, dim=0)  # (T, n_experts)

    # Transpose to (n_experts, T) so we can topk over tokens per expert
    scores = router_probs.T  # (n_experts, T)

    # Each expert selects top-capacity tokens
    expert_weights, token_indices = torch.topk(scores, capacity, dim=-1)  # both (n_experts, capacity)

    # Build binary expert_mask (T, n_experts)
    expert_mask = torch.zeros(T, n_experts, dtype=router_logits.dtype, device=router_logits.device)
    # Scatter ones at selected positions
    expert_mask.scatter_(0, token_indices.T, 1.0)  # token_indices.T is (capacity, n_experts)

    return expert_mask, token_indices, expert_weights


def compute_ec_router_loss(
    router_probs: Tensor,       # (T, n_experts)
    expert_mask: Tensor,        # (T, n_experts)
) -> Tensor:
    """Auxiliary loss to encourage diverse routing.
    loss = mean(fraction_chosen_per_expert^2) — should be ~1/n_experts^2 when balanced.
    Returns scalar.
    """
    T = router_probs.shape[0]
    # fraction of tokens chosen per expert: (n_experts,)
    fraction_chosen = expert_mask.sum(dim=0) / T  # (n_experts,)
    loss = (fraction_chosen ** 2).mean()
    return loss


class ExpertChoiceFFN(nn.Module):
    """Single expert FFN (Linear -> GELU -> Linear)."""

    def __init__(self, d_model: int, d_expert: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_expert)
        self.fc2 = nn.Linear(d_expert, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch_tokens, d_model) -> (batch_tokens, d_model)"""
        return self.fc2(F.gelu(self.fc1(x)))


class ExpertChoiceLayer(nn.Module):
    """Full expert-choice MoE layer."""

    def __init__(self, cfg: ExpertChoiceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertChoiceFFN(cfg.d_model, cfg.d_expert)
            for _ in range(cfg.n_experts)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """x: (B, T, D).

        1. Flatten to (B*T, D), compute router logits
        2. expert_choice_routing to get selected tokens per expert
        3. For each expert: gather selected tokens, run ExpertChoiceFFN, scatter back
        4. Accumulate weighted outputs
        5. Reshape to (B, T, D)

        Returns (output, {"aux_loss": scalar, "mean_capacity_util": float_tensor})
        """
        B, T, D = x.shape
        cfg = self.cfg

        # Flatten to (N, D)
        x_flat = x.reshape(B * T, D)  # (N, D)
        N = B * T

        # Router logits
        router_logits = self.router(x_flat)  # (N, n_experts)

        # Compute capacity
        capacity = compute_expert_capacity(N, cfg.n_experts, cfg.capacity_factor)

        # Expert-choice routing
        expert_mask, token_indices, expert_weights = expert_choice_routing(router_logits, capacity)
        # expert_mask: (N, n_experts)
        # token_indices: (n_experts, capacity)
        # expert_weights: (n_experts, capacity)

        # Initialize output buffer
        output_buffer = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for e, expert in enumerate(self.experts):
            idx = token_indices[e]           # (capacity,)
            w = expert_weights[e]            # (capacity,)
            tokens = x_flat[idx]             # (capacity, D)
            expert_out = expert(tokens)      # (capacity, D)
            # Weighted accumulate
            output_buffer.index_add_(0, idx, w.unsqueeze(-1) * expert_out)

        # Reshape back
        output = output_buffer.reshape(B, T, D)

        # Compute auxiliary loss
        router_probs = F.softmax(router_logits, dim=0)  # (N, n_experts) — over tokens
        aux_loss = compute_ec_router_loss(router_probs, expert_mask)

        # Mean capacity utilization
        mean_cap_util = self.compute_utilization(expert_mask)

        aux_info: dict[str, Tensor] = {
            "aux_loss": aux_loss,
            "mean_capacity_util": mean_cap_util,
        }

        return output, aux_info

    def compute_utilization(self, expert_mask: Tensor) -> Tensor:
        """Mean fraction of capacity used per expert. Shape scalar."""
        # expert_mask: (T, n_experts) — each col sums to capacity
        # utilization = mean over experts of (tokens chosen / total tokens)
        T = expert_mask.shape[0]
        per_expert_frac = expert_mask.sum(dim=0) / T  # (n_experts,)
        return per_expert_frac.mean()


class ExpertChoiceTransformerBlock(nn.Module):
    """Transformer block replacing FFN with ExpertChoiceLayer."""

    def __init__(self, cfg: ExpertChoiceConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, num_heads=2, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.moe = ExpertChoiceLayer(cfg)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """Pre-norm, attention + residual, pre-norm, MoE + residual.
        Returns (output, aux_info).
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm MoE
        normed2 = self.norm2(x)
        moe_out, aux_info = self.moe(normed2)
        x = x + moe_out

        return x, aux_info
