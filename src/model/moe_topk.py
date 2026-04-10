"""MoE with top-p routing: select fewest experts whose probability sum >= p."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MoETopPConfig:
    n_experts: int = 8
    d_model: int = 256
    d_ff: int = 512
    top_p: float = 0.9          # select experts until cumulative prob >= top_p
    min_experts: int = 1        # always use at least this many experts
    max_experts: int = 4        # cap at this many experts
    aux_loss_coeff: float = 0.01


class ExpertFFN(nn.Module):
    """Single expert: linear -> gelu -> linear."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., d_model) -> (..., d_model)"""
        return self.fc2(F.gelu(self.fc1(x)))


def top_p_routing(
    router_logits: Tensor,  # (B*T, n_experts)
    top_p: float,
    min_experts: int = 1,
    max_experts: int = 4,
) -> tuple[Tensor, Tensor]:
    """Top-p routing: for each token, select the fewest experts whose softmax
    probability sum >= top_p (or until max_experts is reached).

    Returns:
        selected_weights: (B*T, max_experts) — routing weights (0 for unused slots)
        selected_indices: (B*T, max_experts) — expert indices (-1 for unused slots)
    """
    N, n_experts = router_logits.shape
    device = router_logits.device
    dtype = router_logits.dtype

    probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)

    # Sort experts by probability descending for each token
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    # Limit to max_experts candidates
    sorted_probs = sorted_probs[:, :max_experts]      # (N, max_experts)
    sorted_indices = sorted_indices[:, :max_experts]  # (N, max_experts)

    # Cumulative sum to determine cutoff
    cumsum = torch.cumsum(sorted_probs, dim=-1)  # (N, max_experts)

    # For each token, determine how many experts to keep:
    # keep slot k if: (k < min_experts) OR (cumsum[k-1] < top_p AND k < max_experts)
    # Equivalently: include slot k if cumsum of the PREVIOUS slot was < top_p,
    # but always include at least min_experts.
    # We build a boolean mask of shape (N, max_experts).
    slot_indices = torch.arange(max_experts, device=device).unsqueeze(0)  # (1, max_experts)

    # shifted cumsum: how much probability was covered BEFORE this slot
    shifted_cumsum = torch.cat(
        [torch.zeros(N, 1, device=device, dtype=dtype), cumsum[:, :-1]], dim=-1
    )  # (N, max_experts)

    # A slot is active if:
    #   - slot index < min_experts (always keep), OR
    #   - shifted_cumsum < top_p (previous experts haven't reached threshold)
    keep_mask = (slot_indices < min_experts) | (shifted_cumsum < top_p)  # (N, max_experts)

    # Apply mask: zero out weights for inactive slots, set indices to -1
    selected_weights = sorted_probs * keep_mask.to(dtype)  # (N, max_experts)
    selected_indices = sorted_indices.clone()
    selected_indices[~keep_mask] = -1

    # Renormalize the kept weights so they sum to 1 per token
    weight_sum = selected_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    selected_weights = selected_weights / weight_sum
    # Zero out unused slots after renorm (mask to be safe)
    selected_weights = selected_weights * keep_mask.to(dtype)

    return selected_weights, selected_indices


def top_p_aux_loss(
    router_logits: Tensor,
    selected_indices: Tensor,
    n_experts: int,
) -> Tensor:
    """Load balancing loss: penalize routing imbalance.
    fraction_routed_to_expert[i] should be uniform = 1/n_experts.
    Returns scalar.
    """
    N = router_logits.shape[0]
    device = router_logits.device
    dtype = router_logits.dtype

    probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)

    # Build expert assignment mask from selected_indices (ignore -1 slots)
    expert_mask = torch.zeros(N, n_experts, device=device, dtype=dtype)
    valid_mask = selected_indices >= 0  # (N, max_experts)
    valid_indices = selected_indices.clone()
    valid_indices[~valid_mask] = 0  # temporary safe index; we'll zero out below

    for k in range(selected_indices.shape[1]):
        slot_valid = valid_mask[:, k]  # (N,)
        idx = valid_indices[:, k]      # (N,)
        one_hot = F.one_hot(idx, num_classes=n_experts).to(dtype)  # (N, n_experts)
        expert_mask += one_hot * slot_valid.to(dtype).unsqueeze(-1)

    # Fraction of total selections going to each expert (discrete)
    total_selections = expert_mask.sum().clamp(min=1.0)
    f_i = expert_mask.sum(dim=0) / total_selections  # (n_experts,)

    # Mean router probability for each expert (differentiable)
    P_i = probs.mean(dim=0)  # (n_experts,)

    # Switch Transformer style: n_experts * sum(f_i * P_i)
    # Always >= 0 because f_i >= 0 and P_i >= 0
    loss = n_experts * (f_i * P_i).sum()
    return loss


class MoETopPLayer(nn.Module):
    """MoE layer with top-p routing."""

    def __init__(self, config: MoETopPConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, config.d_ff)
            for _ in range(config.n_experts)
        ])
        # Cache for routing_stats()
        self._last_selected_indices: Tensor | None = None
        self._last_n_tokens: int = 0

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        x: (B, T, d_model)
        Returns: (output (B, T, d_model), aux_loss scalar)
        """
        B, T, d_model = x.shape
        N = B * T
        x_flat = x.reshape(N, d_model)

        # Compute router logits
        router_logits = self.router(x_flat)  # (N, n_experts)

        # Top-p routing
        selected_weights, selected_indices = top_p_routing(
            router_logits,
            top_p=self.config.top_p,
            min_experts=self.config.min_experts,
            max_experts=self.config.max_experts,
        )
        # selected_weights: (N, max_experts), selected_indices: (N, max_experts)

        # Compute output as weighted sum of expert outputs
        output_flat = torch.zeros(N, d_model, device=x.device, dtype=x.dtype)

        for e in range(self.config.n_experts):
            # Find tokens assigned to expert e
            is_assigned = (selected_indices == e)  # (N, max_experts) bool
            token_mask = is_assigned.any(dim=-1)   # (N,) bool

            if not token_mask.any():
                continue

            token_inputs = x_flat[token_mask]          # (M, d_model)
            expert_out = self.experts[e](token_inputs)  # (M, d_model)

            # Gather per-token weight for expert e
            # For each assigned token, sum the weights where index == e
            weights_for_e = (selected_weights * is_assigned.to(selected_weights.dtype)).sum(dim=-1)
            # (N,) — 0 for non-assigned tokens
            w = weights_for_e[token_mask].unsqueeze(-1)  # (M, 1)
            output_flat[token_mask] += w * expert_out

        output = output_flat.reshape(B, T, d_model)

        # Auxiliary loss
        aux_loss = self.config.aux_loss_coeff * top_p_aux_loss(
            router_logits, selected_indices, self.config.n_experts
        )

        # Cache for stats
        self._last_selected_indices = selected_indices.detach()
        self._last_n_tokens = N

        return output, aux_loss

    def routing_stats(self) -> dict:
        """Return dict with: mean_experts_used, expert_utilization (list of fractions)."""
        if self._last_selected_indices is None:
            return {
                "mean_experts_used": 0.0,
                "expert_utilization": [0.0] * self.config.n_experts,
            }

        indices = self._last_selected_indices  # (N, max_experts)
        N = self._last_n_tokens
        n_experts = self.config.n_experts

        # Count how many active (non -1) slots each token uses
        active = (indices >= 0).sum(dim=-1).float()  # (N,)
        mean_experts_used = active.mean().item()

        # Count how many tokens each expert handles
        counts = torch.zeros(n_experts, device=indices.device, dtype=torch.float)
        for k in range(indices.shape[1]):
            slot = indices[:, k]
            valid = slot >= 0
            if valid.any():
                valid_idx = slot[valid]
                counts.scatter_add_(0, valid_idx, torch.ones(valid_idx.shape[0], device=indices.device))

        total = counts.sum().clamp(min=1.0)
        utilization = (counts / total).tolist()

        return {
            "mean_experts_used": mean_experts_used,
            "expert_utilization": utilization,
        }


class MoETopPTransformer(nn.Module):
    """Simple transformer using MoETopP layers instead of standard FFN."""

    def __init__(self, config: MoETopPConfig, n_layers: int, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            MoETopPLayer(config) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (logits (B, T, vocab_size), total_aux_loss scalar)."""
        x = self.embed(input_ids)  # (B, T, d_model)
        total_aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=x.dtype)

        for layer in self.layers:
            x, aux = layer(x)
            total_aux_loss = total_aux_loss + aux

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits, total_aux_loss
