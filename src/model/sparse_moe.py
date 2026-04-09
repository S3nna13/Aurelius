"""Sparse Mixture-of-Experts with capacity buffers, token dropping, and load balance loss."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SparseMoEConfig:
    d_model: int = 512
    n_experts: int = 8
    n_active: int = 2            # top-k experts per token
    capacity_factor: float = 1.25  # expert capacity = capacity_factor * T * n_active / n_experts
    expert_d_ff: int = 2048
    aux_loss_coeff: float = 0.01
    expert_dropout: float = 0.0
    jitter_noise: float = 0.0    # add noise to router logits for load balance


class ExpertFFN(nn.Module):
    """Single expert: SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = self.dropout(gate * up)
        return self.down_proj(hidden)


class TokenRouter(nn.Module):
    """Routes tokens to experts via a learned linear projection."""

    def __init__(self, d_model: int, n_experts: int, n_active: int, jitter_noise: float = 0.0) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.jitter_noise = jitter_noise
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute router probabilities and top-k selection.

        Args:
            x: (B, T, d_model)

        Returns:
            router_probs:  (B*T, n_experts) — softmax probabilities
            top_k_indices: (B*T, n_active)  — expert indices selected per token
            top_k_weights: (B*T, n_active)  — normalized routing weights
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        logits = self.router(x_flat)  # (N, n_experts)

        if self.training and self.jitter_noise > 0.0:
            noise = torch.zeros_like(logits).uniform_(0.0, self.jitter_noise)
            logits = logits * (1.0 + noise)

        router_probs = F.softmax(logits, dim=-1)  # (N, n_experts)

        top_k_weights_raw, top_k_indices = torch.topk(router_probs, self.n_active, dim=-1)
        # Renormalize so weights for selected experts sum to 1
        top_k_weights = top_k_weights_raw / (top_k_weights_raw.sum(dim=-1, keepdim=True) + 1e-9)

        return router_probs, top_k_indices, top_k_weights


def compute_capacity(T: int, n_experts: int, n_active: int, capacity_factor: float) -> int:
    """Compute per-expert token capacity.

    Returns:
        capacity = max(int(T * n_active * capacity_factor / n_experts), 1)
    """
    return max(int(T * n_active * capacity_factor / n_experts), 1)


def dispatch_tokens(
    x: Tensor,
    indices: Tensor,
    weights: Tensor,
    n_experts: int,
    capacity: int,
) -> tuple[Tensor, Tensor]:
    """Dispatch tokens to expert capacity buffers with token dropping.

    Iterates through (token, slot) pairs in order; the first `capacity` tokens
    assigned to each expert are accepted; the rest are dropped.

    Args:
        x:         (N, d_model) flattened token representations
        indices:   (N, n_active) expert assignment per token/slot
        weights:   (N, n_active) routing weights (unused here; kept for API consistency)
        n_experts: number of experts
        capacity:  max tokens per expert buffer

    Returns:
        expert_inputs: (n_experts, capacity, d_model) — zero-padded token buffers
        dispatch_mask: (N, n_active) bool — True = accepted, False = dropped
    """
    N, d_model = x.shape
    n_active = indices.shape[1]
    device = x.device
    dtype = x.dtype

    expert_inputs = torch.zeros(n_experts, capacity, d_model, device=device, dtype=dtype)
    dispatch_mask = torch.zeros(N, n_active, dtype=torch.bool, device=device)

    # Track how many slots each expert has filled
    expert_fill = [0] * n_experts

    for slot in range(n_active):
        for token_idx in range(N):
            expert_id = int(indices[token_idx, slot].item())
            fill = expert_fill[expert_id]
            if fill < capacity:
                expert_inputs[expert_id, fill] = x[token_idx].detach() if not x.requires_grad else x[token_idx]
                expert_fill[expert_id] = fill + 1
                dispatch_mask[token_idx, slot] = True

    return expert_inputs, dispatch_mask


def combine_expert_outputs(
    expert_outputs: Tensor,
    indices: Tensor,
    weights: Tensor,
    dispatch_mask: Tensor,
    N: int,
) -> Tensor:
    """Gather expert outputs back to token positions, weighted by router weights.

    Args:
        expert_outputs: (n_experts, capacity, d_model)
        indices:        (N, n_active)
        weights:        (N, n_active) routing weights
        dispatch_mask:  (N, n_active) bool
        N:              total tokens (B*T)

    Returns:
        output: (N, d_model)
    """
    n_experts, capacity, d_model = expert_outputs.shape
    n_active = indices.shape[1]
    device = expert_outputs.device
    dtype = expert_outputs.dtype

    output = torch.zeros(N, d_model, device=device, dtype=dtype)

    # Track which buffer slot corresponds to each accepted (token, slot) pair
    expert_fill = [0] * n_experts

    for slot in range(n_active):
        for token_idx in range(N):
            if not dispatch_mask[token_idx, slot].item():
                continue
            expert_id = int(indices[token_idx, slot].item())
            fill = expert_fill[expert_id]
            output[token_idx] = (
                output[token_idx]
                + weights[token_idx, slot] * expert_outputs[expert_id, fill]
            )
            expert_fill[expert_id] = fill + 1

    return output


class SparseMoELayer(nn.Module):
    """Sparse MoE layer with capacity buffers, token dropping, and load balance loss."""

    def __init__(self, config: SparseMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.router = TokenRouter(
            d_model=config.d_model,
            n_experts=config.n_experts,
            n_active=config.n_active,
            jitter_noise=config.jitter_noise,
        )
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, config.expert_d_ff, config.expert_dropout)
            for _ in range(config.n_experts)
        ])
        # Cache for get_routing_stats()
        self._last_router_probs: Tensor | None = None
        self._last_dispatch_mask: Tensor | None = None
        self._last_indices: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Route, dispatch, process through experts, and combine.

        Args:
            x: (B, T, d_model)

        Returns:
            output:   (B, T, d_model)
            aux_loss: scalar load-balance loss >= 0
        """
        B, T, d_model = x.shape
        N = B * T
        x_flat = x.reshape(N, d_model)

        # 1. Route tokens
        router_probs, top_k_indices, top_k_weights = self.router(x)

        # 2. Compute per-expert capacity
        capacity = compute_capacity(T, self.config.n_experts, self.config.n_active, self.config.capacity_factor)

        # 3. Dispatch to expert buffers (with gradient-compatible indexing)
        expert_inputs, dispatch_mask = dispatch_tokens(
            x_flat, top_k_indices, top_k_weights, self.config.n_experts, capacity
        )

        # 4. Run each expert on its buffer
        expert_outputs = torch.zeros_like(expert_inputs)
        for i, expert in enumerate(self.experts):
            expert_outputs[i] = expert(expert_inputs[i])

        # 5. Combine expert outputs back to token positions
        output_flat = combine_expert_outputs(
            expert_outputs, top_k_indices, top_k_weights, dispatch_mask, N
        )
        output = output_flat.reshape(B, T, d_model)

        # 6. Load balance auxiliary loss (Switch Transformer style, extended to top-k)
        n_experts = self.config.n_experts
        expert_counts = torch.zeros(n_experts, device=x.device, dtype=x.dtype)
        for k in range(self.config.n_active):
            one_hot = F.one_hot(top_k_indices[:, k], num_classes=n_experts).to(x.dtype)
            mask_k = dispatch_mask[:, k].to(x.dtype).unsqueeze(-1)
            expert_counts = expert_counts + (one_hot * mask_k).sum(dim=0)

        total_dispatched = dispatch_mask.sum().clamp(min=1).to(x.dtype)
        f_i = expert_counts / total_dispatched  # fraction of accepted tokens per expert
        P_i = router_probs.mean(dim=0)          # mean router probability per expert
        aux_loss = self.config.aux_loss_coeff * n_experts * (f_i * P_i).sum()

        # Cache stats
        self._last_router_probs = router_probs.detach()
        self._last_dispatch_mask = dispatch_mask.detach()
        self._last_indices = top_k_indices.detach()

        return output, aux_loss

    def get_routing_stats(self) -> dict[str, float]:
        """Return routing statistics from the most recent forward pass.

        Returns:
            dict with keys:
                mean_expert_load  — mean fraction of tokens dispatched per expert
                max_expert_load   — max fraction across experts
                token_drop_rate   — fraction of (token, slot) pairs that were dropped
        """
        if self._last_dispatch_mask is None or self._last_indices is None:
            return {"mean_expert_load": 0.0, "max_expert_load": 0.0, "token_drop_rate": 0.0}

        n_experts = self.config.n_experts
        dispatch_mask = self._last_dispatch_mask  # (N, n_active)
        indices = self._last_indices              # (N, n_active)
        N = dispatch_mask.shape[0]

        expert_counts = torch.zeros(n_experts, device=dispatch_mask.device, dtype=torch.float)
        for k in range(self.config.n_active):
            one_hot = F.one_hot(indices[:, k], num_classes=n_experts).float()
            mask_k = dispatch_mask[:, k].float().unsqueeze(-1)
            expert_counts += (one_hot * mask_k).sum(dim=0)

        load = expert_counts / N
        mean_load = load.mean().item()
        max_load = load.max().item()

        total_slots = dispatch_mask.numel()
        dropped = (~dispatch_mask).sum().item()
        token_drop_rate = dropped / total_slots if total_slots > 0 else 0.0

        return {
            "mean_expert_load": mean_load,
            "max_expert_load": max_load,
            "token_drop_rate": token_drop_rate,
        }
