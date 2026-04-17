"""Mixture-of-Experts (MoE) layer — v2.

Implements Switch Transformer / Mixtral-style sparse MoE with:
- Top-k routing with renormalized softmax weights
- Auxiliary load-balancing loss (Switch Transformer formulation)
- Expert capacity enforcement with overflow masking
- Full transformer block and model wrappers

Pure PyTorch / stdlib only — no external dependencies.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """Single FFN expert: Linear -> activation -> Linear."""

    _ACTIVATIONS = {
        "gelu": F.gelu,
        "silu": F.silu,
        "relu": F.relu,
    }

    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu") -> None:
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(self._ACTIVATIONS)}, got {activation!r}")
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act_fn = self._ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        """x: (N_expert, D) -> (N_expert, D)"""
        return self.fc2(self.act_fn(self.fc1(x)))


# ---------------------------------------------------------------------------
# TopKRouter
# ---------------------------------------------------------------------------

class TopKRouter(nn.Module):
    """Route tokens to top-k experts using a learned linear gate."""

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2) -> None:
        super().__init__()
        if top_k < 1 or top_k > n_experts:
            raise ValueError(f"top_k={top_k} must be in [1, n_experts={n_experts}]")
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: (B, T, D) or (N, D).

        Returns
        -------
        dispatch_weights : (N, top_k)  renormalized routing weights (sum=1 per token)
        expert_indices   : (N, top_k)  LongTensor of chosen expert ids
        """
        if x.dim() == 3:
            x_flat = x.view(-1, x.shape[-1])   # (N, D)
        else:
            x_flat = x

        logits = self.gate(x_flat)            # (N, E)
        probs = F.softmax(logits, dim=-1)     # (N, E)

        top_weights, top_indices = torch.topk(probs, self.top_k, dim=-1)  # (N, top_k)

        # Renormalize so weights sum to 1 per token
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        return top_weights, top_indices   # (N, top_k), (N, top_k) LongTensor


# ---------------------------------------------------------------------------
# ExpertCapacityBuffer
# ---------------------------------------------------------------------------

class ExpertCapacityBuffer(nn.Module):
    """Enforce per-expert token capacity limits (no learnable parameters)."""

    def __init__(self, n_experts: int, capacity_factor: float = 1.25) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

    def compute_capacity(self, n_tokens: int, top_k: int) -> int:
        """capacity = ceil(capacity_factor * n_tokens * top_k / n_experts), min 1."""
        raw = self.capacity_factor * n_tokens * top_k / self.n_experts
        return max(1, int(math.ceil(raw)))

    def apply_capacity(
        self,
        dispatch_weights: Tensor,
        expert_indices: Tensor,
        n_tokens: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Cap each expert to capacity tokens; overflow tokens get weight=0.

        Parameters
        ----------
        dispatch_weights : (N, top_k)
        expert_indices   : (N, top_k) LongTensor
        n_tokens         : N

        Returns
        -------
        weights_capped  : (N, top_k)
        indices_capped  : (N, top_k)  overflow slots keep their index but weight=0
        overflow_mask   : (N,) bool   True if the token was dropped from all experts
        """
        top_k = dispatch_weights.shape[1]
        capacity = self.compute_capacity(n_tokens, top_k)

        weights_capped = dispatch_weights.clone()
        # per-expert running count
        expert_count = torch.zeros(self.n_experts, dtype=torch.long, device=dispatch_weights.device)

        for slot in range(top_k):
            for tok_idx in range(n_tokens):
                e = int(expert_indices[tok_idx, slot].item())
                if expert_count[e] >= capacity:
                    weights_capped[tok_idx, slot] = 0.0
                else:
                    expert_count[e] += 1

        # A token is overflowed if all its routing weights became zero
        overflow_mask = (weights_capped.sum(dim=-1) == 0.0)  # (N,)
        return weights_capped, expert_indices, overflow_mask


# ---------------------------------------------------------------------------
# MoELayer
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts feed-forward layer."""

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_ff: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, n_experts, top_k)
        self.capacity_buffer = ExpertCapacityBuffer(n_experts, capacity_factor)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def _aux_loss(self, router_probs: Tensor, expert_indices: Tensor) -> Tensor:
        """Switch Transformer auxiliary load-balancing loss.

        loss = n_experts * sum_i( f_i * p_i )
        f_i = fraction of tokens dispatched to expert i
        p_i = mean router probability for expert i
        """
        N = router_probs.shape[0]
        E = self.n_experts

        # f_i: fraction of tokens routed to expert i (at least one top-k slot)
        one_hot = torch.zeros(N, E, device=router_probs.device, dtype=router_probs.dtype)
        one_hot.scatter_(1, expert_indices, 1.0)
        assigned = one_hot.clamp(max=1.0)  # (N, E) — each expert counted once per token
        f = assigned.mean(dim=0)           # (E,)

        # p_i: mean router probability for expert i
        p = router_probs.mean(dim=0)       # (E,)

        return E * (f * p).sum()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: (B, T, D) -> (output: (B, T, D), aux_loss: scalar)."""
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)  # (N, D)

        # Full softmax probs for aux loss
        logits = self.router.gate(x_flat)                        # (N, E)
        router_probs = F.softmax(logits, dim=-1)                 # (N, E)

        top_weights_raw, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize
        dispatch_weights = top_weights_raw / (top_weights_raw.sum(dim=-1, keepdim=True) + 1e-9)

        # Capacity enforcement
        dispatch_weights, top_indices, _ = self.capacity_buffer.apply_capacity(
            dispatch_weights, top_indices, N
        )

        # Dispatch tokens to experts and accumulate weighted outputs
        output = torch.zeros_like(x_flat)  # (N, D)

        for e_idx, expert in enumerate(self.experts):
            # Tokens that use this expert in any slot
            mask = (top_indices == e_idx)  # (N, top_k) bool
            token_mask = mask.any(dim=-1)  # (N,)
            if not token_mask.any():
                continue

            global_positions = token_mask.nonzero(as_tuple=True)[0]  # indices in [0, N)
            tokens_in = x_flat[global_positions]    # (n_e, D)
            expert_out = expert(tokens_in)          # (n_e, D)

            for slot in range(self.top_k):
                slot_mask_global = mask[:, slot]  # (N,) bool
                if not slot_mask_global.any():
                    continue
                # local positions within tokens_in that correspond to this slot
                local_slot = slot_mask_global[global_positions]  # bool (n_e,)
                weights_e = dispatch_weights[global_positions[local_slot], slot]  # (k,)
                output[global_positions[local_slot]] += (
                    weights_e.unsqueeze(-1) * expert_out[local_slot]
                )

        output = output.view(B, T, D)
        aux_loss = self._aux_loss(router_probs, top_indices)
        return output, aux_loss


# ---------------------------------------------------------------------------
# MoETransformerBlock
# ---------------------------------------------------------------------------

class MoETransformerBlock(nn.Module):
    """Transformer block where the FFN is replaced by MoELayer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        d_ff: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = MoELayer(d_model, n_experts, d_ff, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Upper-triangular additive mask (neg-inf above diagonal)."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x: (B, T, D) -> (output: (B, T, D), aux_loss: scalar)."""
        T = x.shape[1]
        causal = self._causal_mask(T, x.device)

        attn_out, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        x = self.norm1(x + attn_out)

        moe_out, aux_loss = self.moe(x)
        x = self.norm2(x + moe_out)

        return x, aux_loss


# ---------------------------------------------------------------------------
# MoEModel
# ---------------------------------------------------------------------------

class MoEModel(nn.Module):
    """Full language model with MoE transformer blocks."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        n_experts: int,
        d_ff: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, n_experts, d_ff, top_k)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """input_ids: (B, T) LongTensor -> (logits: (B, T, V), total_aux_loss: scalar)."""
        x = self.embedding(input_ids)
        total_aux = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux = block(x)
            total_aux = total_aux + aux
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_aux

    @torch.no_grad()
    def router_statistics(self, input_ids: Tensor) -> Dict[str, object]:
        """Compute routing statistics across all blocks.

        Returns
        -------
        dict with keys:
            mean_utilization : float  average fraction of tokens per expert
            load_balance_cv  : float  coefficient of variation of expert load
            expert_collapse  : bool   any expert below 1/n_experts * 0.1 utilization
        """
        self.eval()
        x = self.embedding(input_ids)
        B, T, _ = x.shape
        N = B * T

        all_f = []

        for block in self.blocks:
            causal = block._causal_mask(T, x.device)
            attn_out, _ = block.attn(x, x, x, attn_mask=causal, need_weights=False)
            x_normed = block.norm1(x + attn_out)

            moe: MoELayer = block.moe
            x_flat = x_normed.view(N, -1)
            logits = moe.router.gate(x_flat)
            probs = F.softmax(logits, dim=-1)
            _, top_indices = torch.topk(probs, moe.top_k, dim=-1)

            E = moe.n_experts
            one_hot = torch.zeros(N, E, device=x.device)
            one_hot.scatter_(1, top_indices, 1.0)
            assigned = one_hot.clamp(max=1.0)
            f = assigned.mean(dim=0)   # (E,)
            all_f.append(f)

            moe_out, _ = moe(x_normed)
            x = block.norm2(x_normed + moe_out)

        x = self.norm(x)

        stacked = torch.stack(all_f, dim=0)   # (n_layers, E)
        mean_f = stacked.mean(dim=0)           # (E,)

        mean_utilization = float(mean_f.mean().item())
        std_util = float(mean_f.std().item())
        load_balance_cv = std_util / (mean_utilization + 1e-9)

        n_experts = mean_f.shape[0]
        threshold = (1.0 / n_experts) * 0.1
        expert_collapse = bool((mean_f < threshold).any().item())

        return {
            "mean_utilization": mean_utilization,
            "load_balance_cv": load_balance_cv,
            "expert_collapse": expert_collapse,
        }
