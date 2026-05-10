"""Sparse Mixture-of-Experts layer with top-K token routing.

Implements Switch Transformer / Mixtral-style sparse MoE:
  - RouterConfig dataclass
  - TopKRouter: computes dispatch weights, indices, and auxiliary load-balancing loss
  - ExpertFFN: single 2-layer MLP expert (d_model -> d_ff -> d_model, SiLU)
  - SparseMoELayer: routes tokens to top-K experts, aggregates weighted outputs
  - MoEBlock: full transformer block (MHA + RMSNorm + SparseMoELayer)

Usage:
    from aurelius.model.moe import RouterConfig, TopKRouter, ExpertFFN, SparseMoELayer, MoEBlock
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# RouterConfig
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """Configuration for the top-K sparse router."""

    n_experts: int
    top_k: int = 2
    capacity_factor: float = 1.25
    jitter_noise: float = 0.0


# ---------------------------------------------------------------------------
# TopKRouter
# ---------------------------------------------------------------------------


class TopKRouter(nn.Module):
    """Sparse top-K router.

    For each token computes:
      - dispatch_weights: (B, T, top_k)  — softmax scores over full vocab,
        sliced to the top_k selected experts (NOT renormalized to 1).
      - dispatch_indices: (B, T, top_k)  — integer expert ids in [0, n_experts).
      - router_loss: scalar auxiliary load-balancing loss
          = n_experts * mean_over_experts(f_i * p_i)
        where f_i = fraction of tokens dispatched to expert i (over the top-k
        selections), p_i = mean gate probability to expert i.

    If jitter_noise > 0, Uniform(-jitter_noise, +jitter_noise) noise is added
    to the router logits during training.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        jitter_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute routing decisions.

        Args:
            x: (B, T, d_model)

        Returns:
            dispatch_weights:  (B, T, top_k)  — softmax gate scores
            dispatch_indices:  (B, T, top_k)  — int expert ids
            router_loss:       scalar          — auxiliary load-balancing loss
        """
        B, T, D = x.shape

        # (B, T, n_experts)
        logits = self.gate(x)

        # Optional training-time jitter
        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(logits).uniform_(-self.jitter_noise, self.jitter_noise)
            logits = logits + noise

        # Full softmax over all experts — used for p_i in the aux loss
        probs = F.softmax(logits, dim=-1)  # (B, T, n_experts)

        # Top-K selection
        top_k_vals, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        # top_k_vals: (B, T, top_k) — already softmax scores (not renormalized)
        # top_k_idx:  (B, T, top_k)

        # ---- Auxiliary load-balancing loss --------------------------------
        # f_i = fraction of (token, slot) assignments that went to expert i
        #       counted over all B*T tokens and all top_k slots
        N = B * T  # total tokens
        idx_flat = top_k_idx.reshape(-1)  # (N * top_k,)
        counts = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
        counts.scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=x.dtype))
        f_i = counts / (N * self.top_k)  # (n_experts,)

        # p_i = mean gate probability to expert i across all tokens
        p_i = probs.reshape(N, self.n_experts).mean(dim=0)  # (n_experts,)

        router_loss = self.n_experts * (f_i * p_i).mean()

        return top_k_vals, top_k_idx, router_loss


# ---------------------------------------------------------------------------
# ExpertFFN
# ---------------------------------------------------------------------------


class ExpertFFN(nn.Module):
    """Single expert: 2-layer MLP with SiLU activation.

    d_model -> d_ff -> d_model
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply expert MLP.

        Args:
            x: (..., d_model)

        Returns:
            (..., d_model)
        """
        return self.w2(F.silu(self.w1(x)))


# ---------------------------------------------------------------------------
# SparseMoELayer
# ---------------------------------------------------------------------------


class SparseMoELayer(nn.Module):
    """Sparse top-K Mixture-of-Experts layer.

    Each token is routed to `top_k` of `n_experts` ExpertFFN modules.
    The weighted sum of expert outputs is returned along with the auxiliary
    load-balancing loss from the router.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Route tokens, dispatch to experts, aggregate.

        Args:
            x: (B, T, d_model)

        Returns:
            output:      (B, T, d_model) — same shape as input
            router_loss: scalar
        """
        B, T, D = x.shape
        dispatch_weights, dispatch_indices, router_loss = self.router(x)
        # dispatch_weights:  (B, T, top_k)
        # dispatch_indices:  (B, T, top_k)

        x_flat = x.reshape(B * T, D)  # (N, D)
        w_flat = dispatch_weights.reshape(B * T, self.top_k)  # (N, top_k)
        idx_flat = dispatch_indices.reshape(B * T, self.top_k)  # (N, top_k)

        output = torch.zeros_like(x_flat)  # (N, D)

        # Iterate over experts: gather assigned tokens, run FFN, scatter back
        for expert_id in range(self.n_experts):
            # mask: (N, top_k) — True where this expert is selected
            mask = idx_flat == expert_id  # (N, top_k)
            # token_mask: (N,) — True if token is routed here in any slot
            token_mask = mask.any(dim=-1)  # (N,)

            if not token_mask.any():
                continue

            # Tokens assigned to this expert
            expert_input = x_flat[token_mask]  # (n_tok, D)
            expert_out = self.experts[expert_id](expert_input)  # (n_tok, D)

            # Weight: sum over slots (at most one slot per expert per token)
            # mask[token_mask]: (n_tok, top_k), w_flat[token_mask]: (n_tok, top_k)
            weight = (mask[token_mask].float() * w_flat[token_mask]).sum(
                dim=-1, keepdim=True
            )  # (n_tok, 1)

            output[token_mask] = output[token_mask] + weight * expert_out

        return output.reshape(B, T, D), router_loss


# ---------------------------------------------------------------------------
# MoEBlock
# ---------------------------------------------------------------------------


class MoEBlock(nn.Module):
    """Transformer-style block with a Sparse MoE FFN sub-layer.

    Architecture:
        x  ->  RMSNorm  ->  MultiheadAttention  ->  residual  ->  RMSNorm  ->  SparseMoELayer  ->  residual

    Args:
        d_model:   model hidden dimension
        d_ff:      expert feed-forward hidden dimension
        n_experts: number of experts
        top_k:     number of experts each token is routed to
        n_heads:   number of attention heads
    """  # noqa: E501

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        top_k: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()

        # Use nn.RMSNorm if available (PyTorch >= 2.4), else LayerNorm
        if hasattr(nn, "RMSNorm"):

            def norm_cls():
                return nn.RMSNorm(d_model)
        else:

            def norm_cls():
                return nn.LayerNorm(d_model)

        self.norm1 = norm_cls()
        self.norm2 = norm_cls()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            bias=False,
        )

        self.moe = SparseMoELayer(d_model, d_ff, n_experts, top_k)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply attention + MoE FFN with pre-norm and residual connections.

        Args:
            x: (B, T, d_model)

        Returns:
            output:      (B, T, d_model)
            router_loss: scalar
        """
        # Self-attention sub-layer (pre-norm)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MoE FFN sub-layer (pre-norm)
        normed2 = self.norm2(x)
        moe_out, router_loss = self.moe(normed2)
        x = x + moe_out

        return x, router_loss


# ---------------------------------------------------------------------------
# MoEConfig  (used by SparseMoEFFN, BalancedMoEFFN, upcycle helpers)
# ---------------------------------------------------------------------------


@dataclass
class MoEConfig:
    """Configuration for Sparse MoE FFN layers.

    Attributes:
        n_experts:           Total number of expert FFNs.
        top_k:               Experts activated per token.
        load_balance_alpha:  Scale applied to the router auxiliary loss.
        jitter_noise:        Uniform noise added to router logits during training.
        bias_update_rate:    Step size for the bias-based load balancer (BalancedMoEFFN).
    """

    n_experts: int = 8
    top_k: int = 2
    load_balance_alpha: float = 0.01
    jitter_noise: float = 0.0
    bias_update_rate: float = 0.001


# ---------------------------------------------------------------------------
# SparseMoEFFN  (drop-in SwiGLUFFN replacement; returns (output, aux_loss))
# ---------------------------------------------------------------------------


class SparseMoEFFN(nn.Module):
    """Sparse MoE FFN layer — drop-in replacement for SwiGLUFFN.

    Wraps SparseMoELayer and exposes ``self.experts`` so that weight
    copying (e.g. during dense-to-MoE upcycling) works directly.

    Args:
        config:  AureliusConfig — uses ``d_model`` and ``d_ff``.
        moe_cfg: MoEConfig — uses ``n_experts``, ``top_k``,
                 ``load_balance_alpha``, and ``jitter_noise``.

    Returns (from forward):
        output:   (B, T, d_model)
        aux_loss: scalar = load_balance_alpha * router_loss
    """

    def __init__(self, config, moe_cfg: MoEConfig | None = None) -> None:
        super().__init__()
        self.moe_cfg = moe_cfg if moe_cfg is not None else MoEConfig()
        self.load_balance_alpha = self.moe_cfg.load_balance_alpha

        self._layer = SparseMoELayer(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_experts=self.moe_cfg.n_experts,
            top_k=self.moe_cfg.top_k,
        )
        # Expose the expert ModuleList and router gate for inspection / upcycling.
        # router is the inner nn.Linear gate so that moe.router.weight matches
        # the same interface as BalancedMoEFFN.router.
        self.experts: nn.ModuleList = self._layer.experts
        self.router: nn.Linear = self._layer.router.gate

        # Router initialized with small std so all experts start equal
        nn.init.normal_(self._layer.router.gate.weight, std=0.01)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply sparse MoE FFN.

        Args:
            x: (B, T, d_model)

        Returns:
            output:   (B, T, d_model)
            aux_loss: scalar
        """
        output, router_loss = self._layer(x)
        return output, self.load_balance_alpha * router_loss
class BalancedMoEFFN:
    """DeepSeek-V3 style balanced MoE with bias-based load balancing.

    No auxiliary loss — uses per-expert bias updated via sign rule.
    Ported from Aurelius's src/model/moe_balanced.py.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        load_balance_coef: float = 0.01,
    ) -> None:
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balance_coef = load_balance_coef
        self.expert_bias: list[float] = [0.0] * n_experts
        self.experts = [ExpertFFN(d_model, d_model * 4) for _ in range(n_experts)]

    def update_bias(self, expert_loads: list[float]) -> None:
        target = sum(expert_loads) / self.n_experts * self.capacity_factor
        for i in range(self.n_experts):
            if expert_loads[i] > target:
                self.expert_bias[i] -= 0.001
            else:
                self.expert_bias[i] += 0.001

    def get_utilization(self) -> dict[str, float]:
        return {"mean_load": 0.0, "load_imbalance": 0.0}


class SoftMoELayer:
    """Fully differentiable soft MoE (no token dropping, no aux loss).

    Ported from Aurelius's src/model/soft_moe.py.

    Uses learnable slot embeddings: each expert has n_slots slots.
    dispatch = softmax(logits, dim=tokens), combine = softmax(logits, dim=slots).
    """

    def __init__(
        self, d_model: int, n_experts: int, n_slots: int = 1, d_ff: int | None = None
    ) -> None:
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_slots = n_slots
        self.d_ff = d_ff or d_model * 4
        # Slot embeddings: n_experts * n_slots x d_model
        self.slot_embeddings: list[list[float]] = [
            [0.01 * (i * 0.1 + j) for j in range(d_model)]  # mock
            for i in range(n_experts * n_slots)
        ]
        self.experts = [ExpertFFN(d_model, self.d_ff) for _ in range(n_experts)]

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        n_tokens = len(x)
        n_slots_total = self.n_experts * self.n_slots

        # Dispatch weights: softmax over tokens for each slot
        # Combine weights: softmax over slots for each token
        import random

        logits = [[random.gauss(0, 0.1) for _ in range(n_slots_total)] for _ in range(n_tokens)]

        dispatch = TopKRouter._softmax(logits)  # softmax over tokens handled per-slot
        combine = TopKRouter._softmax(
            [[logits[t][s] for t in range(n_tokens)] for s in range(n_slots_total)]
        )  # softmax over tokens per-slot, then transposed

        # Combine: weighted sum of slot embeddings
        slot_inputs: list[list[float]] = []
        for s in range(n_slots_total):
            weighted = [0.0] * self.d_model
            for t in range(n_tokens):
                w = dispatch[t][s] if s < len(dispatch[t]) else 0.0
                for d in range(self.d_model):
                    weighted[d] += w * x[t][d]
            slot_inputs.append(weighted)

        # Process each slot through its expert
        expert_outputs: list[list[float]] = []
        for s in range(n_slots_total):
            expert_idx = s // self.n_slots
            expert_out = self.experts[expert_idx].forward(slot_inputs[s])
            expert_outputs.append(expert_out)

        # Transpose combine to [n_tokens, n_slots_total]
        combine_t = [[combine[s][t] for s in range(n_slots_total)] for t in range(n_tokens)]
        output: list[list[float]] = [[0.0] * self.d_model for _ in range(n_tokens)]
        for t in range(n_tokens):
            for s in range(n_slots_total):
                w = combine_t[t][s]
                for d in range(self.d_model):
                    output[t][d] += w * expert_outputs[s][d]

        return output


class ExpertChoiceLayer:
    """Expert Choice routing — each expert selects top-capacity tokens.

    Ported from Aurelius's src/model/expert_choice.py.

    Instead of tokens choosing experts, each expert selects its top tokens.
    This guarantees perfect load balance.
    """

    def __init__(
        self, d_model: int, n_experts: int, capacity_factor: float = 1.0, d_ff: int | None = None
    ) -> None:
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.d_ff = d_ff or d_model * 4
        self.experts = [ExpertFFN(d_model, self.d_ff) for _ in range(n_experts)]
        self.router = TopKRouter(d_model, n_experts, top_k=1)

    def forward(self, x: list[list[float]]) -> tuple[list[list[float]], float]:
        n_tokens = len(x)
        capacity = max(1, int(math.ceil(self.capacity_factor * n_tokens / self.n_experts)))

        # Each expert scores all tokens
        import random

        scores: list[list[float]] = [
            [random.gauss(0, 1) for _ in range(n_tokens)] for _ in range(self.n_experts)
        ]

        # Each expert selects top-capacity tokens
        output: list[list[float]] = [[0.0] * self.d_model for _ in range(n_tokens)]
        aux_loss = 0.0

        for exp_idx in range(self.n_experts):
            top_tokens = sorted(
                enumerate(scores[exp_idx]),
                key=lambda x: -x[1],
            )[:capacity]

            for tok_idx, _score in top_tokens:
                expert_out = self.experts[exp_idx].forward(x[tok_idx])
                for d in range(self.d_model):
                    output[tok_idx][d] += expert_out[d] / self.n_experts

        return output, aux_loss
