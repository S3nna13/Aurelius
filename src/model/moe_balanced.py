"""DeepSeek-V3-style auxiliary-loss-free MoE load balancing.

Instead of an auxiliary loss (which hurts primary task performance), a
per-expert bias vector is maintained. After each forward pass the bias is
incremented for underloaded experts and decremented for overloaded ones.
This steers routing without polluting gradients.

Reference: DeepSeek-V3 Technical Report (2024).

Usage:
    # Drop-in replacement for SparseMoEFFN:
    self.ffn = BalancedMoEFFN(config)

    # In forward, same interface as SparseMoEFFN:
    ffn_out, aux_loss = self.ffn(x)   # aux_loss is always 0.0
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .moe import MoEConfig


class BalancedMoEFFN(nn.Module):
    """Sparse MoE FFN with bias-based load balancing (no auxiliary loss).

    Expert routing uses ``router_logits + expert_bias`` so that underloaded
    experts are gradually preferred over time.  The bias vector is updated
    after every forward pass using a simple sign rule:

        if actual_load > target_load:  bias -= bias_update_rate
        else:                          bias += bias_update_rate

    The bias is stored as an ``nn.Parameter`` with ``requires_grad=False`` so
    it is part of the module state (saved/loaded with ``state_dict``) but
    never receives gradient updates from the optimiser.

    Args:
        config: AureliusConfig (uses ``d_model``, ``d_ff``).
        moe_cfg: MoE-specific configuration (``n_experts``, ``top_k``).
            If ``moe_cfg`` has a ``bias_update_rate`` attribute it is used,
            otherwise the default of 0.001 applies.
    """

    def __init__(self, config, moe_cfg: MoEConfig | None = None) -> None:
        super().__init__()
        from .ffn import SwiGLUFFN

        self.moe_cfg = moe_cfg or MoEConfig()
        self.n_experts = self.moe_cfg.n_experts
        self.top_k = self.moe_cfg.top_k
        self.bias_update_rate: float = getattr(moe_cfg, "bias_update_rate", 0.001)

        # Expert networks — each is a full SwiGLUFFN
        self.experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(self.n_experts)
        ])

        # Router: token hidden state → expert logits
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        # Per-expert bias — updated manually, never by the optimiser
        self.expert_bias = nn.Parameter(
            torch.zeros(self.n_experts), requires_grad=False
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k routing given flat token matrix.

        Args:
            x_flat: (N, D)

        Returns:
            top_k_probs:   (N, top_k) renormalized routing weights
            top_k_indices: (N, top_k) selected expert indices
        """
        router_logits = self.router(x_flat)                       # (N, n_experts)
        biased_logits = router_logits + self.expert_bias          # broadcast over N
        router_probs = F.softmax(biased_logits, dim=-1)           # (N, n_experts)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize so weights sum to 1 per token
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return top_k_probs, top_k_indices

    def _update_bias(
        self, top_k_indices: torch.Tensor, N: int
    ) -> None:
        """Update expert_bias in-place based on observed load vs target load.

        Args:
            top_k_indices: (N, top_k) expert selections for this batch
            N: total number of tokens (B * S)
        """
        target_load = self.top_k / self.n_experts  # ideal fraction per expert

        with torch.no_grad():
            for i in range(self.n_experts):
                # Actual fraction of (token, slot) pairs assigned to expert i
                count = (top_k_indices == i).sum().float()
                actual_load = (count / (N * self.top_k)).item()

                if actual_load > target_load:
                    self.expert_bias[i] -= self.bias_update_rate
                else:
                    self.expert_bias[i] += self.bias_update_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts and compute weighted output.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (output, aux_loss):
            - output:   (batch, seq_len, d_model)
            - aux_loss: scalar ``torch.tensor(0.0)`` — no auxiliary loss
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)   # (N, D)
        N = x_flat.shape[0]

        top_k_probs, top_k_indices = self._route(x_flat)  # (N, top_k) each

        # Dispatch tokens to experts and accumulate weighted output
        output = torch.zeros_like(x_flat)  # (N, D)

        for expert_idx in range(self.n_experts):
            mask = (top_k_indices == expert_idx)    # (N, top_k)
            token_mask = mask.any(dim=-1)            # (N,)

            if not token_mask.any():
                continue

            expert_tokens = x_flat[token_mask]                      # (n_tok, D)
            expert_out = self.experts[expert_idx](expert_tokens)    # (n_tok, D)

            # Sum weights across top_k slots (at most one match per token)
            expert_weights = (
                mask[token_mask].float() * top_k_probs[token_mask]
            ).sum(dim=-1, keepdim=True)  # (n_tok, 1)

            output[token_mask] = output[token_mask] + expert_weights * expert_out

        output = output.view(B, S, D)

        # Update bias based on observed load (no gradient involved)
        self._update_bias(top_k_indices, N)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return output, aux_loss

    def get_expert_loads(self, x: torch.Tensor) -> dict[str, float]:
        """Return the actual load fraction per expert for input ``x``.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            Dict mapping ``"expert_i"`` to its fraction of (token, slot) pairs.
        """
        with torch.no_grad():
            B, S, D = x.shape
            x_flat = x.view(-1, D)
            N = x_flat.shape[0]
            _, top_k_indices = self._route(x_flat)

            loads: dict[str, float] = {}
            for i in range(self.n_experts):
                count = (top_k_indices == i).sum().float()
                loads[f"expert_{i}"] = (count / (N * self.top_k)).item()
        return loads


# ---------------------------------------------------------------------------
# New-style MoE with explicit load-balance auxiliary loss
# ---------------------------------------------------------------------------

@dataclass
class MoEBalancedConfig:
    """Configuration for MoEBalancedLayer and related components."""
    n_experts: int = 8
    top_k: int = 2
    d_model: int = 64
    d_ff: int = 256
    capacity_factor: float = 1.25
    load_balance_coef: float = 1e-2


class ExpertLayer(nn.Module):
    """Simple FFN expert: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class RouterWithLoadBalancing(nn.Module):
    """Router that computes top-k routing weights and an auxiliary load-balance loss.

    The load-balance loss follows the Switch Transformer formulation:
        L_aux = load_balance_coef * sum_i(f_i * p_i)
    where f_i is the fraction of tokens routed to expert i (computed with a
    straight-through / detached token count) and p_i is the mean router
    probability for expert i (differentiable).
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 load_balance_coef: float = 1e-2) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.load_balance_coef = load_balance_coef
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights, expert indices, and load-balance loss.

        Args:
            x: (N, d_model) — flat token matrix

        Returns:
            routing_weights:    (N, top_k)  — renormalized softmax weights
            expert_indices:     (N, top_k)  — selected expert indices per token
            load_balance_loss:  scalar      — auxiliary loss encouraging balance
        """
        N = x.shape[0]
        logits = self.gate(x)                              # (N, n_experts)
        router_probs = F.softmax(logits, dim=-1)          # (N, n_experts)

        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize so weights sum to 1 per token
        routing_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # --- load balance loss (Switch Transformer style) ---
        # f_i: fraction of tokens whose argmax (top-1) is expert i — detached
        # p_i: mean router probability for expert i — differentiable
        # Use top-1 assignment for f_i to avoid gradient through discrete selection
        top1_indices = top_k_indices[:, 0]  # (N,)
        one_hot = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)
        one_hot.scatter_(1, top1_indices.unsqueeze(1), 1.0)
        f_i = one_hot.mean(dim=0)           # (n_experts,) — fraction of tokens

        p_i = router_probs.mean(dim=0)      # (n_experts,) — mean prob per expert

        load_balance_loss = self.load_balance_coef * (f_i.detach() * p_i).sum()

        return routing_weights, top_k_indices, load_balance_loss


class MoEBalancedLayer(nn.Module):
    """MoE layer that dispatches tokens to top-k experts with load-balance loss.

    Args:
        config: MoEBalancedConfig
    """

    def __init__(self, config: MoEBalancedConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            ExpertLayer(config.d_model, config.d_ff)
            for _ in range(config.n_experts)
        ])
        self.router = RouterWithLoadBalancing(
            d_model=config.d_model,
            n_experts=config.n_experts,
            top_k=config.top_k,
            load_balance_coef=config.load_balance_coef,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens and aggregate expert outputs.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output:   (batch, seq_len, d_model)
            aux_loss: scalar load-balance auxiliary loss
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (N, D)
        N = x_flat.shape[0]

        routing_weights, expert_indices, aux_loss = self.router(x_flat)

        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.config.n_experts):
            mask = (expert_indices == expert_idx)   # (N, top_k) bool
            token_mask = mask.any(dim=-1)            # (N,) bool

            if not token_mask.any():
                continue

            expert_input = x_flat[token_mask]
            expert_out = self.experts[expert_idx](expert_input)  # (n_tok, D)

            # Weight for each token-expert pair
            expert_weights = (
                mask[token_mask].float() * routing_weights[token_mask]
            ).sum(dim=-1, keepdim=True)  # (n_tok, 1)

            output[token_mask] = output[token_mask] + expert_weights * expert_out

        output = output.view(B, S, D)
        return output, aux_loss


class MoEBalancedTransformer(nn.Module):
    """Stack of MoEBalancedLayer blocks for testing purposes.

    Each block applies a MoEBalancedLayer. Outputs total aux_loss summed
    across all layers.

    Args:
        config: MoEBalancedConfig
        n_layers: number of MoE layers to stack
    """

    def __init__(self, config: MoEBalancedConfig, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MoEBalancedLayer(config) for _ in range(n_layers)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply stacked MoE layers.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output:         (batch, seq_len, d_model)
            total_aux_loss: scalar sum of aux losses across all layers
        """
        total_aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype).squeeze()
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss
        return x, total_aux_loss


def compute_expert_utilization(
    routing_indices: torch.Tensor, n_experts: int
) -> dict:
    """Compute expert utilization statistics from routing indices.

    Args:
        routing_indices: (N, top_k) integer tensor of expert assignments
        n_experts: total number of experts

    Returns:
        dict with:
          - per_expert_load: list of float fractions (sums to ~1.0)
          - load_imbalance:  max_load / mean_load  (>= 1.0)
          - entropy:         routing entropy (higher = more balanced)
    """
    N, top_k = routing_indices.shape
    total_assignments = N * top_k

    counts = []
    for i in range(n_experts):
        count = (routing_indices == i).sum().item()
        counts.append(count)

    per_expert_load = [c / total_assignments for c in counts]

    mean_load = sum(per_expert_load) / n_experts
    max_load = max(per_expert_load)
    load_imbalance = max_load / mean_load if mean_load > 0 else 1.0

    # Entropy of the load distribution (using per_expert_load as probabilities)
    eps = 1e-9
    entropy = -sum(
        p * (p + eps) ** 0 * (torch.tensor(p + eps).log().item())
        for p in per_expert_load
        if p > 0
    )

    return {
        "per_expert_load": per_expert_load,
        "load_imbalance": load_imbalance,
        "entropy": entropy,
    }
