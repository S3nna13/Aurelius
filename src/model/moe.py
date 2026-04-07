"""Sparse Mixture of Experts FFN (Shazeer et al., 2017 - arXiv:1701.06538).

Drop-in replacement for SwiGLUFFN. Each token is routed to top-k of N
expert FFNs. Load balancing auxiliary loss prevents expert collapse.

Usage:
    # In TransformerBlock, replace:
    self.ffn = SwiGLUFFN(config)
    # With:
    self.ffn = SparseMoEFFN(config)

    # In forward, collect aux loss:
    ffn_out, aux_loss = self.ffn(x)  # SparseMoEFFN returns (output, aux_loss)
    # vs SwiGLUFFN returns just output
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MoEConfig:
    n_experts: int = 8          # total expert count
    top_k: int = 2              # experts activated per token
    load_balance_alpha: float = 0.01  # auxiliary loss weight


class SparseMoEFFN(nn.Module):
    """Sparse MoE FFN with top-k gating and load balancing.

    Each forward call returns (output, aux_loss). The aux_loss should be
    added to the main training loss to encourage balanced expert utilization.

    Args:
        config: AureliusConfig (uses d_model, d_ff).
        moe_cfg: MoE-specific configuration.
    """

    def __init__(self, config, moe_cfg: MoEConfig | None = None) -> None:
        super().__init__()
        from .ffn import SwiGLUFFN

        self.moe_cfg = moe_cfg or MoEConfig()
        self.n_experts = self.moe_cfg.n_experts
        self.top_k = self.moe_cfg.top_k

        # Expert networks — each is a full SwiGLUFFN
        self.experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(self.n_experts)
        ])

        # Router: token hidden state → expert logits
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts and compute weighted output.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (output, aux_loss):
            - output: (batch, seq_len, d_model)
            - aux_loss: scalar auxiliary load-balancing loss
        """
        B, S, D = x.shape
        # Flatten to (B*S, D) for routing
        x_flat = x.view(-1, D)  # (N, D) where N = B*S
        N = x_flat.shape[0]

        # Router scores → softmax probabilities
        router_logits = self.router(x_flat)              # (N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize top-k weights so they sum to 1 per token
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Dispatch to experts and accumulate weighted output
        output = torch.zeros_like(x_flat)  # (N, D)

        for expert_idx in range(self.n_experts):
            # expert_mask: (N, top_k) — True where top_k_indices == expert_idx
            mask = (top_k_indices == expert_idx)   # (N, top_k)
            token_mask = mask.any(dim=-1)           # (N,) — True if token goes to this expert

            if not token_mask.any():
                continue

            # Get tokens for this expert
            expert_tokens = x_flat[token_mask]                     # (n_tokens, D)
            expert_out = self.experts[expert_idx](expert_tokens)   # (n_tokens, D)

            # Routing weight: sum over top_k slots that matched this expert
            # (in practice at most one slot per token matches a given expert)
            expert_weights = (mask[token_mask].float() * top_k_probs[token_mask]).sum(
                dim=-1, keepdim=True
            )  # (n_tokens, 1)

            output[token_mask] = output[token_mask] + expert_weights * expert_out

        output = output.view(B, S, D)

        # Load balancing auxiliary loss (Switch Transformer, Eq. 4)
        # f_i = fraction of tokens dispatched to expert i (over all top-k selections)
        # P_i = mean routing probability to expert i
        # L_aux = alpha * n_experts * sum(f_i * P_i)
        expert_counts = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            one_hot = F.one_hot(top_k_indices[:, k], num_classes=self.n_experts).to(x.dtype)
            expert_counts = expert_counts + one_hot.sum(dim=0)

        f_i = expert_counts / (N * self.top_k)   # fraction per expert
        P_i = router_probs.mean(dim=0)            # mean routing prob per expert

        aux_loss = self.moe_cfg.load_balance_alpha * self.n_experts * (f_i * P_i).sum()

        return output, aux_loss

    def expert_utilization(self, x: torch.Tensor) -> dict[str, float]:
        """Compute per-expert utilization stats (for monitoring).

        Returns dict: expert_i -> fraction of tokens routed to it.
        """
        with torch.no_grad():
            B, S, D = x.shape
            x_flat = x.view(-1, D)
            router_logits = self.router(x_flat)
            router_probs = F.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

            utilization = {}
            for i in range(self.n_experts):
                frac = (top_k_indices == i).float().mean().item()
                utilization[f"expert_{i}"] = frac
        return utilization
