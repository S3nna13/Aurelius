"""Expert Choice routing for Mixture of Experts (Zhou et al., 2022 - arXiv:2202.09368).

Drop-in replacement for SparseMoEFFN. Unlike token-choice routing (each token
picks top-k experts), expert-choice routing lets each expert pick its top-k
tokens. This guarantees perfect load balance by construction — no auxiliary
load balancing loss is needed.

Key properties:
- Every expert processes exactly `capacity` tokens per forward pass.
- No token dropping: auxiliary loss is 0.0 always.
- Some tokens may be chosen by fewer or more than `top_k` experts
  (popular tokens may be picked by many, unpopular by none).

Usage:
    # Same interface as SparseMoEFFN:
    ffn = ExpertChoiceFFN(config, moe_cfg)
    output, aux_loss = ffn(x)   # aux_loss is always 0.0
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoEConfig


class ExpertChoiceFFN(nn.Module):
    """MoE with expert-choice routing.

    Each expert selects exactly `capacity` tokens from the sequence, where:
        capacity = ceil(N * top_k / n_experts)   (N = batch * seq_len)

    This guarantees:
    - Every expert processes exactly `capacity` tokens.
    - No load balancing loss needed (aux_loss is always 0.0).
    - Some tokens may be processed by fewer or more experts than `top_k`
      because selection is per-expert, not per-token.

    Args:
        config: AureliusConfig (uses d_model).
        moe_cfg: MoEConfig (uses n_experts, top_k).
    """

    def __init__(self, config, moe_cfg: MoEConfig | None = None) -> None:
        super().__init__()
        from .ffn import SwiGLUFFN

        self.moe_cfg = moe_cfg or MoEConfig()
        self.n_experts = self.moe_cfg.n_experts
        self.top_k = self.moe_cfg.top_k

        # Expert networks — each is a full SwiGLUFFN (same as SparseMoEFFN)
        self.experts = nn.ModuleList([
            SwiGLUFFN(config) for _ in range(self.n_experts)
        ])

        # Router: token hidden state → per-expert logit
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route using expert-choice: each expert picks its top tokens.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (output, aux_loss):
            - output: (batch, seq_len, d_model) — positions not chosen by any
              expert remain zero; the residual connection in the transformer
              passes the original token representation through.
            - aux_loss: scalar 0.0 (load balance is perfect by construction).
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)           # (N, D)
        N = x_flat.shape[0]

        # Router scores: (N, n_experts)
        router_logits = self.router(x_flat)

        # Expert-choice transpose: each expert scores all N tokens
        # scores[i, j] = affinity of expert i for token j
        scores = router_logits.T        # (n_experts, N)

        # Each expert selects exactly `capacity` tokens
        capacity = math.ceil(N * self.top_k / self.n_experts)
        # Clamp to N so we never request more tokens than exist
        capacity = min(capacity, N)

        output = torch.zeros_like(x_flat)   # (N, D)

        for i, expert in enumerate(self.experts):
            # Top-`capacity` tokens for expert i
            expert_scores = scores[i]                                   # (N,)
            top_vals, top_indices = torch.topk(expert_scores, capacity) # both (capacity,)

            # Softmax over selected tokens (normalize the selection weights)
            expert_weights = F.softmax(top_vals, dim=0)                 # (capacity,)

            # Run expert on chosen tokens
            chosen_tokens = x_flat[top_indices]                         # (capacity, D)
            expert_out = expert(chosen_tokens)                          # (capacity, D)

            # Weighted accumulate into output
            output.index_add_(
                0,
                top_indices,
                expert_weights.unsqueeze(1) * expert_out,
            )

        output = output.view(B, S, D)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return output, aux_loss

    def token_coverage(self, x: torch.Tensor) -> torch.Tensor:
        """Count how many experts chose each token.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            coverage: (batch * seq_len,) integer tensor.
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]

        router_logits = self.router(x_flat)   # (N, n_experts)
        scores = router_logits.T              # (n_experts, N)

        capacity = math.ceil(N * self.top_k / self.n_experts)
        capacity = min(capacity, N)

        coverage = torch.zeros(N, dtype=torch.long, device=x.device)
        for i in range(self.n_experts):
            _, top_indices = torch.topk(scores[i], capacity)
            coverage.index_add_(0, top_indices, torch.ones(capacity, dtype=torch.long, device=x.device))

        return coverage
