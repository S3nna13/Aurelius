"""MTP with Parameter Sharing — GLM-5 §3.3 (arXiv:2602.15763).

3 prediction heads share a single projection weight.
Acceptance rate: 2.76 vs 2.55 unshared (DeepSeek-V3 baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedMTPHead(nn.Module):
    """Multi-Token Prediction head with shared projection weights.

    Architecture (GLM-5 §3.3):
      shared_proj: nn.Linear(d_model, vocab_size)  — ONE weight, used by all n heads
      input_projs: n × nn.Linear(d_model, d_model) — per-head input transforms (NOT shared)

    For head i:
      logits_i = shared_proj(input_projs[i](hidden))

    Key result: acceptance rate 2.76 (shared) vs 2.55 (unshared DeepSeek-V3 baseline).
    """

    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 3) -> None:
        super().__init__()
        self.n_heads = n_heads
        # Single shared projection — all heads use the SAME weight tensor
        self.shared_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Per-head input transforms (not shared)
        self.input_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(n_heads)]
        )

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Produce per-head logit tensors.

        Args:
            hidden: Float tensor of shape [B, T, d_model].

        Returns:
            List of n_heads tensors, each of shape [B, T, vocab_size].
        """
        return [self.shared_proj(proj(hidden)) for proj in self.input_projs]

    def acceptance_rate(
        self,
        logits_list: list[torch.Tensor],  # [B, T, V] each
        targets: torch.Tensor,  # [B, T] token IDs
    ) -> float:
        """Compute speculative-decoding acceptance rate.

        For each MTP offset i (0 = next token, 1 = token+2, ...):
          Compare pred[:, :T-i-1] against targets[:, i+1:T]
          acceptance_rate = total_correct / total_tokens

        Args:
            logits_list: List of logit tensors from forward().
            targets:     Ground-truth token IDs, shape [B, T].

        Returns:
            Float in [0, 1] — fraction of speculative predictions that match.
        """
        total, accepted = 0, 0
        for i, logits in enumerate(logits_list):
            preds = logits.argmax(-1)  # [B, T]
            offset = i + 1
            if offset >= targets.shape[1]:
                continue
            gt = targets[:, offset:]  # [B, T-offset]
            pr = preds[:, : gt.shape[1]]  # [B, T-offset]
            accepted += (pr == gt).float().sum().item()
            total += gt.numel()
        return accepted / max(total, 1)
