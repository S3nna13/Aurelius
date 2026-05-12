"""Multi-Token Prediction (MTP) — DeepSeek-V3 style.

Trains D additional prediction heads that cascade: the projected representation
from head i feeds head i+1.  At inference the main path is unchanged; MTP is a
training-only augmentation.

Reference: DeepSeek-V3 (2024) — MTP improves overall performance on evaluation
benchmarks by predicting D additional tokens at each position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig
from .rms_norm import RMSNorm

# ---------------------------------------------------------------------------
# MTPConfig
# ---------------------------------------------------------------------------


@dataclass
class MTPConfig:
    """Configuration for Multi-Token Prediction heads."""

    depth: int = 1
    lambda_mtp: float = 0.3


# ---------------------------------------------------------------------------
# MultiTokenPredictionHead
# ---------------------------------------------------------------------------


class MultiTokenPredictionHead(nn.Module):
    """DeepSeek-V3 style multi-token prediction head.

    Architecture:
        - RMSNorm(d_model)
        - Shared linear projection (d_model -> d_model)
        - D independent output heads (d_model -> vocab_size)
        - Cascade: head i+1 receives the projected output of head i

    For depth=1: predicts token at position t+1 from hidden state at t.
    For depth>1: each subsequent head predicts the next future token.

    Args:
        config: AureliusConfig
        mtp_config: MTPConfig (optional)
    """

    def __init__(
        self,
        config: AureliusConfig,
        mtp_config: MTPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mtp_config = mtp_config or MTPConfig()
        self.depth = self.mtp_config.depth

        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.shared_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(self.depth)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        """
        Args:
            hidden_states: (B, T, D) — final hidden states from backbone.
            labels: Optional (B, T) target token ids.

        Returns:
            (loss_mtp, all_logits) where all_logits has length D.
            loss_mtp is None when labels are not provided.
        """
        h = self.shared_proj(self.norm(hidden_states))
        all_logits: list[torch.Tensor] = []

        for _ in range(self.depth):
            logits = self.heads[_](h)
            all_logits.append(logits)

        if labels is None:
            return None, all_logits

        losses: list[torch.Tensor] = []
        for i, logits in enumerate(all_logits, start=1):
            if i >= logits.shape[1]:
                continue
            # Position t in logits predicts token at position t+i
            pred = logits[:, :-i, :].contiguous()
            target = labels[:, i : i + pred.shape[1]].contiguous()
            losses.append(
                F.cross_entropy(
                    pred.view(-1, pred.size(-1)),
                    target.view(-1),
                )
            )

        if not losses:
            loss = hidden_states.new_tensor(0.0)
        else:
            loss = sum(losses) / len(losses)

        return loss, all_logits

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return the raw averaged MTP loss (without lambda_mtp scaling).

        Compatible with the AureliusTransformer training loop.
        """
        loss, _ = self.forward(hidden_states, labels)
        return loss
