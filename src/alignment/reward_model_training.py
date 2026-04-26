"""Multi-head reward model with text and tabular feature fusion.

Combines a text encoder projection with structured tabular features
to produce quality scores and risk classifications simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.alignment import ALIGNMENT_REGISTRY


@dataclass
class RewardModelFeatures:
    """Structured tabular features accompanying a text sample."""

    task_complexity: float = 0.5
    domain: str = "general"
    top_k_sim: float = 0.0
    citation_coverage: float = 0.0
    tool_call_count: int = 0
    failure_count: int = 0
    safety_flag_count: int = 0

    def to_tensor(self) -> Tensor:
        """Convert features to a float tensor of shape (7,).

        The ``domain`` field is encoded as a hash-based float in [0, 1].
        """
        domain_hash = float(hash(self.domain) % 10_000) / 10_000.0
        return torch.tensor(
            [
                self.task_complexity,
                domain_hash,
                self.top_k_sim,
                self.citation_coverage,
                float(self.tool_call_count),
                float(self.failure_count),
                float(self.safety_flag_count),
            ],
            dtype=torch.float32,
        )


class MultiHeadRewardModel(nn.Module):
    """Reward model that fuses text hidden states with tabular features.

    Produces two outputs:
    - ``quality_score``: scalar quality rating in (-inf, inf)
    - ``risk_logits``: 4-class risk classification logits
    """

    def __init__(self, text_hidden: int = 768) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_hidden, 256)
        self.tabular_proj = nn.Linear(7, 256)
        self.quality_head = nn.Linear(512, 1)
        self.risk_head = nn.Linear(512, 4)

    def forward(
        self,
        text_hidden: Tensor,
        features_vec: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute quality score and risk logits.

        Args:
            text_hidden: ``(B, T, text_hidden_size)`` — encoder hidden states.
            features_vec: ``(B, 7)`` — tabular feature vector.

        Returns:
            Tuple of ``(quality_score, risk_logits)`` where ``quality_score``
            has shape ``(B, 1)`` and ``risk_logits`` has shape ``(B, 4)``.
        """
        text_repr = self.text_proj(text_hidden.mean(dim=1))  # (B, 256)
        tabular_repr = self.tabular_proj(features_vec)  # (B, 256)
        fused = torch.cat([text_repr, tabular_repr], dim=-1)  # (B, 512)
        quality_score = self.quality_head(fused)  # (B, 1)
        risk_logits = self.risk_head(fused)  # (B, 4)
        return quality_score, risk_logits


ALIGNMENT_REGISTRY["multi_head_rm"] = MultiHeadRewardModel
