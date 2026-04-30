"""Value head for RLHF PPO training.

Wraps an AureliusTransformer backbone with a scalar value estimator
used for GAE advantage computation during online RLHF.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Policy wrapper that adds a value estimation head.

    Attributes:
        backbone: The base transformer model.
        value_head: Linear layer projecting final hidden state to scalar value.
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        _hidden = (
            hidden_dim
            or getattr(backbone, "d_model", None)
            or getattr(getattr(backbone, "config", None), "d_model", 768)
        )
        self.value_head = nn.Linear(_hidden, 1)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, hidden_states, values).

        Args:
            input_ids: Token IDs of shape (B, S).

        Returns:
            logits: (B, S, V) from the backbone LM head.
            hidden: (B, S, D) final-layer hidden states.
            values: (B, S) scalar value estimates.
        """
        backbone_out = self.backbone(input_ids)
        if isinstance(backbone_out, tuple):
            logits, hidden = (
                backbone_out[0],
                backbone_out[1] if len(backbone_out) > 1 else backbone_out[0],
            )
        else:
            logits = backbone_out
            hidden = backbone_out

        if hidden.dim() == 3:
            values = self.value_head(hidden).squeeze(-1)
        else:
            values = self.value_head(hidden).squeeze(-1)

        return logits, hidden, values
