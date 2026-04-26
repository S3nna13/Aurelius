"""Supervised cross-entropy on tool-span tokens (Toolformer-style objective).

When fine-tuning on transcripts that contain tool calls, it is useful to
up-weight or isolate loss on the JSON/XML tool segments so the model learns
reliable emission (Schick et al., Toolformer, arXiv:2302.04761).

This module exposes a small :class:`torch.nn.Module` that computes **only** the
mean NLL over positions marked by a boolean mask, ignoring label=-100 entries
everywhere else (standard LM ignore index).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToolCallSupervisionLoss(nn.Module):
    """Masked cross-entropy over tool-call span positions.

    Args:
        ignore_index: Label value to ignore (defaults to -100).
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tool_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar mean NLL over selected positions.

        logits:   ``[B, T, V]`` unnormalized scores.
        labels:   ``[B, T]`` target token ids (may include ``ignore_index``).
        tool_mask: ``[B, T]`` bool — ``True`` means "include in tool loss if
                   label is not ignored".
        """
        if logits.dim() != 3:
            raise ValueError(f"logits must be [B,T,V], got shape {tuple(logits.shape)}")
        if labels.shape != logits.shape[:2]:
            raise ValueError(
                f"labels {tuple(labels.shape)} must match logits[:2] {tuple(logits.shape[:2])}"
            )
        if tool_mask.shape != labels.shape:
            raise ValueError(
                f"tool_mask {tuple(tool_mask.shape)} must match labels {tuple(labels.shape)}"
            )
        if not tool_mask.dtype == torch.bool:
            raise TypeError("tool_mask must be a bool tensor")

        active = tool_mask & (labels != self.ignore_index)
        n = int(active.sum().item())
        if n == 0:
            raise RuntimeError(
                "tool_call_supervision_loss: zero active positions — refusing silent zero loss"
            )

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        flat_active = active.reshape(-1)

        # CE only on active rows (stable two-step masked mean)
        ce = F.cross_entropy(
            flat_logits[flat_active],
            flat_labels[flat_active],
            reduction="mean",
        )
        return ce


__all__ = ["ToolCallSupervisionLoss"]
