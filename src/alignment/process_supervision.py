"""Process supervision for step-level verifier training.

Based on Lightman et al., arXiv:2305.20050 "Let's Verify Step by Step".
The paper supervises a solution as a sequence of steps ``z = (z_1, ..., z_T)``
with binary step labels ``y = (y_1, ..., y_T)``. This module implements a
minimal neural verifier that maps step representations to verifier logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ProcessSupervisionOutput:
    """Verifier outputs for a padded step sequence."""

    v: Tensor
    p: Tensor
    loss: Tensor | None = None


def reference_process_supervision_loss(v: Tensor, y: Tensor, m: Tensor) -> Tensor:
    """Reference masked Bernoulli NLL for step labels.

    Uses the stable logistic-loss identity
    ``softplus(v) - y * v`` averaged over valid steps ``m``.
    """
    if v.dim() != 2:
        raise ValueError(f"v must have shape (B, T), got {tuple(v.shape)}")
    if y.shape != v.shape:
        raise ValueError(f"y must match v shape, got {tuple(y.shape)} vs {tuple(v.shape)}")
    if m.shape != v.shape:
        raise ValueError(f"m must match v shape, got {tuple(m.shape)} vs {tuple(v.shape)}")

    m = m.to(dtype=torch.bool)
    if not m.any():
        return v.sum() * 0.0

    y = y.to(dtype=v.dtype)
    loss = F.softplus(v) - y * v
    return loss[m].mean()


def process_supervision_loss(v: Tensor, y: Tensor, m: Tensor) -> Tensor:
    """Masked BCE loss for process supervision labels ``y`` over steps ``z``."""
    if v.dim() != 2:
        raise ValueError(f"v must have shape (B, T), got {tuple(v.shape)}")
    if y.shape != v.shape:
        raise ValueError(f"y must match v shape, got {tuple(y.shape)} vs {tuple(v.shape)}")
    if m.shape != v.shape:
        raise ValueError(f"m must match v shape, got {tuple(m.shape)} vs {tuple(v.shape)}")

    m = m.to(dtype=torch.bool)
    if not m.any():
        return v.sum() * 0.0

    y = y.to(dtype=v.dtype)
    return F.binary_cross_entropy_with_logits(v[m], y[m], reduction="mean")


class ProcessSupervision(nn.Module):
    """Step-level verifier trained with process supervision.

    Args:
        d_model: Hidden size of each step representation ``z_t``.

    Inputs:
        z: Step representations with shape ``(B, T, d_model)``.
        y: Optional binary labels with shape ``(B, T)``.
        m: Optional boolean mask with shape ``(B, T)``.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.w = nn.Linear(d_model, 1)

    def forward(
        self,
        z: Tensor,
        y: Tensor | None = None,
        m: Tensor | None = None,
    ) -> ProcessSupervisionOutput:
        if z.dim() != 3:
            raise ValueError(f"z must have shape (B, T, d_model), got {tuple(z.shape)}")

        v = self.w(z).squeeze(-1)
        p = torch.sigmoid(v)

        loss = None
        if y is not None:
            if m is None:
                m = torch.ones_like(y, dtype=torch.bool)
            loss = process_supervision_loss(v, y, m)
        elif m is not None and m.shape != v.shape:
            raise ValueError(f"m must match v shape, got {tuple(m.shape)} vs {tuple(v.shape)}")

        return ProcessSupervisionOutput(v=v, p=p, loss=loss)
