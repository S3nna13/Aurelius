from __future__ import annotations

import torch
from torch import Tensor


class PrecisionFusion:
    """Bayesian inverse-variance weighting over multiple reward signals."""

    def __init__(self, n_signals: int, eps: float = 1e-6) -> None:
        self.n_signals = n_signals
        self.eps = eps

    def fuse(self, means: list[Tensor], stds: list[Tensor]) -> Tensor:
        """Fuse reward signals weighted by precision (1/variance).

        Args:
            means: list of (B,) reward mean tensors, one per signal.
            stds:  list of (B,) reward std tensors, one per signal.

        Returns:
            (B,) fused reward.
        """
        means_stack = torch.stack(means, dim=0)  # (n_signals, B)
        stds_stack = torch.stack(stds, dim=0).clamp(min=self.eps)  # (n_signals, B)
        precision = 1.0 / (stds_stack**2 + self.eps)  # (n_signals, B)
        weights = precision / precision.sum(dim=0, keepdim=True)  # (n_signals, B) normalized
        return (weights * means_stack).sum(dim=0)  # (B,)
