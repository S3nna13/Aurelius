"""TEAL — Training-Free Activation Sparsity wrapper for FFN modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class TEALSparsityWrapper(nn.Module):
    """Wraps an FFN module with training-free activation sparsity.

    A fixed fraction of the smallest-magnitude activations are zeroed out
    before being passed to the underlying FFN.
    """

    def __init__(self, ffn_module: nn.Module, sparsity: float = 0.4):
        super().__init__()
        self.ffn = ffn_module
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = torch.quantile(x.abs(), self.sparsity)
        x = x * (x.abs() >= threshold)
        return self.ffn(x)
