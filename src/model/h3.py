"""Simplified H3-style hybrid long-convolution block."""

from __future__ import annotations

import torch
import torch.nn as nn


class H3Block(nn.Module):
    """A lightweight gated long-convolution module."""

    def __init__(self, d_model: int, kernel_size: int = 5) -> None:
        super().__init__()
        if d_model <= 0 or kernel_size <= 0:
            raise ValueError("d_model and kernel_size must be positive")
        padding = kernel_size - 1
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=padding,
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated convolution over the sequence dimension."""
        if x.dim() != 3:
            raise ValueError(f"x must be 3D, got {tuple(x.shape)}")
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        conv = self.depthwise_conv(value.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        output = torch.sigmoid(gate) * conv
        return self.out_proj(output)
