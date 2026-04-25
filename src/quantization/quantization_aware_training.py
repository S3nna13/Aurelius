"""Quantization-Aware Training (QAT) utilities for Aurelius.

Provides differentiable fake-quantization (STE) and a wrapper that adds
fake-quant nodes around an nn.Linear layer's weights and activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.autograd import Function

# Registry — import from awq if available, else define locally
try:
    from .awq_quantizer import QUANTIZATION_REGISTRY
except ImportError:
    QUANTIZATION_REGISTRY: dict[str, object] = {}  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QATConfig:
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    ema_decay: float = 0.9


# ---------------------------------------------------------------------------
# STE (Straight-Through Estimator) autograd function
# ---------------------------------------------------------------------------

class _STEFunction(Function):
    """Round in forward; identity gradient in backward (STE)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.detach().round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Straight-Through: pass gradient unchanged
        return grad_output


# ---------------------------------------------------------------------------
# FakeQuantize module
# ---------------------------------------------------------------------------

class FakeQuantize(nn.Module):
    """Differentiable fake-quantize operator with STE gradient.

    Performs per-tensor or per-channel affine quantization in the forward pass
    (using STE rounding) and lets gradients flow straight through on the backward
    pass.
    """

    def __init__(self, config: QATConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else QATConfig()
        bits = self.config.bits
        if self.config.symmetric:
            self.q_min = -(2 ** (bits - 1))
            self.q_max = (2 ** (bits - 1)) - 1
        else:
            self.q_min = 0
            self.q_max = (2 ** bits) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Fake-quantize x; STE gradient flows through rounding."""
        if self.config.symmetric:
            x_max = x.abs().max().clamp(min=1e-8)
            scale = x_max / self.q_max
            zero_point = 0.0
        else:
            x_min = x.min()
            x_max = x.max()
            scale = (x_max - x_min).clamp(min=1e-8) / (self.q_max - self.q_min)
            zero_point = self.q_min - (x_min / scale).round()

        # Quantise (STE round)
        x_scaled = x / scale + zero_point
        x_rounded = _STEFunction.apply(x_scaled)
        x_clamped = x_rounded.clamp(self.q_min, self.q_max)
        # Dequantise
        x_dequant = (x_clamped - zero_point) * scale
        return x_dequant


# ---------------------------------------------------------------------------
# QATWrapper
# ---------------------------------------------------------------------------

class QATWrapper(nn.Module):
    """Wraps an nn.Linear with fake-quantize nodes on weights and activations.

    Compatible with standard PyTorch training loops.
    """

    def __init__(
        self,
        linear: nn.Linear,
        config: QATConfig | None = None,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.config = config if config is not None else QATConfig()
        self.weight_fq = FakeQuantize(self.config)
        self.act_fq = FakeQuantize(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Fake-quantize activations
        x_fq = self.act_fq(x)
        # Fake-quantize weights (does not modify underlying parameter)
        w_fq = self.weight_fq(self.linear.weight)
        bias = self.linear.bias
        return nn.functional.linear(x_fq, w_fq, bias)

    def fold_bn(
        self,
        bn_mean: torch.Tensor,
        bn_var: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> None:
        """Absorb BatchNorm statistics into the wrapped linear weight and bias.

        Modifies self.linear.weight (and bias) in-place.

        Args:
            bn_mean:   (out_features,) BatchNorm running mean.
            bn_var:    (out_features,) BatchNorm running variance.
            bn_weight: (out_features,) BatchNorm gamma (scale).
            bn_bias:   (out_features,) BatchNorm beta (shift).
            eps:       Numerical stability epsilon.
        """
        std = (bn_var + eps).sqrt()
        scale = bn_weight / std  # (out_features,)

        with torch.no_grad():
            # Fold scale into weight rows
            self.linear.weight.mul_(scale.unsqueeze(1))
            # Fold into bias
            if self.linear.bias is not None:
                self.linear.bias.mul_(scale).add_(bn_bias - bn_mean * scale)
            else:
                folded_bias = bn_bias - bn_mean * scale
                self.linear.bias = nn.Parameter(folded_bias)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
QUANTIZATION_REGISTRY["qat"] = QATWrapper
