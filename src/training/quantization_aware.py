"""Quantization-Aware Training (QAT) with fake quantization.

Implements fake quantization (quantize then immediately dequantize) during
training so the model learns to tolerate quantization noise. Gradients flow
through the rounding operation via the Straight-Through Estimator (STE).

Usage:
    config = QATConfig(n_bits=8, symmetric=True, per_channel=False)
    model = nn.Sequential(nn.Linear(16, 8))
    n_replaced = apply_qat(model, config)
    # Train as normal -- weights are fake-quantized each forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class QATConfig:
    """Configuration for quantization-aware training.

    Attributes:
        n_bits: Number of quantization bits (e.g. 8 for INT8).
        symmetric: If True use symmetric quantization (zero_point=0).
        per_channel: If True compute scale per output channel; otherwise global.
        quant_min: Minimum quantized integer value. Auto-computed when None.
        quant_max: Maximum quantized integer value. Auto-computed when None.
    """
    n_bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None

    def __post_init__(self) -> None:
        if self.quant_min is None:
            if self.symmetric:
                self.quant_min = -(2 ** (self.n_bits - 1) - 1)
            else:
                self.quant_min = 0
        if self.quant_max is None:
            if self.symmetric:
                self.quant_max = 2 ** (self.n_bits - 1) - 1
            else:
                self.quant_max = 2 ** self.n_bits - 1


def compute_quantization_params(
    x: torch.Tensor,
    n_bits: int,
    symmetric: bool,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute quantization scale and zero-point for a tensor.

    Args:
        x: Input tensor.
        n_bits: Number of quantization bits.
        symmetric: Use symmetric quantization if True.
        per_channel: Compute per output-channel params when True.
        channel_dim: Which dimension represents output channels.

    Returns:
        (scale, zero_point) -- tensors broadcastable to x.
    """
    if per_channel:
        # Reduce over all dims except channel_dim
        reduce_dims = [i for i in range(x.dim()) if i != channel_dim]
        if symmetric:
            abs_max = x.abs()
            for d in sorted(reduce_dims, reverse=True):
                abs_max = abs_max.amax(dim=d)
            scale = abs_max.clamp(min=1e-8) / (2 ** (n_bits - 1) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            x_min = x.clone()
            x_max = x.clone()
            for d in sorted(reduce_dims, reverse=True):
                x_min = x_min.amin(dim=d)
                x_max = x_max.amax(dim=d)
            scale = (x_max - x_min).clamp(min=1e-8) / (2 ** n_bits - 1)
            zero_point = torch.round(-x_min / scale)
    else:
        if symmetric:
            abs_max = x.abs().max().clamp(min=1e-8)
            scale = abs_max / (2 ** (n_bits - 1) - 1)
            zero_point = torch.zeros(1, dtype=x.dtype, device=x.device)
        else:
            x_min = x.min()
            x_max = x.max()
            scale = (x_max - x_min).clamp(min=1e-8) / (2 ** n_bits - 1)
            zero_point = torch.round(-x_min / scale)

    return scale, zero_point


class _FakeQuantizeFunction(torch.autograd.Function):
    """Fake quantize with Straight-Through Estimator for the backward pass."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
    ) -> torch.Tensor:
        x_scaled = x / scale + zero_point
        x_q = torch.clamp(torch.round(x_scaled), quant_min, quant_max)
        x_dq = (x_q - zero_point) * scale
        return x_dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: pass gradient through unchanged; no gradient for non-tensor params
        return grad_output, None, None, None, None


def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    """Fake-quantize a tensor: quantize then immediately dequantize.

    During the forward pass the tensor is snapped to the quantization grid.
    During the backward pass the gradient flows through unchanged (STE).

    Args:
        x: Input floating-point tensor.
        scale: Quantization scale (broadcastable to x).
        zero_point: Quantization zero-point (broadcastable to x).
        quant_min: Minimum representable integer value.
        quant_max: Maximum representable integer value.

    Returns:
        Dequantized tensor with the same shape and dtype as x.
    """
    return _FakeQuantizeFunction.apply(x, scale, zero_point, quant_min, quant_max)


class FakeQuantize(nn.Module):
    """Fake-quantization module that can be inserted into any nn.Module graph.

    In train mode: computes quantization parameters from the input tensor
    and applies fake quantization (quantize + dequantize with STE gradient).
    In eval mode: returns the input unchanged.

    Args:
        config: QATConfig controlling bitwidth, symmetry and per-channel mode.
    """

    def __init__(self, config: QATConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        scale, zero_point = compute_quantization_params(
            x,
            n_bits=self.config.n_bits,
            symmetric=self.config.symmetric,
            per_channel=self.config.per_channel,
        )
        return fake_quantize(
            x, scale, zero_point,
            self.config.quant_min,
            self.config.quant_max,
        )

    def quant_error(self, x: torch.Tensor) -> float:
        """Mean absolute error between x and its fake-quantized version.

        Always computes in train-like mode regardless of self.training.

        Args:
            x: Input tensor.

        Returns:
            Mean absolute quantization error as a Python float.
        """
        scale, zero_point = compute_quantization_params(
            x,
            n_bits=self.config.n_bits,
            symmetric=self.config.symmetric,
            per_channel=self.config.per_channel,
        )
        x_fq = fake_quantize(
            x, scale, zero_point,
            self.config.quant_min,
            self.config.quant_max,
        )
        return (x - x_fq).abs().mean().item()


class QATLinear(nn.Module):
    """nn.Linear with fake quantization applied to weights during training.

    The underlying weight is stored in full precision. At each forward pass
    during training the weight is fake-quantized before the matrix multiply,
    so gradients flow back into the full-precision weight via STE.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        config: QATConfig controlling quantization behaviour.
        bias: If True, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QATConfig,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_fake_quant = FakeQuantize(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fq = self.weight_fake_quant(self.linear.weight)
        return F.linear(x, w_fq, self.linear.bias)


def apply_qat(
    model: nn.Module,
    config: QATConfig,
    target_types: tuple = (nn.Linear,),
) -> int:
    """Replace target module types with QATLinear in-place.

    Args:
        model: Model to modify.
        config: Quantization configuration to use for all replaced layers.
        target_types: Tuple of module types to replace (default: nn.Linear).

    Returns:
        Number of modules replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, tuple(target_types)):
            continue
        # Navigate to the parent module
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            attr = parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        qat_linear = QATLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            config=config,
            bias=module.bias is not None,
        )
        # Copy pre-trained weights
        qat_linear.linear.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            qat_linear.linear.bias.data.copy_(module.bias.data)

        setattr(parent, attr, qat_linear)
        replaced += 1

    return replaced


def compute_model_quantization_error(
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[str, float]:
    """Measure the effect of fake quantization on model output.

    Runs the model in train mode (fake quant active) and eval mode (no quant)
    and compares the outputs.

    Args:
        model: Model containing QATLinear / FakeQuantize layers.
        x: Input tensor.

    Returns:
        Dictionary with 'mean_abs_error' and 'max_abs_error' keys.
    """
    model.train()
    with torch.no_grad():
        out_train = model(x)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    diff = (out_train - out_eval).abs()
    return {
        "mean_abs_error": diff.mean().item(),
        "max_abs_error": diff.max().item(),
    }
