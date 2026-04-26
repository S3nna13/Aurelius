"""Quantization-Aware Training (QAT) with Straight-Through Estimator.

Inserts fake quantization into the forward pass during training:
  forward: W_deq = dequantize(quantize(W))  # introduces quantization noise
  backward: gradient passes through round() unchanged (STE)

This makes the model robust to INT8/INT4 quantization, yielding better
accuracy than post-training quantization (PTQ).

Usage:
    # Wrap model for QAT
    qat_model = prepare_qat(model, QuantConfig(bits=8))

    # Train as normal
    loss, _, _ = qat_model(input_ids, labels=labels)
    loss.backward()

    # Convert to actual quantized model for inference
    quantized_model = convert_qat(qat_model)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QATConfig:
    bits: int = 8  # 8 or 4
    per_channel: bool = True  # per output-channel quantization
    group_size: int = 128  # for 4-bit grouped quantization
    skip_modules: tuple[str, ...] = ("lm_head", "embed")


class _StraightThroughRound(torch.autograd.Function):
    """Round function with straight-through gradient estimator.

    Forward: y = round(x)
    Backward: dy/dx = 1 (identity, straight-through)
    """

    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


_ste_round = _StraightThroughRound.apply


def fake_quantize_int8(
    weight: torch.Tensor,
    per_channel: bool = True,
) -> torch.Tensor:
    """Simulate INT8 quantization with straight-through gradient.

    Quantizes and immediately dequantizes, introducing rounding noise.
    Gradients flow through via STE.

    Args:
        weight: (out, in) float weight tensor.
        per_channel: Per output-channel scale.

    Returns:
        (out, in) float tensor — same dtype, but values snapped to INT8 grid.
    """
    if per_channel:
        scale = weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
    else:
        scale = weight.abs().max().clamp(min=1e-8) / 127.0

    # Quantize with STE
    w_scaled = weight / scale
    w_rounded = _ste_round(w_scaled)
    w_clamped = w_rounded.clamp(-127, 127)

    # Dequantize
    return w_clamped * scale


def fake_quantize_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Simulate INT4 quantization with straight-through gradient.

    Uses asymmetric quantization within groups of group_size.

    Args:
        weight: (out, in) float weight tensor.
        group_size: Quantization group size.

    Returns:
        (out, in) float tensor snapped to INT4 grid.
    """
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        # Fall back to per-tensor if not divisible
        scale = (weight.max() - weight.min()).clamp(min=1e-8) / 15.0
        zero_point = -weight.min() / scale.clamp(min=1e-8)
        w_q = _ste_round(weight / scale + zero_point).clamp(0, 15)
        return (w_q - zero_point) * scale

    n_groups = in_features // group_size
    w_grouped = weight.reshape(out_features, n_groups, group_size)

    w_min = w_grouped.min(dim=-1, keepdim=True).values
    w_max = w_grouped.max(dim=-1, keepdim=True).values

    scale = (w_max - w_min).clamp(min=1e-8) / 15.0
    zero_point = -w_min / scale.clamp(min=1e-8)

    w_q = _ste_round(w_grouped / scale + zero_point).clamp(0, 15)
    w_deq = (w_q - zero_point) * scale

    return w_deq.reshape(out_features, in_features)


class QATLinear(nn.Module):
    """nn.Linear wrapper with fake quantization in the forward pass.

    During training: weight is fake-quantized (round + dequantize with STE)
    before computing the linear transformation. After calling convert_qat(),
    this becomes a real QuantizedLinear with stored int8/int4 weights.

    Args:
        weight: Original float weight (will be stored as nn.Parameter).
        bias: Optional bias.
        cfg: QAT configuration.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        cfg: QATConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
        self.out_features, self.in_features = weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute linear transformation with fake-quantized weights."""
        if self.cfg.bits == 8:
            w_fq = fake_quantize_int8(self.weight, self.cfg.per_channel)
        else:
            w_fq = fake_quantize_int4(self.weight, self.cfg.group_size)
        return F.linear(x, w_fq, self.bias)


def prepare_qat(
    model: nn.Module,
    cfg: QATConfig | None = None,
) -> nn.Module:
    """Replace nn.Linear layers with QATLinear for quantization-aware training.

    Modifies the model in-place and returns it.

    Args:
        model: Model to prepare for QAT.
        cfg: QAT configuration.

    Returns:
        The same model with QATLinear layers.
    """
    if cfg is None:
        cfg = QATConfig()

    for name, module in list(model.named_modules()):
        if any(skip in name for skip in cfg.skip_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if cfg.bits == 4 and module.in_features % cfg.group_size != 0:
            continue

        # Navigate to parent
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        qat_linear = QATLinear(
            weight=module.weight.data,
            bias=module.bias.data if module.bias is not None else None,
            cfg=cfg,
        )
        setattr(parent, attr, qat_linear)

    return model


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert QATLinear layers to QuantizedLinear for inference.

    After QAT training, call this to produce the final quantized model
    with actual INT8/INT4 storage (not fake quantization).

    Args:
        model: Model with QATLinear layers (after QAT training).

    Returns:
        Model with QuantizedLinear layers (real quantization).
    """
    from src.inference.quantize import QuantConfig, QuantizedLinear

    for name, module in list(model.named_modules()):
        if not isinstance(module, QATLinear):
            continue

        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        q_cfg = QuantConfig(
            bits=module.cfg.bits,
            per_channel=module.cfg.per_channel,
            group_size=module.cfg.group_size,
        )
        q_linear = QuantizedLinear(
            weight=module.weight.data,
            bias=module.bias.data if module.bias is not None else None,
            cfg=q_cfg,
        )
        setattr(parent, attr, q_linear)

    return model
