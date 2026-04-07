"""Weight-only quantization for Aurelius linear layers.

Supports:
- INT8 symmetric (absmax per-channel): W_q = round(W / scale), scale = max(|W_row|) / 127
- INT4 grouped (absmax per group): group W into blocks of group_size, same formula at 4-bit

Dequantization happens on-the-fly during forward(); inference-time memory savings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class QuantConfig:
    bits: int = 8              # 8 or 4
    per_channel: bool = True   # per output-channel (True) or per-tensor (False)
    group_size: int = 128      # used only when bits=4 (groups within each row)


def quantize_tensor_int8(
    weight: torch.Tensor,
    per_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization of a weight tensor.

    Args:
        weight: (out_features, in_features) float weight.
        per_channel: If True, one scale per output channel (row).
                     If False, one scale for the entire tensor.

    Returns:
        (weight_q, scale) where:
        - weight_q: (out_features, in_features) int8 tensor
        - scale: (out_features, 1) if per_channel else (1,) float32
    """
    if per_channel:
        scale = weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
    else:
        scale = weight.abs().max().clamp(min=1e-8) / 127.0

    weight_q = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return weight_q, scale.float()


def dequantize_int8(
    weight_q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize INT8 weights to float32.

    Args:
        weight_q: (out_features, in_features) int8.
        scale: (out_features, 1) or (1,) float32.

    Returns:
        (out_features, in_features) float32.
    """
    return weight_q.float() * scale


def quantize_tensor_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric INT4 quantization with groups within each row.

    Groups weight rows into blocks of group_size, quantizes each block
    independently using zero-point quantization.

    Args:
        weight: (out_features, in_features) float weight.
        group_size: Number of elements per quantization group.

    Returns:
        (weight_q, scale, zero_point) where:
        - weight_q: (out_features * in_features / group_size, group_size) uint8 packed as int8
        - scale: (out_features, n_groups) float32
        - zero_point: (out_features, n_groups) float32

    Note: weight_q stores values in [0, 15] packed into int8 (lower 4 bits).
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )
    n_groups = in_features // group_size

    # Reshape to (out_features, n_groups, group_size)
    w_grouped = weight.reshape(out_features, n_groups, group_size)

    # Per-group min/max for asymmetric quantization
    w_min = w_grouped.min(dim=-1).values  # (out, n_groups)
    w_max = w_grouped.max(dim=-1).values  # (out, n_groups)

    scale = (w_max - w_min).clamp(min=1e-8) / 15.0  # (out, n_groups)
    zero_point = -w_min / scale.clamp(min=1e-8)      # (out, n_groups)

    # Quantize: q = round(w/scale + zero_point), clamp to [0, 15]
    scale_exp = scale.unsqueeze(-1)    # (out, n_groups, 1)
    zp_exp = zero_point.unsqueeze(-1)  # (out, n_groups, 1)
    weight_q = (w_grouped / scale_exp + zp_exp).round().clamp(0, 15).to(torch.uint8)

    # Flatten back to (out_features, in_features)
    weight_q = weight_q.reshape(out_features, in_features)

    return weight_q, scale.float(), zero_point.float()


def dequantize_int4(
    weight_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize INT4 weights.

    Args:
        weight_q: (out_features, in_features) uint8.
        scale: (out_features, n_groups) float32.
        zero_point: (out_features, n_groups) float32.
        group_size: Group size used during quantization.

    Returns:
        (out_features, in_features) float32.
    """
    out_features, in_features = weight_q.shape
    n_groups = in_features // group_size

    w_grouped = weight_q.float().reshape(out_features, n_groups, group_size)
    scale_exp = scale.unsqueeze(-1)
    zp_exp = zero_point.unsqueeze(-1)

    w_deq = (w_grouped - zp_exp) * scale_exp
    return w_deq.reshape(out_features, in_features)


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with quantized weights.

    Stores quantized weights (INT8 or INT4) and dequantizes on-the-fly
    during forward pass. Reduces memory usage while maintaining original dtype
    for activations.

    Args:
        weight: Original float weight tensor (out, in).
        bias: Optional bias tensor (out,).
        cfg: Quantization configuration.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        cfg: QuantConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_features, self.in_features = weight.shape

        if cfg.bits == 8:
            weight_q, scale = quantize_tensor_int8(weight, cfg.per_channel)
            self.register_buffer("weight_q", weight_q)
            self.register_buffer("scale", scale)
        elif cfg.bits == 4:
            weight_q, scale, zp = quantize_tensor_int4(weight, cfg.group_size)
            self.register_buffer("weight_q", weight_q)
            self.register_buffer("scale", scale)
            self.register_buffer("zero_point", zp)
        else:
            raise ValueError(f"Unsupported bits: {cfg.bits}. Use 8 or 4.")

        if bias is not None:
            self.register_buffer("bias", bias.clone())
        else:
            self.bias = None

    def dequantize(self) -> torch.Tensor:
        """Return the dequantized weight as float32."""
        if self.cfg.bits == 8:
            return dequantize_int8(self.weight_q, self.scale)
        else:
            return dequantize_int4(self.weight_q, self.scale, self.zero_point, self.cfg.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weight and compute linear transformation."""
        weight = self.dequantize().to(x.dtype)
        return F.linear(x, weight, self.bias)


def quantize_model(
    model: nn.Module,
    cfg: QuantConfig | None = None,
    skip_modules: tuple[str, ...] = ("lm_head", "embed"),
) -> nn.Module:
    """Replace all nn.Linear layers with QuantizedLinear, in-place.

    Args:
        model: Model to quantize.
        cfg: Quantization configuration.
        skip_modules: Module name substrings to skip (e.g., lm_head for tied embeddings).

    Returns:
        The same model with quantized layers (in-place modification).
    """
    if cfg is None:
        cfg = QuantConfig()

    for name, module in list(model.named_modules()):
        # Skip specified modules
        if any(skip in name for skip in skip_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Skip if in_features not divisible by group_size (INT4)
        if cfg.bits == 4 and module.in_features % cfg.group_size != 0:
            continue

        # Navigate to the parent module
        parent_name = name.rsplit(".", 1)
        if len(parent_name) == 1:
            parent = model
            attr = parent_name[0]
        else:
            parent = model
            for part in parent_name[0].split("."):
                parent = getattr(parent, part)
            attr = parent_name[1]

        q_linear = QuantizedLinear(
            weight=module.weight.data.detach(),
            bias=module.bias.data.detach() if module.bias is not None else None,
            cfg=cfg,
        )
        setattr(parent, attr, q_linear)

    return model


def estimate_memory_savings(model: nn.Module, bits: int = 8) -> dict[str, float]:
    """Estimate memory savings from quantization.

    Returns dict with original_mb, quantized_mb, savings_pct.
    """
    total_params = 0
    quantizable_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n = module.weight.numel()
            total_params += n
            quantizable_params += n
        elif isinstance(module, QuantizedLinear):
            quantizable_params += module.weight_q.numel()
            total_params += module.out_features * module.in_features

    original_bytes = total_params * 2  # bfloat16 = 2 bytes
    quantized_bytes = quantizable_params * (bits / 8) + (total_params - quantizable_params) * 2

    return {
        "original_mb": original_bytes / 1e6,
        "quantized_mb": quantized_bytes / 1e6,
        "savings_pct": (1 - quantized_bytes / original_bytes) * 100,
    }
