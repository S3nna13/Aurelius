"""INT4 weight quantization: pack two 4-bit values per byte, group-wise quantization, and dequant matmul."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class INT4Config:
    """Configuration for INT4 weight quantization."""
    group_size: int = 128           # quantization group size (rows grouped along in_features)
    symmetric: bool = True          # symmetric (zero_point=0) vs asymmetric
    pack_format: str = "row_major"  # packing order
    compute_dtype: torch.dtype = torch.float32


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 128,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize a weight tensor to 4-bit integers, group-wise along in_features.

    Args:
        weight: (out_features, in_features) float tensor.
        group_size: Number of elements per quantization group along in_features.
        symmetric: If True, use symmetric quantization (zero_point=0, values in [-8, 7]).
                   If False, use asymmetric quantization (values in [0, 15]).

    Returns:
        weight_int4: (out_features, in_features) int8 tensor.
                     Symmetric: values in [-8, 7]. Asymmetric: values in [0, 15].
        scales: (out_features, n_groups) float32 tensor.
        zero_points: (out_features, n_groups) int8 tensor, or None if symmetric.
    """
    out_features, in_features = weight.shape

    # Pad in_features to be divisible by group_size
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        weight = F.pad(weight, (0, pad))
    padded_in = weight.shape[1]

    n_groups = padded_in // group_size
    # Reshape to (out_features, n_groups, group_size)
    w = weight.reshape(out_features, n_groups, group_size).float()

    if symmetric:
        # scale = max(|w|) / 7
        abs_max = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        scales = (abs_max / 7.0).squeeze(2)  # (out_features, n_groups)
        w_scaled = (w / abs_max * 7.0).round().clamp(-8, 7)
        weight_int4 = w_scaled.reshape(out_features, padded_in).to(torch.int8)
        # Trim back to original in_features
        weight_int4 = weight_int4[:, :in_features].contiguous()
        return weight_int4, scales.float(), None
    else:
        # Asymmetric: map [w_min, w_max] -> [0, 15]
        w_min = w.amin(dim=2, keepdim=True)
        w_max = w.amax(dim=2, keepdim=True)
        scales_3d = ((w_max - w_min) / 15.0).clamp(min=1e-8)
        zero_points_float = (-w_min / scales_3d).round().clamp(0, 15)
        w_scaled = ((w / scales_3d) + zero_points_float).round().clamp(0, 15)

        scales = scales_3d.squeeze(2)  # (out_features, n_groups)
        zero_points = zero_points_float.squeeze(2).to(torch.int8)  # (out_features, n_groups)

        weight_int4 = w_scaled.reshape(out_features, padded_in).to(torch.int8)
        weight_int4 = weight_int4[:, :in_features].contiguous()
        return weight_int4, scales.float(), zero_points


def pack_int4(weight_int4: torch.Tensor) -> torch.Tensor:
    """Pack two INT4 values into one INT8 byte.

    Args:
        weight_int4: (out_features, in_features) int8 tensor with values in [0, 15].
                     If values are signed [-8, 7], they are reinterpreted as unsigned
                     by adding 8 before packing.

    Returns:
        packed: (out_features, in_features // 2) int8 tensor.
    """
    out_features, in_features = weight_int4.shape
    if in_features % 2 != 0:
        raise ValueError(f"in_features must be even for INT4 packing, got {in_features}")

    # Ensure values are in [0, 15] unsigned range
    w = weight_int4.to(torch.int32)
    # If values appear to be signed (negative values), shift to unsigned
    # For the packing, treat raw bit pattern as uint4
    low = w[:, 0::2] & 0xF
    high = w[:, 1::2] & 0xF
    packed = (low | (high << 4)).to(torch.int8)
    return packed


def unpack_int4(packed: torch.Tensor, original_in_features: int) -> torch.Tensor:
    """Unpack INT8 bytes back to INT4 values in [0, 15].

    Args:
        packed: (out_features, in_features // 2) int8 tensor.
        original_in_features: The original in_features before packing.

    Returns:
        unpacked: (out_features, original_in_features) int8 tensor with values in [0, 15].
    """
    p = packed.to(torch.int32)
    low = p & 0xF         # even columns
    high = (p >> 4) & 0xF  # odd columns

    out_features = packed.shape[0]
    # Interleave: low goes to even indices, high to odd indices
    result = torch.empty(out_features, packed.shape[1] * 2, dtype=torch.int32, device=packed.device)
    result[:, 0::2] = low
    result[:, 1::2] = high

    # Trim to original size
    result = result[:, :original_in_features]
    return result.to(torch.int8)


def dequantize_int4(
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    """Reconstruct float weights from int4 + scales (and optional zero_points).

    Args:
        weight_int4: (out_features, in_features) int8 tensor.
        scales: (out_features, n_groups) float32.
        zero_points: (out_features, n_groups) int8 or None (symmetric).
        group_size: Number of elements per group along in_features.

    Returns:
        dequant: (out_features, in_features) float32 tensor.
    """
    out_features, in_features = weight_int4.shape

    # Pad to be divisible by group_size if needed
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        w = F.pad(weight_int4.float(), (0, pad))
    else:
        w = weight_int4.float()

    padded_in = w.shape[1]
    n_groups = padded_in // group_size

    # Reshape to (out_features, n_groups, group_size)
    w = w.reshape(out_features, n_groups, group_size)

    # Expand scales and zero_points to match
    scales_exp = scales.unsqueeze(2)  # (out_features, n_groups, 1)

    if zero_points is not None:
        zp_exp = zero_points.float().unsqueeze(2)  # (out_features, n_groups, 1)
        dequant = (w - zp_exp) * scales_exp
    else:
        dequant = w * scales_exp

    dequant = dequant.reshape(out_features, padded_in)
    # Trim back
    return dequant[:, :in_features].contiguous()


class INT4Linear(nn.Module):
    """Linear layer with INT4 weight quantization (packed two 4-bit values per byte).

    Weights are stored in packed INT4 format and dequantized on-the-fly during forward().
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[INT4Config] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config if config is not None else INT4Config()

        n_groups = (in_features + self.config.group_size - 1) // self.config.group_size
        packed_cols = in_features // 2  # will use actual in_features for unpack

        # Register buffers for quantized weight, scales, zero_points
        self.register_buffer(
            "packed_weight",
            torch.zeros(out_features, packed_cols, dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.ones(out_features, n_groups, dtype=torch.float32),
        )

        if not self.config.symmetric:
            self.register_buffer(
                "zero_points",
                torch.zeros(out_features, n_groups, dtype=torch.int8),
            )
        else:
            self.zero_points = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: Optional[INT4Config] = None) -> "INT4Linear":
        """Create an INT4Linear by quantizing an existing nn.Linear layer.

        Args:
            linear: Source nn.Linear module.
            config: INT4Config; defaults to INT4Config() if None.

        Returns:
            INT4Linear with quantized packed weights.
        """
        if config is None:
            config = INT4Config()

        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        layer = cls(in_features, out_features, bias=has_bias, config=config)

        # Quantize weight
        weight_int4, scales, zero_points = quantize_to_int4(
            linear.weight.data.float(),
            group_size=config.group_size,
            symmetric=config.symmetric,
        )

        # For asymmetric, convert to [0, 15] range before packing
        # For symmetric, shift [-8, 7] to [0, 15] for packing, store sign separately
        # We pack the raw int4 values; unpack restores them with the same bit pattern
        if config.symmetric:
            # Shift [-8, 7] -> [0, 15] for packing
            w_unsigned = (weight_int4.to(torch.int32) + 8).to(torch.int8)
        else:
            w_unsigned = weight_int4  # already [0, 15]

        packed = pack_int4(w_unsigned)

        layer.packed_weight.copy_(packed)
        layer.scales.copy_(scales)

        if not config.symmetric and zero_points is not None:
            layer.zero_points.copy_(zero_points)

        if has_bias:
            layer.bias = nn.Parameter(linear.bias.data.clone())

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weights and compute linear transformation.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        # Unpack packed_weight -> [0, 15]
        w_unpacked = unpack_int4(self.packed_weight, self.in_features)

        if self.config.symmetric:
            # Shift [0, 15] back to [-8, 7]
            w_int4 = (w_unpacked.to(torch.int32) - 8).to(torch.int8)
        else:
            w_int4 = w_unpacked

        # Dequantize
        dequant_weight = dequantize_int4(
            w_int4,
            self.scales,
            self.zero_points,
            self.config.group_size,
        ).to(self.config.compute_dtype)

        return F.linear(x, dequant_weight, self.bias)


def convert_model_to_int4(
    model: nn.Module,
    config: INT4Config,
    skip_layers: Optional[list[str]] = None,
) -> nn.Module:
    """Replace all nn.Linear layers in a model with INT4Linear.

    Args:
        model: The model to convert (modified in-place).
        config: INT4Config to use for all converted layers.
        skip_layers: List of layer name patterns to skip (e.g., ["lm_head"]).
                     A layer is skipped if its name contains any pattern in the list.

    Returns:
        The modified model.
    """
    if skip_layers is None:
        skip_layers = []

    def _should_skip(name: str) -> bool:
        return any(pattern in name for pattern in skip_layers)

    def _replace_linear(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}".lstrip(".")
            if isinstance(child, nn.Linear) and not _should_skip(full_name):
                setattr(module, child_name, INT4Linear.from_linear(child, config))
            else:
                _replace_linear(child, prefix=full_name)

    _replace_linear(model)
    return model


def estimate_int4_memory_savings(model: nn.Module) -> dict[str, float]:
    """Estimate memory savings from INT4 quantization vs float32.

    Counts float32 vs INT4 parameter bytes for all parameters in the model.
    INT4 stores 2 params per byte.

    Args:
        model: The model to analyze.

    Returns:
        dict with keys:
            "float32_mb": Total float32 size in MB.
            "int4_mb": Equivalent INT4 size in MB (2 params per byte).
            "reduction_factor": float32_mb / int4_mb.
    """
    n_params = sum(p.numel() for p in model.parameters())

    float32_bytes = n_params * 4  # 4 bytes per float32
    int4_bytes = n_params / 2     # 2 params per byte in INT4

    float32_mb = float32_bytes / 1e6
    int4_mb = int4_bytes / 1e6

    reduction_factor = float32_mb / int4_mb if int4_mb > 0 else float("inf")

    return {
        "float32_mb": float32_mb,
        "int4_mb": int4_mb,
        "reduction_factor": reduction_factor,
    }
