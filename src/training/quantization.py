"""Post-training quantization: INT8 and INT4 weight quantization."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class QuantizationConfig:
    bits: int = 8                    # 4 or 8
    symmetric: bool = True           # symmetric (zero_point=0) or asymmetric
    per_channel: bool = True         # per-channel or per-tensor quantization
    calibration_batches: int = 10


def quantize_tensor(
    w: Tensor,
    bits: int,
    symmetric: bool = True,
    per_channel: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize weight tensor.

    Per-channel: scale/zero computed per output channel (dim 0).
    Symmetric: zero_point = 0, scale = max(|w|) / (2^(bits-1) - 1)
    Asymmetric: scale = (max-min) / (2^bits - 1), zero_point = round(-min/scale)

    Returns (quantized_w, scale, zero_point)
    - quantized_w: same shape as w, float32 but rounded to grid
    - scale: (out_features,) or scalar
    - zero_point: (out_features,) or scalar
    """
    if per_channel:
        # Operate row-wise (dim 0 = out_features)
        # Flatten all dims except the first for min/max computation
        w_flat = w.view(w.shape[0], -1)  # (out_features, *)

        if symmetric:
            q_max = 2 ** (bits - 1) - 1
            abs_max = w_flat.abs().max(dim=1).values  # (out_features,)
            scale = abs_max / q_max
            # Avoid division by zero
            scale = scale.clamp(min=1e-8)
            zero_point = torch.zeros_like(scale)

            # Quantize: clamp to [-q_max, q_max], round
            scale_bc = scale.view(-1, *([1] * (w.dim() - 1)))
            q_w = (w / scale_bc).clamp(-q_max, q_max).round()
        else:
            q_max = 2 ** bits - 1
            w_min = w_flat.min(dim=1).values   # (out_features,)
            w_max = w_flat.max(dim=1).values   # (out_features,)
            scale = (w_max - w_min) / q_max
            scale = scale.clamp(min=1e-8)
            zero_point = (-w_min / scale).round()

            scale_bc = scale.view(-1, *([1] * (w.dim() - 1)))
            zp_bc = zero_point.view(-1, *([1] * (w.dim() - 1)))
            q_w = (w / scale_bc + zp_bc).clamp(0, q_max).round()
    else:
        # Per-tensor
        if symmetric:
            q_max = 2 ** (bits - 1) - 1
            abs_max = w.abs().max()
            scale = abs_max / q_max
            scale = scale.clamp(min=1e-8)
            zero_point = torch.zeros_like(scale)
            q_w = (w / scale).clamp(-q_max, q_max).round()
        else:
            q_max = 2 ** bits - 1
            w_min = w.min()
            w_max = w.max()
            scale = (w_max - w_min) / q_max
            scale = scale.clamp(min=1e-8)
            zero_point = (-w_min / scale).round()
            q_w = (w / scale + zero_point).clamp(0, q_max).round()

    return q_w, scale, zero_point


def dequantize_tensor(q_w: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
    """Reconstruct float tensor from quantized representation.
    w_float = (q_w - zero_point) * scale
    """
    if scale.dim() > 0 and scale.shape[0] > 1:
        # Per-channel: broadcast along dim 0
        scale_bc = scale.view(-1, *([1] * (q_w.dim() - 1)))
        zp_bc = zero_point.view(-1, *([1] * (q_w.dim() - 1)))
        return (q_w - zp_bc) * scale_bc
    else:
        return (q_w - zero_point) * scale


def quantization_error(original: Tensor, quantized: Tensor) -> float:
    """Compute relative L2 error: ||original - quantized||_F / ||original||_F."""
    denom = original.norm()
    if denom == 0:
        return 0.0
    return ((original - quantized).norm() / denom).item()


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with quantized weights."""

    def __init__(self, linear: nn.Linear, config: QuantizationConfig) -> None:
        """Quantize linear.weight using config settings."""
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = config.bits

        # Store bias as a parameter if present
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

        # Quantize and store as buffers (not trainable Parameters)
        with torch.no_grad():
            q_w, scale, zero_point = quantize_tensor(
                linear.weight.data,
                bits=config.bits,
                symmetric=config.symmetric,
                per_channel=config.per_channel,
            )

        self.register_buffer("q_weight", q_w)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

        # Store config flags for dequant
        self.symmetric = config.symmetric
        self.per_channel = config.per_channel

    def forward(self, x: Tensor) -> Tensor:
        """Dequantize weight at forward time and compute linear."""
        w = dequantize_tensor(self.q_weight, self.scale, self.zero_point)
        return F.linear(x, w, self.bias)

    @property
    def compression_ratio(self) -> float:
        """Ratio of original bits (32) to quantized bits."""
        return 32.0 / self.bits


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    target_modules: list[str] | None = None,
) -> tuple[nn.Module, dict]:
    """Replace all (or target) nn.Linear layers with QuantizedLinear.

    target_modules: list of module name substrings to match (e.g. ["gate_proj", "up_proj"])
    Returns (quantized_model, stats) where stats has:
        'n_quantized': int
        'total_params_saved': int (theoretical bits saved)
        'mean_error': float
    """
    n_quantized = 0
    total_params_saved = 0
    errors: list[float] = []

    # Collect replacements first to avoid mutating during iteration
    replacements: list[tuple[nn.Module, str, QuantizedLinear]] = []

    for name, module in model.named_modules():
        # Find the parent and attribute name
        parts = name.split(".")
        if not isinstance(module, nn.Linear):
            continue

        # Filter by target_modules if specified
        if target_modules is not None:
            if not any(sub in name for sub in target_modules):
                continue

        # Compute quantization error before replacement
        with torch.no_grad():
            q_w, scale, zero_point = quantize_tensor(
                module.weight.data,
                bits=config.bits,
                symmetric=config.symmetric,
                per_channel=config.per_channel,
            )
            deq_w = dequantize_tensor(q_w, scale, zero_point)
            err = quantization_error(module.weight.data, deq_w)
            errors.append(err)

        # Compute bits saved: original is float32 (32 bits), quantized is config.bits
        n_params = module.weight.numel()
        bits_saved = n_params * (32 - config.bits)
        total_params_saved += bits_saved

        # Build QuantizedLinear
        ql = QuantizedLinear(module, config)

        # Navigate to parent module
        if len(parts) == 1:
            parent = model
            attr = parts[0]
        else:
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]

        replacements.append((parent, attr, ql))
        n_quantized += 1

    # Apply replacements
    for parent, attr, ql in replacements:
        setattr(parent, attr, ql)

    mean_error = float(sum(errors) / len(errors)) if errors else 0.0

    stats = {
        "n_quantized": n_quantized,
        "total_params_saved": total_params_saved,
        "mean_error": mean_error,
    }

    return model, stats


class CalibrationDataset:
    """Manages calibration data for quantization."""

    def __init__(self, sequences: list[Tensor]) -> None:
        self._sequences = sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def get_batch(self, batch_size: int) -> Tensor:
        """Return random batch of sequences, padded to same length."""
        chosen = random.choices(self._sequences, k=batch_size)
        max_len = max(seq.shape[-1] for seq in chosen)
        # Pad each sequence to max_len (pad with 0 on the right)
        padded = []
        for seq in chosen:
            seq_len = seq.shape[-1]
            if seq_len < max_len:
                pad = torch.zeros(max_len - seq_len, dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, pad], dim=-1)
            padded.append(seq)
        return torch.stack(padded, dim=0)  # (batch_size, T)
