"""KV cache quantization: reduce memory for cached key-value tensors at inference time."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KVCacheQuantConfig:
    """Configuration for KV cache quantization."""

    bits: int = 8
    group_size: int = 64
    symmetric: bool = True
    per_channel: bool = False
    outlier_threshold: float = 6.0


# ---------------------------------------------------------------------------
# Symmetric quantization
# ---------------------------------------------------------------------------


def quantize_symmetric(
    tensor: Tensor,
    bits: int,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """Symmetric per-group quantization.

    Args:
        tensor: ``(B, n_heads, T, head_dim)``
        bits: quantization bit-width (4 or 8)
        group_size: number of elements per quantization group

    Returns:
        ``(quantized_int, scales)`` where *quantized_int* holds integer values
        and *scales* has one entry per group.
    """
    B, n_heads, T, head_dim = tensor.shape
    assert head_dim % group_size == 0, (  # noqa: S101
        f"head_dim ({head_dim}) must be divisible by group_size ({group_size})"
    )
    n_groups = head_dim // group_size
    qmax = 2 ** (bits - 1) - 1  # e.g. 127 for 8-bit

    # (B, n_heads, T, n_groups, group_size)
    grouped = tensor.reshape(B, n_heads, T, n_groups, group_size)
    absmax = grouped.abs().amax(dim=-1, keepdim=True)  # per-group max
    scale = absmax / qmax
    scale = scale.clamp(min=1e-10)  # avoid division by zero

    quantized = (
        (grouped / scale).round().clamp(-qmax, qmax).to(torch.int8 if bits == 8 else torch.int8)
    )
    # scales shape: (B, n_heads, T, n_groups, 1)
    return quantized, scale


def dequantize_symmetric(quantized: Tensor, scales: Tensor) -> Tensor:
    """Reverse symmetric quantization.

    Returns:
        Float tensor with shape ``(B, n_heads, T, head_dim)``.
    """
    dequant = quantized.float() * scales
    B, n_heads, T, n_groups, group_size = dequant.shape
    return dequant.reshape(B, n_heads, T, n_groups * group_size)


# ---------------------------------------------------------------------------
# Asymmetric quantization
# ---------------------------------------------------------------------------


def quantize_asymmetric(
    tensor: Tensor,
    bits: int,
    group_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Asymmetric per-group quantization using min/max.

    Returns:
        ``(quantized_int, scales, zero_points)``
    """
    B, n_heads, T, head_dim = tensor.shape
    assert head_dim % group_size == 0  # noqa: S101
    n_groups = head_dim // group_size
    qmax = 2**bits - 1  # e.g. 255 for 8-bit

    grouped = tensor.reshape(B, n_heads, T, n_groups, group_size)
    gmin = grouped.amin(dim=-1, keepdim=True)
    gmax = grouped.amax(dim=-1, keepdim=True)

    scale = (gmax - gmin) / qmax
    scale = scale.clamp(min=1e-10)
    zero_point = (-gmin / scale).round()

    quantized = ((grouped - gmin) / scale).round().clamp(0, qmax).to(torch.int16)
    return quantized, scale, zero_point


def dequantize_asymmetric(
    quantized: Tensor,
    scales: Tensor,
    zero_points: Tensor,
) -> Tensor:
    """Reverse asymmetric quantization.

    Returns:
        Float tensor with shape ``(B, n_heads, T, head_dim)``.
    """
    dequant = (quantized.float() - zero_points) * scales
    B, n_heads, T, n_groups, group_size = dequant.shape
    return dequant.reshape(B, n_heads, T, n_groups * group_size)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def detect_outliers(tensor: Tensor, threshold: float) -> Tensor:
    """Return boolean mask where ``|tensor| > threshold * tensor.std()``.

    The returned mask has the same shape as *tensor*.
    """
    std = tensor.std()
    return tensor.abs() > threshold * std


# ---------------------------------------------------------------------------
# QuantizedKVCache
# ---------------------------------------------------------------------------


class QuantizedKVCache:
    """Drop-in KV cache that stores keys and values in quantized form."""

    def __init__(self, config: KVCacheQuantConfig) -> None:
        self.config = config

        # Quantized storage lists (concatenated on get)
        self._key_quant: list[Tensor] = []
        self._key_scales: list[Tensor] = []
        self._key_zp: list[Tensor] = []

        self._val_quant: list[Tensor] = []
        self._val_scales: list[Tensor] = []
        self._val_zp: list[Tensor] = []

        self._length: int = 0
        self._fp16_bytes: float = 0.0
        self._quantized_bytes: float = 0.0

    # ----- public API -----

    def update(self, keys: Tensor, values: Tensor) -> None:
        """Quantize and append new key/value tensors to the cache.

        Args:
            keys: ``(B, n_heads, T_new, head_dim)``
            values: ``(B, n_heads, T_new, head_dim)``
        """
        T_new = keys.shape[2]
        fp16_size = keys.nelement() * 2 + values.nelement() * 2  # 2 bytes per fp16
        self._fp16_bytes += fp16_size

        if self.config.symmetric:
            kq, ks = quantize_symmetric(keys, self.config.bits, self.config.group_size)
            vq, vs = quantize_symmetric(values, self.config.bits, self.config.group_size)
            self._key_quant.append(kq)
            self._key_scales.append(ks)
            self._val_quant.append(vq)
            self._val_scales.append(vs)
            # quantized bytes: int tensor + scales
            q_bytes = kq.nelement() * kq.element_size() + ks.nelement() * ks.element_size()
            q_bytes += vq.nelement() * vq.element_size() + vs.nelement() * vs.element_size()
        else:
            kq, ks, kzp = quantize_asymmetric(keys, self.config.bits, self.config.group_size)
            vq, vs, vzp = quantize_asymmetric(values, self.config.bits, self.config.group_size)
            self._key_quant.append(kq)
            self._key_scales.append(ks)
            self._key_zp.append(kzp)
            self._val_quant.append(vq)
            self._val_scales.append(vs)
            self._val_zp.append(vzp)
            q_bytes = (
                kq.nelement() * kq.element_size()
                + ks.nelement() * ks.element_size()
                + kzp.nelement() * kzp.element_size()
                + vq.nelement() * vq.element_size()
                + vs.nelement() * vs.element_size()
                + vzp.nelement() * vzp.element_size()
            )

        self._quantized_bytes += q_bytes
        self._length += T_new

    def get(self) -> tuple[Tensor, Tensor]:
        """Dequantize and return full K, V as float tensors."""
        if self._length == 0:
            raise RuntimeError("Cache is empty")

        if self.config.symmetric:
            keys = torch.cat(
                [dequantize_symmetric(q, s) for q, s in zip(self._key_quant, self._key_scales)],
                dim=2,
            )
            values = torch.cat(
                [dequantize_symmetric(q, s) for q, s in zip(self._val_quant, self._val_scales)],
                dim=2,
            )
        else:
            keys = torch.cat(
                [
                    dequantize_asymmetric(q, s, z)
                    for q, s, z in zip(self._key_quant, self._key_scales, self._key_zp)
                ],
                dim=2,
            )
            values = torch.cat(
                [
                    dequantize_asymmetric(q, s, z)
                    for q, s, z in zip(self._val_quant, self._val_scales, self._val_zp)
                ],
                dim=2,
            )
        return keys, values

    def length(self) -> int:
        """Number of cached tokens."""
        return self._length

    def clear(self) -> None:
        """Reset the cache."""
        self._key_quant.clear()
        self._key_scales.clear()
        self._key_zp.clear()
        self._val_quant.clear()
        self._val_scales.clear()
        self._val_zp.clear()
        self._length = 0
        self._fp16_bytes = 0.0
        self._quantized_bytes = 0.0

    def memory_savings(self) -> dict[str, float]:
        """Return memory usage statistics."""
        ratio = self._fp16_bytes / self._quantized_bytes if self._quantized_bytes > 0 else 0.0
        return {
            "fp16_bytes": self._fp16_bytes,
            "quantized_bytes": self._quantized_bytes,
            "compression_ratio": ratio,
        }


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def compute_quantization_error(
    original: Tensor,
    reconstructed: Tensor,
) -> dict[str, float]:
    """Compute quantization error metrics.

    Returns:
        ``{"mse": float, "max_error": float, "snr_db": float}``
    """
    diff = original.float() - reconstructed.float()
    mse = (diff**2).mean().item()
    max_error = diff.abs().max().item()

    signal_power = (original.float() ** 2).mean().item()
    noise_power = mse
    if noise_power > 0:
        snr_db = 10.0 * math.log10(signal_power / noise_power)
    else:
        snr_db = float("inf")

    return {"mse": mse, "max_error": max_error, "snr_db": snr_db}
