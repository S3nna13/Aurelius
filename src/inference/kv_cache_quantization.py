"""KV-cache quantization helpers for memory-efficient decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedKVCache:
    key_q: torch.Tensor
    value_q: torch.Tensor
    key_scale: torch.Tensor
    value_scale: torch.Tensor


def _quantize_per_token(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize per batch/token slice to int8 with symmetric scaling."""
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor.shape)}")
    abs_max = tensor.abs().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(abs_max > 0, abs_max / 127.0, torch.ones_like(abs_max))
    q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1).squeeze(-1)


def _dequantize_per_token(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 per-token tensors back to float."""
    if q.dim() != 4:
        raise ValueError(f"Expected 4D quantized tensor, got shape {tuple(q.shape)}")
    return q.to(torch.float32) * scale.unsqueeze(-1).unsqueeze(-1)


def quantize_kv_cache(key_cache: torch.Tensor, value_cache: torch.Tensor) -> QuantizedKVCache:
    """Quantize key/value caches with per-token scales."""
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must match")
    key_q, key_scale = _quantize_per_token(key_cache)
    value_q, value_scale = _quantize_per_token(value_cache)
    return QuantizedKVCache(
        key_q=key_q,
        value_q=value_q,
        key_scale=key_scale,
        value_scale=value_scale,
    )


def dequantize_kv_cache(cache: QuantizedKVCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a quantized KV cache pair."""
    return (
        _dequantize_per_token(cache.key_q, cache.key_scale),
        _dequantize_per_token(cache.value_q, cache.value_scale),
    )


def kv_cache_quantization_error(key_cache: torch.Tensor, value_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean absolute quantization errors for key and value caches."""
    quantized = quantize_kv_cache(key_cache, value_cache)
    key_hat, value_hat = dequantize_kv_cache(quantized)
    return (
        (key_hat - key_cache).abs().mean(),
        (value_hat - value_cache).abs().mean(),
    )
