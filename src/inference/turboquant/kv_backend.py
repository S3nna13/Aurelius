# src/inference/turboquant/kv_backend.py
"""Compressed KV cache backend using TurboQuant.

Drop-in conceptual replacement for the plain (k, v) tuple cache.
Stores compressed keys and values per layer.
"""

from __future__ import annotations

import torch

from .compressor import CompressedKV, TurboQuantCompressor


class CompressedKVCache:
    """KV cache that stores compressed keys and values.

    Each layer gets its own compressor (so rotation matrices are per-layer).
    Keys and values are compressed independently.

    Args:
        n_layers: Number of transformer layers.
        head_dim: Feature dimension per head.
        n_kv_heads: Number of KV heads.
        n_codes: PolarQuant codebook size.
        sketch_dim: QJL sketch dimension.
    """

    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        n_kv_heads: int,
        n_codes: int = 256,
        sketch_dim: int = 64,
    ) -> None:
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        # One compressor per layer, per key/value
        self._key_compressors = [
            TurboQuantCompressor(dim=head_dim, n_codes=n_codes, sketch_dim=sketch_dim, seed=i * 2)
            for i in range(n_layers)
        ]
        self._val_compressors = [
            TurboQuantCompressor(
                dim=head_dim, n_codes=n_codes, sketch_dim=sketch_dim, seed=i * 2 + 1
            )
            for i in range(n_layers)
        ]

        # Stored compressed tensors per layer: list of (CompressedKV | None, CompressedKV | None)
        self._cache: list[tuple[CompressedKV | None, CompressedKV | None]] = [
            (None, None) for _ in range(n_layers)
        ]

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Compress and store new key/value tensors for a layer.

        Args:
            layer_idx: Which layer's cache to update.
            k: Key tensor, shape (B, S, n_kv_heads, head_dim).
            v: Value tensor, shape (B, S, n_kv_heads, head_dim).
        """
        ck = self._key_compressors[layer_idx].compress(k)
        cv = self._val_compressors[layer_idx].compress(v)
        self._cache[layer_idx] = (ck, cv)

    def get_decompressed(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Decompress and return (k, v) for a layer, or None if empty.

        Returns approximate reconstruction (lossy).
        """
        ck, cv = self._cache[layer_idx]
        if ck is None or cv is None:
            return None
        k_hat = self._key_compressors[layer_idx].decompress(ck)
        v_hat = self._val_compressors[layer_idx].decompress(cv)
        return k_hat, v_hat

    def is_empty(self, layer_idx: int) -> bool:
        """Check if a layer's cache is empty."""
        return self._cache[layer_idx][0] is None

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache = [(None, None) for _ in range(self.n_layers)]
