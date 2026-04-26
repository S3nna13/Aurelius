"""Compressive Transformer memory (Rae et al. 2019, arXiv:1911.05507).

Provides a two-level memory structure:
  * recent memory -- a sliding window of the most recent K/V tokens
  * compressed memory -- older tokens evicted from the recent window and
    downsampled via a compression function (mean pool, max pool, or a
    learned 1D convolution).

This module exposes only the memory structure and compression helpers.
Attention integration is the caller's responsibility -- use
``concatenated_kv()`` to obtain the ``[compressed | recent]`` K/V to feed
into an attention layer.

Shapes
------
All K/V tensors are ``(batch, n_heads, seq, head_dim)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CompressiveMemoryState:
    """Snapshot of the two-level memory."""

    recent_k: torch.Tensor | None
    recent_v: torch.Tensor | None
    compressed_k: torch.Tensor | None
    compressed_v: torch.Tensor | None


_VALID_FNS = ("mean_pool", "max_pool", "conv1d")


class CompressiveMemory(nn.Module):
    """Two-level K/V memory with learnable-or-fixed compression of evictions.

    Parameters
    ----------
    n_heads, head_dim
        Per-head attention geometry.
    recent_size
        Maximum number of recent tokens retained per batch/head.
    compressed_size
        Maximum number of compressed tokens retained per batch/head.
    compression_rate
        Downsampling factor applied to evicted tokens. Must divide the
        number of tokens evicted in a single update call.
    compression_fn
        One of ``"mean_pool"``, ``"max_pool"``, ``"conv1d"``.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        recent_size: int = 512,
        compressed_size: int = 512,
        compression_rate: int = 4,
        compression_fn: str = "mean_pool",
    ) -> None:
        super().__init__()
        if compression_fn not in _VALID_FNS:
            raise ValueError(f"compression_fn must be one of {_VALID_FNS}, got {compression_fn!r}")
        if n_heads <= 0 or head_dim <= 0:
            raise ValueError("n_heads and head_dim must be positive")
        if recent_size <= 0 or compressed_size <= 0:
            raise ValueError("recent_size and compressed_size must be positive")
        if compression_rate <= 0:
            raise ValueError("compression_rate must be positive")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.recent_size = recent_size
        self.compressed_size = compressed_size
        self.compression_rate = compression_rate
        self.compression_fn = compression_fn

        if compression_fn == "conv1d":
            # One conv per stream (K, V). Group per-head so heads don't mix.
            # Input channels: n_heads * head_dim treated as groups=n_heads.
            ch = n_heads * head_dim
            self.k_conv = nn.Conv1d(
                in_channels=ch,
                out_channels=ch,
                kernel_size=compression_rate,
                stride=compression_rate,
                groups=n_heads,
                bias=True,
            )
            self.v_conv = nn.Conv1d(
                in_channels=ch,
                out_channels=ch,
                kernel_size=compression_rate,
                stride=compression_rate,
                groups=n_heads,
                bias=True,
            )
        else:
            self.k_conv = None
            self.v_conv = None

        self._batch_size: int | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        # Buffers (None until reset).
        self._recent_k: torch.Tensor | None = None
        self._recent_v: torch.Tensor | None = None
        self._compressed_k: torch.Tensor | None = None
        self._compressed_v: torch.Tensor | None = None

    # ------------------------------------------------------------------ api

    def reset(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Clear all memory and fix the expected batch size."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size
        self._device = device
        self._dtype = dtype
        self._recent_k = None
        self._recent_v = None
        self._compressed_k = None
        self._compressed_v = None

    def state(self) -> CompressiveMemoryState:
        return CompressiveMemoryState(
            recent_k=self._recent_k,
            recent_v=self._recent_v,
            compressed_k=self._compressed_k,
            compressed_v=self._compressed_v,
        )

    def concatenated_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(K, V)`` with ``[compressed | recent]`` layout.

        If both levels are empty, returns zero-length tensors with the
        correct leading dimensions.
        """
        parts_k = []
        parts_v = []
        if self._compressed_k is not None:
            parts_k.append(self._compressed_k)
            parts_v.append(self._compressed_v)
        if self._recent_k is not None:
            parts_k.append(self._recent_k)
            parts_v.append(self._recent_v)

        if not parts_k:
            if self._batch_size is None:
                raise RuntimeError("CompressiveMemory.reset() has not been called")
            shape = (self._batch_size, self.n_heads, 0, self.head_dim)
            empty = torch.zeros(shape, device=self._device, dtype=self._dtype)
            return empty, empty.clone()
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> CompressiveMemoryState:
        """Append ``new_k`` / ``new_v`` to recent memory, evicting/compressing overflow.

        Both tensors must be shape ``(batch, n_heads, seq_new, head_dim)``.
        """
        if self._batch_size is None:
            raise RuntimeError("CompressiveMemory.reset() has not been called")
        if new_k.shape != new_v.shape:
            raise ValueError("new_k and new_v must have identical shapes")
        if new_k.dim() != 4:
            raise ValueError("expected 4D tensors (B, H, S, D)")
        B, H, S, D = new_k.shape
        if B != self._batch_size:
            raise ValueError(f"batch size {B} does not match reset() batch size {self._batch_size}")
        if H != self.n_heads or D != self.head_dim:
            raise ValueError(
                f"expected n_heads={self.n_heads}, head_dim={self.head_dim}; "
                f"got n_heads={H}, head_dim={D}"
            )

        # Append to recent.
        if self._recent_k is None:
            recent_k = new_k
            recent_v = new_v
        else:
            recent_k = torch.cat([self._recent_k, new_k], dim=2)
            recent_v = torch.cat([self._recent_v, new_v], dim=2)

        # Evict oldest tokens from recent when we exceed recent_size.
        overflow = recent_k.shape[2] - self.recent_size
        if overflow > 0:
            if overflow % self.compression_rate != 0:
                raise ValueError(
                    f"eviction batch size {overflow} is not divisible by "
                    f"compression_rate {self.compression_rate}"
                )
            evict_k = recent_k[:, :, :overflow, :]
            evict_v = recent_v[:, :, :overflow, :]
            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]

            comp_k = self._compress(evict_k, which="k")
            comp_v = self._compress(evict_v, which="v")

            if self._compressed_k is None:
                self._compressed_k = comp_k
                self._compressed_v = comp_v
            else:
                self._compressed_k = torch.cat([self._compressed_k, comp_k], dim=2)
                self._compressed_v = torch.cat([self._compressed_v, comp_v], dim=2)

            # Trim oldest compressed tokens if we exceed compressed_size.
            excess = self._compressed_k.shape[2] - self.compressed_size
            if excess > 0:
                self._compressed_k = self._compressed_k[:, :, excess:, :]
                self._compressed_v = self._compressed_v[:, :, excess:, :]

        self._recent_k = recent_k
        self._recent_v = recent_v
        return self.state()

    # -------------------------------------------------------------- compress

    def _compress(self, x: torch.Tensor, which: str) -> torch.Tensor:
        """Downsample along the sequence dim by ``compression_rate``.

        Input/output shape: ``(B, H, S, D)`` -> ``(B, H, S // rate, D)``.
        """
        B, H, S, D = x.shape
        rate = self.compression_rate
        assert S % rate == 0, "caller must ensure divisibility"  # noqa: S101

        if self.compression_fn == "mean_pool":
            return x.reshape(B, H, S // rate, rate, D).mean(dim=3)
        if self.compression_fn == "max_pool":
            return x.reshape(B, H, S // rate, rate, D).amax(dim=3)
        if self.compression_fn == "conv1d":
            conv = self.k_conv if which == "k" else self.v_conv
            # (B, H, S, D) -> (B, H*D, S) with head-grouped channels.
            y = x.permute(0, 1, 3, 2).reshape(B, H * D, S)
            y = conv(y)  # (B, H*D, S//rate)
            S_out = y.shape[-1]
            y = y.reshape(B, H, D, S_out).permute(0, 1, 3, 2).contiguous()
            return y
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["CompressiveMemory", "CompressiveMemoryState"]
