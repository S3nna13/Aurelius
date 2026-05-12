"""KIVI — 2-bit asymmetric KV cache quantization.

KIVI (Liu et al., 2024) quantizes keys per-channel and values per-token
using asymmetric min-max scaling.  A small residual window at the end of
the sequence is kept in full precision to protect the most recent tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor


class KIVIQuantizer:
    """Asymmetric KV cache quantizer with per-channel (K) and per-token (V) schemes."""

    def __init__(self, bits: int = 2, residual_length: int = 128, group_size: int = 32) -> None:
        """Initialize the KIVI quantizer.

        Args:
            bits: Bit-width for quantization (e.g. 2 gives levels ``{0,1,2,3}``).
            residual_length: Number of trailing tokens kept in full precision.
            group_size: Unused in this implementation (reserved for future group-wise KIVI).
        """
        self.bits = bits
        self.residual_length = residual_length
        self.group_size = group_size
        self.qmax = 2**bits - 1  # e.g. 3 for 2-bit

    # ------------------------------------------------------------------
    # Per-scheme quantization
    # ------------------------------------------------------------------

    def quantize_k(self, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-channel (dim=-1) asymmetric min-max quantization for keys.

        Args:
            k: Key tensor of shape ``(..., seq_len, head_dim)``.

        Returns:
            ``(quantized_k, scale, zero_point)`` where *quantized_k* is an
            integer tensor in ``[0, 2^bits - 1]``, and *scale* / *zero_point*
            have the same broadcastable shape as the min/max statistics.
        """
        kmin = k.amin(dim=-1, keepdim=True)
        kmax = k.amax(dim=-1, keepdim=True)
        scale = (kmax - kmin) / self.qmax
        scale = scale.clamp(min=1e-10)
        zero_point = kmin
        quantized = ((k - kmin) / scale).round().clamp(0, self.qmax).to(torch.int8)
        return quantized, scale, zero_point

    def quantize_v(self, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-token (dim=-2) asymmetric min-max quantization for values.

        Args:
            v: Value tensor of shape ``(..., seq_len, head_dim)``.

        Returns:
            ``(quantized_v, scale, zero_point)``.
        """
        vmin = v.amin(dim=-2, keepdim=True)
        vmax = v.amax(dim=-2, keepdim=True)
        scale = (vmax - vmin) / self.qmax
        scale = scale.clamp(min=1e-10)
        zero_point = vmin
        quantized = ((v - vmin) / scale).round().clamp(0, self.qmax).to(torch.int8)
        return quantized, scale, zero_point

    def dequantize(
        self,
        quantized: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        per: str,
    ) -> Tensor:
        """Reconstruct a float tensor from asymmetric integer codes.

        Args:
            quantized: Integer tensor from :meth:`quantize_k` or :meth:`quantize_v`.
            scale: Scale tensor produced during quantization.
            zero_point: Zero-point (min) tensor produced during quantization.
            per: Either ``"channel"`` (for keys) or ``"token"`` (for values).
                Currently informational only; the math is identical.

        Returns:
            Dequantized float tensor with the same shape as the original input.
        """
        _ = per  # noqa: F841  # reserved for future per-scheme validation
        return quantized.float() * scale + zero_point

    # ------------------------------------------------------------------
    # KV cache compression / decompression with residual window
    # ------------------------------------------------------------------

    def compress_kv_cache(self, k_cache: Tensor, v_cache: Tensor) -> dict[str, Tensor]:
        """Quantize all but the last ``residual_length`` tokens of the KV cache.

        Args:
            k_cache: Key cache of shape ``(..., seq_len, head_dim)``.
            v_cache: Value cache of shape ``(..., seq_len, head_dim)``.

        Returns:
            A dictionary containing:
                - ``k_q``, ``k_scale``, ``k_zp`` — quantized key history
                - ``v_q``, ``v_scale``, ``v_zp`` — quantized value history
                - ``k_residual``, ``v_residual`` — full-precision residual window
            If ``seq_len <= residual_length``, no quantization is performed and
            the residual tensors are the full caches.
        """
        seq_len = k_cache.shape[-2]
        if seq_len <= self.residual_length:
            return {
                "k_q": torch.empty(0, dtype=torch.int8, device=k_cache.device),
                "k_scale": torch.empty(0, device=k_cache.device),
                "k_zp": torch.empty(0, device=k_cache.device),
                "v_q": torch.empty(0, dtype=torch.int8, device=v_cache.device),
                "v_scale": torch.empty(0, device=v_cache.device),
                "v_zp": torch.empty(0, device=v_cache.device),
                "k_residual": k_cache,
                "v_residual": v_cache,
            }

        history_k = k_cache[..., : -self.residual_length, :]
        residual_k = k_cache[..., -self.residual_length :, :]
        history_v = v_cache[..., : -self.residual_length, :]
        residual_v = v_cache[..., -self.residual_length :, :]

        k_q, k_scale, k_zp = self.quantize_k(history_k)
        v_q, v_scale, v_zp = self.quantize_v(history_v)

        return {
            "k_q": k_q,
            "k_scale": k_scale,
            "k_zp": k_zp,
            "v_q": v_q,
            "v_scale": v_scale,
            "v_zp": v_zp,
            "k_residual": residual_k,
            "v_residual": residual_v,
        }

    def decompress_kv_cache(self, compressed: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Dequantize history and concatenate with the residual window.

        Args:
            compressed: Dictionary produced by :meth:`compress_kv_cache`.

        Returns:
            ``(k_cache, v_cache)`` reconstructed float tensors.
        """
        k_residual = compressed["k_residual"]
        v_residual = compressed["v_residual"]
        k_q = compressed["k_q"]
        v_q = compressed["v_q"]

        # If nothing was quantized (empty history), return residuals directly
        if k_q.numel() == 0:
            return k_residual, v_residual

        k_hist = self.dequantize(k_q, compressed["k_scale"], compressed["k_zp"], per="channel")
        v_hist = self.dequantize(v_q, compressed["v_scale"], compressed["v_zp"], per="token")

        k_cache = torch.cat([k_hist, k_residual], dim=-2)
        v_cache = torch.cat([v_hist, v_residual], dim=-2)
        return k_cache, v_cache
