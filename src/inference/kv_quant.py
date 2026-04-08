"""KV cache quantization for memory-efficient long-context inference.

The KV cache grows linearly with sequence length and becomes the dominant
memory consumer during long-context generation.  Quantizing keys and values
to INT8 or INT4 reduces cache size 2–4× with minimal quality degradation.

Design:
    * **KVQuantizer** handles per-group symmetric and asymmetric quantization.
    * Groups of ``group_size`` elements share a single scale (and zero-point for
      asymmetric).  Smaller groups → better accuracy; larger groups → less
      overhead.
    * Keys and values are quantized independently because their distributions
      often differ.

References:
    - GPTQ (Frantar et al., 2022): group-wise weight quantization.
    - Keyformer (Adnan et al., 2024): KV cache reduction for LLM inference.
    - KVQuant (Hooper et al., 2024): nuanced KV cache quantization.
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# KVQuantizer
# ---------------------------------------------------------------------------

class KVQuantizer:
    """Quantize KV cache tensors for memory-efficient long-context inference.

    Supports per-group symmetric (INT8 / INT4) and asymmetric (UINT8)
    quantization.  Keys and values often have different optimal quantization
    parameters and are always quantized independently.

    Args:
        bits: Quantization bit-width — ``4`` or ``8``.
        group_size: Number of elements per quantization group along the last
            dimension (default 128).  Must evenly divide ``head_dim`` or the
            last dimension will be zero-padded to the next multiple.
        symmetric: If ``True``, use symmetric quantization
            ``[-2^(bits-1), 2^(bits-1)-1]`` and store as ``torch.int8``.
            If ``False``, use asymmetric zero-point quantization
            ``[0, 2^bits - 1]`` and store as ``torch.uint8``.
    """

    def __init__(
        self,
        bits: int = 8,
        group_size: int = 128,
        symmetric: bool = True,
    ) -> None:
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric

        if symmetric:
            self.q_min = -(2 ** (bits - 1))
            self.q_max = 2 ** (bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** bits - 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_to_group(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Zero-pad last dimension so it is divisible by group_size.

        Returns:
            ``(padded, original_last_dim)``
        """
        orig = x.shape[-1]
        remainder = orig % self.group_size
        if remainder == 0:
            return x, orig
        pad = self.group_size - remainder
        x = torch.nn.functional.pad(x, (0, pad))
        return x, orig

    # ------------------------------------------------------------------
    # Core quantize / dequantize
    # ------------------------------------------------------------------

    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Quantize a float tensor to ``bits``-bit integers.

        The last dimension is split into groups of ``group_size``; each group
        gets its own scale (and zero-point for asymmetric).

        Args:
            tensor: Arbitrary-shape float tensor ``(..., d)``.

        Returns:
            ``(quantized, scales, zero_points)`` where

            * ``quantized``: same leading shape, last dim padded to next
              multiple of ``group_size``; dtype ``torch.int8`` (symmetric) or
              ``torch.uint8`` (asymmetric).
            * ``scales``: ``(*leading_shape, n_groups)`` float32 per-group
              scales.
            * ``zero_points``: ``(*leading_shape, n_groups)`` float32 for
              asymmetric, or ``None`` for symmetric.
        """
        x = tensor.float()
        leading_shape = x.shape[:-1]
        x_pad, orig_last = self._pad_to_group(x)
        d_pad = x_pad.shape[-1]
        n_groups = d_pad // self.group_size

        # Reshape: (..., n_groups, group_size)
        xg = x_pad.reshape(*leading_shape, n_groups, self.group_size)

        if self.symmetric:
            # scale = max(|group|) / q_max
            scales = xg.abs().amax(dim=-1).clamp(min=1e-8) / self.q_max  # (..., n_groups)
            # Quantize
            xq = (xg / scales.unsqueeze(-1)).round().clamp(self.q_min, self.q_max)
            quantized = xq.reshape(*leading_shape, d_pad).to(torch.int8)
            return quantized, scales.float(), None
        else:
            x_min = xg.amin(dim=-1)  # (..., n_groups)
            x_max = xg.amax(dim=-1)
            scales = (x_max - x_min).clamp(min=1e-8) / self.q_max  # (..., n_groups)
            zero_points = (-x_min / scales).round().clamp(0, self.q_max)
            xq = ((xg / scales.unsqueeze(-1)).round() + zero_points.unsqueeze(-1)).clamp(
                self.q_min, self.q_max
            )
            quantized = xq.reshape(*leading_shape, d_pad).to(torch.uint8)
            return quantized, scales.float(), zero_points.float()

    def dequantize(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None = None,
        original_shape: tuple | None = None,
    ) -> torch.Tensor:
        """Reconstruct a float32 tensor from a quantized representation.

        Args:
            quantized: Integer tensor ``(..., d_pad)`` as returned by
                :meth:`quantize`.
            scales: ``(*leading_shape, n_groups)`` float32 per-group scales.
            zero_points: ``(*leading_shape, n_groups)`` for asymmetric, or
                ``None`` for symmetric.
            original_shape: If provided, trim the output to this shape (removes
                zero-padding added during quantization).

        Returns:
            Float32 tensor of shape ``original_shape`` (or ``(..., d_pad)`` if
            ``original_shape`` is ``None``).
        """
        leading_shape = quantized.shape[:-1]
        d_pad = quantized.shape[-1]
        n_groups = scales.shape[-1]

        xq = quantized.float().reshape(*leading_shape, n_groups, self.group_size)

        if zero_points is None:
            # Symmetric
            x = xq * scales.unsqueeze(-1)
        else:
            x = (xq - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)

        x = x.reshape(*leading_shape, d_pad)

        if original_shape is not None:
            # Trim padding: only last dim may have changed
            x = x[..., : original_shape[-1]]
            # Restore full shape in case leading dims changed too
            x = x.reshape(original_shape)

        return x

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def quantize_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> dict:
        """Quantize both K and V cache tensors.

        Args:
            k: ``(B, seq_len, n_kv_heads, head_dim)`` float keys.
            v: ``(B, seq_len, n_kv_heads, head_dim)`` float values.

        Returns:
            Dictionary with keys::

                {
                    'k_q':          quantized keys,
                    'k_scales':     per-group scales for K,
                    'k_zp':         zero-points for K (or None),
                    'v_q':          quantized values,
                    'v_scales':     per-group scales for V,
                    'v_zp':         zero-points for V (or None),
                    'original_shape': tuple (shape of k and v — they must match),
                }
        """
        assert k.shape == v.shape, "K and V must have the same shape"
        original_shape = tuple(k.shape)

        k_q, k_scales, k_zp = self.quantize(k)
        v_q, v_scales, v_zp = self.quantize(v)

        return {
            "k_q": k_q,
            "k_scales": k_scales,
            "k_zp": k_zp,
            "v_q": v_q,
            "v_scales": v_scales,
            "v_zp": v_zp,
            "original_shape": original_shape,
        }

    def dequantize_kv_cache(
        self,
        cache_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct K and V from a quantized cache dict.

        Args:
            cache_dict: Dict as returned by :meth:`quantize_kv_cache`.

        Returns:
            ``(k, v)`` — float32 tensors of ``cache_dict['original_shape']``.
        """
        original_shape = cache_dict["original_shape"]

        k = self.dequantize(
            cache_dict["k_q"],
            cache_dict["k_scales"],
            cache_dict["k_zp"],
            original_shape=original_shape,
        )
        v = self.dequantize(
            cache_dict["v_q"],
            cache_dict["v_scales"],
            cache_dict["v_zp"],
            original_shape=original_shape,
        )
        return k, v

    # ------------------------------------------------------------------
    # Memory analysis
    # ------------------------------------------------------------------

    def memory_saved_mb(
        self,
        seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        n_layers: int,
    ) -> dict:
        """Compute memory savings vs float16 KV cache.

        The formula counts *both* K and V caches across all layers.

        Args:
            seq_len: Sequence length.
            n_kv_heads: Number of KV attention heads.
            head_dim: Dimension of each head.
            n_layers: Number of transformer layers.

        Returns:
            Dictionary::

                {
                    'fp16_mb':       total FP16 KV cache size in MiB,
                    'quantized_mb':  quantized KV cache size in MiB,
                    'savings_factor': fp16_mb / quantized_mb,
                }

        Notes:
            Overhead for scales and zero-points is included.  Each group of
            ``group_size`` elements requires one float32 scale (and optionally
            one float32 zero-point).
        """
        # Elements per K or V tensor (one layer, batch=1)
        n_elements = seq_len * n_kv_heads * head_dim

        # FP16 bytes: 2 bytes per element, K+V, all layers
        fp16_bytes = 2 * n_elements * 2 * n_layers

        # Quantized integer bytes
        int_bytes_per_element = self.bits / 8  # 1.0 for int8, 0.5 for int4
        int_bytes = int_bytes_per_element * n_elements * 2 * n_layers

        # Scale overhead: one float32 per group per (K or V) per layer
        n_groups = math.ceil(n_elements / self.group_size)
        scale_bytes = 4 * n_groups * 2 * n_layers  # 4 bytes per float32

        # Zero-point overhead (asymmetric only)
        zp_bytes = 0.0
        if not self.symmetric:
            zp_bytes = 4 * n_groups * 2 * n_layers

        quantized_bytes = int_bytes + scale_bytes + zp_bytes

        fp16_mb = fp16_bytes / (1024 ** 2)
        quantized_mb = quantized_bytes / (1024 ** 2)
        savings_factor = fp16_mb / max(quantized_mb, 1e-9)

        return {
            "fp16_mb": fp16_mb,
            "quantized_mb": quantized_mb,
            "savings_factor": savings_factor,
        }
