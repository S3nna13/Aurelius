"""INT4 KV cache quantization (KIVI, Liu 2024, arXiv:2402.02750).

This module implements the canonical KIVI configuration:

* K cache is quantized **per-channel** (along the last, head-dim axis) using
  asymmetric min/max scaling. Groups are formed along ``D`` per
  ``(batch, head, token)`` tuple, giving scale/zero per
  ``(B, H, S, D // group_size)``.
* V cache is quantized **per-token** (along the sequence axis) using
  asymmetric min/max scaling. Groups are formed along ``S`` per
  ``(batch, head, dim)`` tuple, giving scale/zero per
  ``(B, H, S // group_size, D)``.
* Every value is packed to 4 bits; two INT4 values share one ``uint8``
  byte. Quantized buffers therefore use ``[B, H, S, D // 2]`` storage
  for K and ``[B, H, S // 2, D]``-compatible storage for V (packed along
  the same axis as the grouping to keep streaming appends trivial, we
  pack V along the ``D`` axis so the streaming ``S`` dimension stays
  independent).

This is intentionally a separate strategy from the baseline
``kv_compression.KVInt8Compressor`` (INT8 symmetric per-head). The two
live side-by-side so callers can A/B memory vs. reconstruction error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# INT4 pack / unpack helpers
# ---------------------------------------------------------------------------


def pack_int4(x: Tensor) -> Tensor:
    """Pack an even-length last-axis tensor of ints in ``[0, 15]`` to uint8.

    Two 4-bit values are packed into one byte: low nibble = even index,
    high nibble = odd index. The input must be an integer-valued tensor
    with values already clamped to ``[0, 15]``; out-of-range inputs raise
    ``ValueError`` (no silent wrap-around).
    """
    if x.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"pack_int4 expects integer dtype, got {x.dtype}")
    if x.numel() == 0:
        return x.to(torch.uint8)
    xmin = int(x.min().item())
    xmax = int(x.max().item())
    if xmin < 0 or xmax > 15:
        raise ValueError(f"pack_int4 input out of range [0,15]: min={xmin} max={xmax}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"pack_int4 requires even last dim, got shape {tuple(x.shape)}")
    x_u8 = x.to(torch.uint8)
    low = x_u8[..., 0::2]
    high = x_u8[..., 1::2]
    return (low | (high << 4)).contiguous()


def unpack_int4(x_uint8: Tensor) -> Tensor:
    """Inverse of :func:`pack_int4`. Returns an ``int8`` tensor in ``[0, 15]``.

    Output last-axis length is ``2 * x_uint8.shape[-1]``.
    """
    if x_uint8.dtype != torch.uint8:
        raise ValueError(f"unpack_int4 expects uint8, got {x_uint8.dtype}")
    low = (x_uint8 & 0x0F).to(torch.int8)
    high = ((x_uint8 >> 4) & 0x0F).to(torch.int8)
    out = torch.stack([low, high], dim=-1)
    # Interleave along last axis: [..., N, 2] -> [..., 2N]
    out = out.reshape(*x_uint8.shape[:-1], x_uint8.shape[-1] * 2)
    return out.contiguous()


# ---------------------------------------------------------------------------
# Asymmetric min/max INT4 quantizer (grouped)
# ---------------------------------------------------------------------------


def _quantize_asymmetric_int4(
    x: Tensor, group_dim: int, group_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize ``x`` to INT4 with asymmetric min/max scaling along ``group_dim``.

    Returns ``(q_int8, scale, zero)`` where ``q_int8`` has the same shape
    as ``x`` with integer values in ``[0, 15]`` (still unpacked), and
    ``scale`` / ``zero`` are broadcastable to ``x``'s shape with the
    grouping axis collapsed per ``group_size``.
    """
    if x.shape[group_dim] % group_size != 0:
        raise ValueError(
            f"group_size={group_size} must divide size along dim {group_dim} "
            f"(got {x.shape[group_dim]})"
        )

    x_fp = x.to(torch.float32)
    x_fp.dim()
    # Reshape so the group axis becomes two axes: (num_groups, group_size).
    new_shape = list(x_fp.shape)
    n_groups = new_shape[group_dim] // group_size
    new_shape[group_dim] = n_groups
    new_shape.insert(group_dim + 1, group_size)
    x_grouped = x_fp.reshape(new_shape)

    # Min / max reduce over the inner group axis.
    reduce_axis = group_dim + 1
    g_min = x_grouped.amin(dim=reduce_axis, keepdim=True)
    g_max = x_grouped.amax(dim=reduce_axis, keepdim=True)
    span = (g_max - g_min).clamp(min=1e-8)
    scale = span / 15.0  # strictly positive

    # floor((x - min) / span * 15), clamp to [0, 15].
    q = torch.floor((x_grouped - g_min) / span * 15.0)
    q = q.clamp(0.0, 15.0).to(torch.int8)

    # Flatten the group axis back into the original layout.
    q = q.reshape(x.shape)

    # Keep scale / zero in the "grouped" shape (group axis replaced by
    # n_groups, inner axis of size 1) so we can broadcast on dequant.
    zero = g_min  # asymmetric: dequant = q * scale + zero
    return q, scale.to(torch.float32), zero.to(torch.float32)


def _dequantize_asymmetric_int4(
    q_int8: Tensor, scale: Tensor, zero: Tensor, group_dim: int, group_size: int
) -> Tensor:
    """Inverse of :func:`_quantize_asymmetric_int4`."""
    new_shape = list(q_int8.shape)
    n_groups = new_shape[group_dim] // group_size
    new_shape[group_dim] = n_groups
    new_shape.insert(group_dim + 1, group_size)
    q_grouped = q_int8.to(torch.float32).reshape(new_shape)
    # Reconstruct at the bin center (q + 0.5) * scale + zero. Using bin
    # centres halves the peak reconstruction error vs. dequantising at the
    # bin floor, while staying consistent with the spec's floor-quantize.
    x = (q_grouped + 0.5) * scale + zero
    return x.reshape(q_int8.shape).to(torch.float32)


# ---------------------------------------------------------------------------
# Compressed container
# ---------------------------------------------------------------------------


@dataclass
class CompressedKIVI:
    """Container for INT4-packed KIVI KV cache state.

    Attributes
    ----------
    k_q : uint8 Tensor
        Packed K buffer of shape ``[B, H, S, D // 2]`` (two INT4 values
        per byte along ``D``).
    k_scale, k_zero : fp32 Tensor
        Per-channel K scale/zero with shape ``[B, H, S, D // group_size, 1]``.
    v_q : uint8 Tensor
        Packed V buffer of shape ``[B, H, S, D // 2]`` (two INT4 values
        per byte along ``D``; V uses per-token grouping along ``S``).
    v_scale, v_zero : fp32 Tensor
        Per-token V scale/zero with shape ``[B, H, S // group_size, 1, D]``.
    shape : tuple
        Original ``(B, H, S, D)`` of the uncompressed tensors.
    """

    k_q: Tensor
    k_scale: Tensor
    k_zero: Tensor
    v_q: Tensor
    v_scale: Tensor
    v_zero: Tensor
    shape: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# KIVIQuantizer
# ---------------------------------------------------------------------------


class KIVIQuantizer:
    """INT4 asymmetric KV-cache quantizer matching the KIVI paper.

    Parameters
    ----------
    n_heads : int
        Number of attention heads. Retained for validation against
        supplied tensors.
    head_dim : int
        Head dimension ``D``. ``group_size`` must divide ``head_dim``.
    group_size : int, default 32
        Size of the quantization groups. For K this groups along ``D``;
        for V this groups along ``S``. Must divide both ``head_dim`` and
        any sequence length handed in.
    """

    def __init__(self, n_heads: int, head_dim: int, group_size: int = 32) -> None:
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if head_dim % group_size != 0:
            raise ValueError(f"group_size ({group_size}) must divide head_dim ({head_dim})")
        if group_size % 2 != 0:
            raise ValueError(
                f"group_size must be even so INT4 pairs pack cleanly, got {group_size}"
            )
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.group_size = group_size

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_kv(self, k: Tensor, v: Tensor) -> tuple[int, int, int, int]:
        if k.shape != v.shape:
            raise ValueError(f"k and v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.dim() != 4:
            raise ValueError(f"expected 4-D [B,H,S,D] tensor, got {tuple(k.shape)}")
        B, H, S, D = k.shape
        if H != self.n_heads:
            raise ValueError(f"n_heads mismatch: tensor has {H}, quantizer {self.n_heads}")
        if D != self.head_dim:
            raise ValueError(f"head_dim mismatch: tensor has {D}, quantizer {self.head_dim}")
        if S % self.group_size != 0:
            raise ValueError(
                f"sequence length {S} must be a multiple of group_size "
                f"{self.group_size} (no silent fallback for V per-token grouping)"
            )
        return B, H, S, D

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, k: Tensor, v: Tensor) -> CompressedKIVI:
        """Quantize and pack ``k``, ``v`` into a :class:`CompressedKIVI`."""
        B, H, S, D = self._check_kv(k, v)

        # K: per-channel along D (group_dim=-1), group_size = self.group_size.
        k_q_int8, k_scale, k_zero = _quantize_asymmetric_int4(
            k, group_dim=3, group_size=self.group_size
        )
        # V: per-token along S (group_dim=2), group_size = self.group_size.
        v_q_int8, v_scale, v_zero = _quantize_asymmetric_int4(
            v, group_dim=2, group_size=self.group_size
        )

        k_q_packed = pack_int4(k_q_int8)  # [B, H, S, D/2]
        v_q_packed = pack_int4(v_q_int8)  # [B, H, S, D/2]

        return CompressedKIVI(
            k_q=k_q_packed,
            k_scale=k_scale,
            k_zero=k_zero,
            v_q=v_q_packed,
            v_scale=v_scale,
            v_zero=v_zero,
            shape=(B, H, S, D),
        )

    def decompress(self, c: CompressedKIVI) -> tuple[Tensor, Tensor]:
        """Dequantize a :class:`CompressedKIVI` back to ``(k, v)`` fp32."""
        B, H, S, D = c.shape
        k_int8 = unpack_int4(c.k_q)
        v_int8 = unpack_int4(c.v_q)
        if k_int8.shape != (B, H, S, D):
            raise ValueError(
                f"decompressed k shape {tuple(k_int8.shape)} != expected {(B, H, S, D)}"
            )
        k = _dequantize_asymmetric_int4(
            k_int8, c.k_scale, c.k_zero, group_dim=3, group_size=self.group_size
        )
        v = _dequantize_asymmetric_int4(
            v_int8, c.v_scale, c.v_zero, group_dim=2, group_size=self.group_size
        )
        return k, v

    def append(self, c: CompressedKIVI, new_k: Tensor, new_v: Tensor) -> CompressedKIVI:
        """Extend a compressed state with additional tokens.

        Semantics: decompress-then-recompress over the concatenated
        sequence. This keeps scales exactly correct for V (per-token
        grouping straddles group boundaries if ``new_S % group_size != 0``,
        which the validator rejects anyway). For KIVI's paging layout the
        concatenation is O(total_tokens); callers that need streaming
        O(new_tokens) should page in fixed-size blocks upstream.
        """
        Bn, Hn, Sn, Dn = new_k.shape
        B, H, S, D = c.shape
        if (Bn, Hn, Dn) != (B, H, D):
            raise ValueError(
                f"append shape mismatch: existing {(B, H, S, D)} vs new {tuple(new_k.shape)}"
            )
        # Validate divisibility on the combined length up front so the
        # user gets a clear error before we do work.
        if (S + Sn) % self.group_size != 0:
            raise ValueError(
                f"combined sequence length {S + Sn} must be a multiple of "
                f"group_size {self.group_size}"
            )

        k_old, v_old = self.decompress(c)
        k_cat = torch.cat([k_old, new_k.to(k_old.dtype)], dim=2)
        v_cat = torch.cat([v_old, new_v.to(v_old.dtype)], dim=2)
        return self.compress(k_cat, v_cat)
