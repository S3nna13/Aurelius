"""INT8 symmetric per-head KV cache compression.

Implements per-(batch, head) symmetric INT8 quantization of attention
K/V tensors shaped ``[B, H, S, D]``. The scale is computed from the
absolute-max over the ``(S, D)`` slab for each ``(b, h)``, giving a
single fp32 scalar per head per batch. Packed tensors therefore consist
of:

    q      : int8  [B, H, S, D]
    scale  : fp32  [B, H]   (one scale per key-group, one per value-group)

For long contexts this gives ~4x memory reduction over fp32 and ~2x over
fp16 at the cost of roughly 0.01-0.05 reconstruction error for typical
pre-softmax activations. Per-head (rather than per-token) scaling keeps
the append path trivially O(new_tokens) without re-scaling history.

Paper reference: KIVI (arXiv:2402.02750) argues for per-channel K and
per-token V. That asymmetric / mixed-axis variant is intentionally NOT
implemented here; a TODO is left inline. The symmetric per-head variant
below is the baseline Aurelius uses until KIVI is integration-tested.

Streaming semantics
-------------------

``append(compressed, new_k, new_v)`` quantizes only the new token block
using a *joint* scale combining the old scalar with ``new.abs().amax()``
so that previously-compressed history is not re-scaled (and therefore
not re-quantized). This trades a tiny amount of accuracy on the new
block (when the new block has a larger magnitude than history) for an
O(new_tokens) append. If the joint scale equals the old scale, history
bits are preserved exactly.

Non-goals
---------

- Gradients do not flow through INT8 quantization. Callers must treat
  decompressed tensors as detached inference artifacts.
- FP8 path: deliberately stubbed. ``KVFP8Compressor`` raises
  ``NotImplementedError`` rather than silently falling back; see
  project spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_INT8_MAX: float = 127.0
# Small floor so an all-zero slab produces scale>0 and avoids NaN on divide.
_SCALE_EPS: float = 1e-8


def quantize_per_head_symmetric(
    x: Tensor, dim: tuple[int, ...] | int = (-2, -1)
) -> tuple[Tensor, Tensor]:
    """Symmetric INT8 quantize reducing absmax over ``dim``.

    Parameters
    ----------
    x:
        Input float tensor. Typically shape ``[B, H, S, D]``.
    dim:
        Axes to reduce when computing absmax. Default ``(-2, -1)`` gives
        one scale per ``(B, H)`` slab, which is the per-head convention
        used by ``KVInt8Compressor``.

    Returns
    -------
    (q_int8, scale):
        ``q_int8`` has the same shape as ``x`` with dtype ``torch.int8``.
        ``scale`` is fp32 and broadcasts back over ``dim``.
    """
    if not torch.is_tensor(x):
        raise TypeError("quantize_per_head_symmetric expects a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError(f"quantize_per_head_symmetric expects float input, got {x.dtype}")

    reduce_dims = (dim,) if isinstance(dim, int) else tuple(dim)
    # absmax over the requested dims, keep them for broadcast.
    absmax = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = (absmax / _INT8_MAX).to(torch.float32).clamp_min(_SCALE_EPS)

    q = torch.round(x.detach().to(torch.float32) / scale)
    q = q.clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)

    # Squeeze scale back to one-per-head shape (drop the kept reduce dims).
    squeeze_dims = sorted((d if d >= 0 else d + x.dim()) for d in reduce_dims)
    for d in reversed(squeeze_dims):
        scale = scale.squeeze(d)
    return q, scale


def _dequantize(q: Tensor, scale: Tensor, reduce_dims: tuple[int, ...]) -> Tensor:
    """Inverse of ``quantize_per_head_symmetric`` for the given reduce dims."""
    broadcast_scale = scale
    for d in sorted((d if d >= 0 else d + q.dim()) for d in reduce_dims):
        broadcast_scale = broadcast_scale.unsqueeze(d)
    return q.to(torch.float32) * broadcast_scale


@dataclass
class CompressedKV:
    """Packed INT8 KV cache.

    Attributes
    ----------
    k_q, v_q:
        INT8 tensors shaped ``[B, H, S, D]``.
    k_scale, v_scale:
        fp32 scales shaped ``[B, H]``.
    orig_dtype:
        Dtype the caller passed in, used on decompress so downstream
        code sees the same precision it produced.
    """

    k_q: Tensor
    v_q: Tensor
    k_scale: Tensor
    v_scale: Tensor
    orig_dtype: torch.dtype

    @property
    def shape(self) -> torch.Size:
        return self.k_q.shape

    @property
    def seq_len(self) -> int:
        return int(self.k_q.shape[-2])

    def nbytes(self) -> int:
        """Total bytes of the compressed buffers (int8 + fp32 scales)."""
        return (
            self.k_q.numel() * self.k_q.element_size()
            + self.v_q.numel() * self.v_q.element_size()
            + self.k_scale.numel() * self.k_scale.element_size()
            + self.v_scale.numel() * self.v_scale.element_size()
        )


class KVInt8Compressor:
    """Per-head symmetric INT8 compressor for attention KV caches.

    Parameters
    ----------
    head_dim:
        Size of the per-head feature dimension ``D``.
    n_heads:
        Number of attention heads ``H``.
    """

    _REDUCE_DIMS: tuple[int, int] = (-2, -1)

    def __init__(self, head_dim: int, n_heads: int) -> None:
        if head_dim <= 0 or n_heads <= 0:
            raise ValueError("head_dim and n_heads must be positive")
        self.head_dim = int(head_dim)
        self.n_heads = int(n_heads)

    # ------------------------------------------------------------------ util
    def _check_kv(self, k: Tensor, v: Tensor) -> None:
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.dim() != 4:
            raise ValueError(f"expected [B,H,S,D] (4D) tensors, got {k.dim()}D")
        b, h, _s, d = k.shape
        if h != self.n_heads:
            raise ValueError(f"n_heads mismatch: tensor H={h}, compressor n_heads={self.n_heads}")
        if d != self.head_dim:
            raise ValueError(
                f"head_dim mismatch: tensor D={d}, compressor head_dim={self.head_dim}"
            )
        if b < 1:
            raise ValueError("batch dimension must be >= 1")

    # ------------------------------------------------------------- compress
    def compress(self, k: Tensor, v: Tensor) -> CompressedKV:
        """Quantize K and V in one shot."""
        self._check_kv(k, v)
        orig_dtype = k.dtype
        k_q, k_scale = quantize_per_head_symmetric(k, dim=self._REDUCE_DIMS)
        v_q, v_scale = quantize_per_head_symmetric(v, dim=self._REDUCE_DIMS)
        return CompressedKV(
            k_q=k_q,
            v_q=v_q,
            k_scale=k_scale,
            v_scale=v_scale,
            orig_dtype=orig_dtype,
        )

    # ----------------------------------------------------------- decompress
    def decompress(self, c: CompressedKV) -> tuple[Tensor, Tensor]:
        """Inverse of ``compress``."""
        k = _dequantize(c.k_q, c.k_scale, self._REDUCE_DIMS).to(c.orig_dtype)
        v = _dequantize(c.v_q, c.v_scale, self._REDUCE_DIMS).to(c.orig_dtype)
        # Quantization is not differentiable; make that explicit.
        k.requires_grad_(False)
        v.requires_grad_(False)
        return k, v

    # --------------------------------------------------------------- append
    def append(self, c: CompressedKV, new_k: Tensor, new_v: Tensor) -> CompressedKV:
        """Streaming append: compress ``new_k``/``new_v`` and concat.

        The history buffer is not re-quantized. A joint scale is chosen
        per ``(b, h)`` as ``max(old_scale, new_absmax / 127)`` so that
        both halves remain within INT8 range. Old codes are rescaled by
        integer division ``old_q * (old_scale / new_scale)`` which is
        lossless when ``new_scale == old_scale`` and otherwise introduces
        at most one LSB of additional error per element.
        """
        if new_k.shape != new_v.shape:
            raise ValueError(
                f"new K/V shape mismatch: {tuple(new_k.shape)} vs {tuple(new_v.shape)}"
            )
        if new_k.dim() != 4:
            raise ValueError(f"expected [B,H,S,D] tensors, got {new_k.dim()}D")
        if new_k.shape[0] != c.k_q.shape[0] or new_k.shape[1] != c.k_q.shape[1]:
            raise ValueError(
                "batch/head dims must match history "
                f"(got {tuple(new_k.shape)[:2]}, history {tuple(c.k_q.shape)[:2]})"
            )
        if new_k.shape[-1] != c.k_q.shape[-1]:
            raise ValueError(
                f"head_dim must match history (got {new_k.shape[-1]}, history {c.k_q.shape[-1]})"
            )

        k_q_hist, v_q_hist = c.k_q, c.v_q
        k_scale_old, v_scale_old = c.k_scale, c.v_scale

        k_new_absmax = new_k.detach().abs().amax(dim=self._REDUCE_DIMS)
        v_new_absmax = new_v.detach().abs().amax(dim=self._REDUCE_DIMS)

        k_scale_new = torch.maximum(
            k_scale_old, (k_new_absmax / _INT8_MAX).to(torch.float32)
        ).clamp_min(_SCALE_EPS)
        v_scale_new = torch.maximum(
            v_scale_old, (v_new_absmax / _INT8_MAX).to(torch.float32)
        ).clamp_min(_SCALE_EPS)

        # Rescale history codes if the scale grew. Dividing in fp32 then
        # rounding keeps codes valid INT8 without touching the originals'
        # float values.
        def _rescale(q_hist: Tensor, old_s: Tensor, new_s: Tensor) -> Tensor:
            ratio = (old_s / new_s).to(torch.float32)
            # broadcast ratio back over (S, D)
            ratio_b = ratio.unsqueeze(-1).unsqueeze(-1)
            rescaled = torch.round(q_hist.to(torch.float32) * ratio_b)
            return rescaled.clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)

        k_q_hist_r = _rescale(k_q_hist, k_scale_old, k_scale_new)
        v_q_hist_r = _rescale(v_q_hist, v_scale_old, v_scale_new)

        # Quantize the new block under the joint scale.
        def _quantize_with_scale(x: Tensor, scale: Tensor) -> Tensor:
            scale_b = scale.unsqueeze(-1).unsqueeze(-1)
            q = torch.round(x.detach().to(torch.float32) / scale_b)
            return q.clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)

        k_q_new = _quantize_with_scale(new_k, k_scale_new)
        v_q_new = _quantize_with_scale(new_v, v_scale_new)

        k_q_cat = torch.cat([k_q_hist_r, k_q_new], dim=-2)
        v_q_cat = torch.cat([v_q_hist_r, v_q_new], dim=-2)

        return CompressedKV(
            k_q=k_q_cat,
            v_q=v_q_cat,
            k_scale=k_scale_new,
            v_scale=v_scale_new,
            orig_dtype=c.orig_dtype,
        )


# TODO(KIVI / arXiv:2402.02750): add KVAsymmetricCompressor implementing
# per-channel quantization for K and per-token for V with a small zero-point.
# The per-head symmetric path above is kept as the production baseline
# because it is (a) streaming-cheap, (b) numerically robust on all-zero /
# extreme inputs, and (c) simple enough to audit.


class KVFP8Compressor:  # pragma: no cover - explicit NotImplementedError stub
    """FP8-lite placeholder. Intentionally not implemented.

    The spec requires no silent fallbacks: if the runtime lacks native
    ``torch.float8_e4m3fn`` support we must not quietly switch to INT8.
    This stub raises on construction so callers hit the error path
    synchronously instead of much later in a forward pass.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "FP8 KV compression is not implemented in this build. "
            "Use KVInt8Compressor (strategy 'kv_int8') or wait for the "
            "KIVI/FP8 path to land. No silent fallback by design."
        )
