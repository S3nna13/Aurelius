"""Ring Attention (Liu & Abbeel, 2023 — arXiv:2310.01889).

Single-device emulation of Ring Attention. We chunk K and V along the
sequence dimension and iterate, maintaining online-softmax running
statistics (max `m`, log-sum-exp normalizer `l`, unnormalized weighted
output `out`). Each new chunk rebases the running state so that the final
normalized output is bit-equivalent (up to fp rounding) to a single-pass
scaled-dot-product-attention.

IMPORTANT: this is a **single-device emulation**. There is no distributed
all-to-all ring communication here — the "ring" is simulated by a Python
loop over K/V chunks. A production distributed implementation (collective
ring-send across ranks) is out of scope for this module.

Memory: the full [B, H, S, D] K and V tensors still live in memory on the
caller's side, but intermediate attention-score and probability buffers
are only [B, H, S_q, chunk_size] per step instead of [B, H, S_q, S_k].
That is the memory win this module actually delivers on a single device.
The distributed variant is what gets you the K/V-sharded win; the algebra
is identical.

Public surface:
    ring_attention(q, k, v, chunk_size=128, causal=False, mask=None,
                   scale=None) -> Tensor
    RingAttention(chunk_size=128, causal=False)  # registry-friendly
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_qkv(q: Tensor, k: Tensor, v: Tensor) -> None:
    if not (isinstance(q, Tensor) and isinstance(k, Tensor) and isinstance(v, Tensor)):
        raise TypeError("ring_attention: q, k, v must all be torch.Tensor")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"ring_attention: expected 4-D [B,H,S,D] tensors, got "
            f"q.dim={q.dim()}, k.dim={k.dim()}, v.dim={v.dim()}"
        )
    B_q, H_q, _, D_q = q.shape
    B_k, H_k, S_k, D_k = k.shape
    B_v, H_v, S_v, D_v = v.shape
    if (B_q, H_q) != (B_k, H_k) or (B_q, H_q) != (B_v, H_v):
        raise ValueError(f"ring_attention: batch/head mismatch q={q.shape} k={k.shape} v={v.shape}")
    if D_q != D_k:
        raise ValueError(f"ring_attention: q/k head-dim mismatch D_q={D_q} D_k={D_k}")
    if S_k != S_v:
        raise ValueError(f"ring_attention: k/v seq-len mismatch S_k={S_k} S_v={S_v}")
    if D_v <= 0 or D_q <= 0:
        raise ValueError("ring_attention: head dim must be positive")


def ring_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int = 128,
    causal: bool = False,
    mask: Tensor | None = None,
    scale: float | None = None,
) -> Tensor:
    """Compute scaled-dot-product attention via a chunked online-softmax ring.

    Args:
        q: [B, H, S_q, D]
        k: [B, H, S_k, D]
        v: [B, H, S_k, D_v]
        chunk_size: size of each K/V chunk along the sequence axis (>=1).
            If chunk_size >= S_k, a single-chunk fallback is used (still
            exercises the online-softmax code path for equivalence).
        causal: if True, apply a causal mask (keys with absolute position
            > query absolute position are masked out). Query positions
            are taken as [0, S_q) and key positions as [0, S_k); this
            matches the standard decoder-self-attention convention when
            S_q == S_k.
        mask: optional additive float mask broadcastable to
            [B, H, S_q, S_k]. Use -inf to forbid attention; 0 is a no-op.
        scale: softmax temperature; defaults to 1/sqrt(D).

    Returns:
        Tensor of shape [B, H, S_q, D_v] with dtype/device matching q.
    """
    _validate_qkv(q, k, v)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"ring_attention: chunk_size must be int >=1, got {chunk_size!r}")

    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    D_v = v.shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if mask is not None:
        if not isinstance(mask, Tensor):
            raise TypeError("ring_attention: mask must be a Tensor or None")
        # Try broadcast-shape check eagerly.
        try:
            torch.broadcast_shapes(mask.shape, (B, H, S_q, S_k))
        except RuntimeError as exc:
            raise ValueError(
                f"ring_attention: mask shape {tuple(mask.shape)} not broadcastable "
                f"to [B,H,S_q,S_k]=({B},{H},{S_q},{S_k})"
            ) from exc

    # Running online-softmax state, all in q's dtype/device.
    # m: running max, shape [B, H, S_q, 1]
    # l: running denominator sum-of-exps (rebased), shape [B, H, S_q, 1]
    # out: running weighted value sum (rebased), shape [B, H, S_q, D_v]
    neg_inf = torch.finfo(q.dtype).min
    m = torch.full((B, H, S_q, 1), neg_inf, dtype=q.dtype, device=q.device)
    item = torch.zeros((B, H, S_q, 1), dtype=q.dtype, device=q.device)
    out = torch.zeros((B, H, S_q, D_v), dtype=q.dtype, device=q.device)

    # Query absolute positions (for causal mask). Standard decoder convention:
    # query i attends to keys [0..i] when S_q == S_k. For S_q != S_k we
    # align queries to the *tail* of keys: q_pos = i + (S_k - S_q).
    q_offset = S_k - S_q

    chunk = min(chunk_size, S_k) if S_k > 0 else 1

    # S_k == 0 edge case: no keys -> undefined; return zeros rather than NaN.
    if S_k == 0:
        return out

    for start in range(0, S_k, chunk):
        end = min(start + chunk, S_k)
        k_chunk = k[:, :, start:end, :]  # [B,H,c,D]
        v_chunk = v[:, :, start:end, :]  # [B,H,c,D_v]

        # logits: [B,H,S_q,c]
        logits = torch.matmul(q, k_chunk.transpose(-1, -2)) * scale

        if causal:
            # q_pos shape [S_q,1], k_pos shape [1,c]
            q_pos = torch.arange(S_q, device=q.device).unsqueeze(-1) + q_offset
            k_pos = torch.arange(start, end, device=q.device).unsqueeze(0)
            causal_block = k_pos > q_pos  # [S_q,c] bool
            if causal_block.any():
                logits = logits.masked_fill(causal_block, neg_inf)

        if mask is not None:
            mask_chunk = mask[..., start:end] if mask.shape[-1] == S_k else mask
            # Broadcast-add; for chunked mask slice along last dim.
            logits = logits + mask_chunk

        # Online-softmax update.
        # m_new = max(m, rowmax(logits))
        chunk_max = logits.amax(dim=-1, keepdim=True)  # [B,H,S_q,1]
        # If an entire query row is fully masked in this chunk, chunk_max
        # will be neg_inf; that's fine — exp(logits - m_new) stays 0.
        m_new = torch.maximum(m, chunk_max)

        # Rescale previous accumulators to the new max.
        # exp(m - m_new) is in (0, 1]. For the very first chunk m == neg_inf,
        # so (m - m_new) is -inf; exp -> 0, wiping the uninitialized state.
        alpha = torch.exp(m - m_new)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))

        # Probs for this chunk, rebased.
        p = torch.exp(logits - m_new)  # [B,H,S_q,c]
        # If m_new is neg_inf for a row (no finite logits ever), p is NaN
        # due to -inf - -inf; zero it out.
        p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))

        # Update accumulators.
        out = out * alpha + torch.matmul(p, v_chunk)  # [B,H,S_q,D_v]
        item = item * alpha + p.sum(dim=-1, keepdim=True)  # [B,H,S_q,1]
        m = m_new

    # Normalize. Guard against l == 0 (fully-masked rows) -> output zeros.
    safe_l = torch.where(item > 0, item, torch.ones_like(item))
    out = out / safe_l
    out = torch.where(item > 0, out, torch.zeros_like(out))
    return out


class RingAttention:
    """Callable wrapper around :func:`ring_attention` for registry use.

    Stateless apart from configuration (`chunk_size`, `causal`). Matches
    the LONGCONTEXT_STRATEGY_REGISTRY constructor convention used by
    sibling strategies.
    """

    def __init__(self, chunk_size: int = 128, causal: bool = False) -> None:
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError(f"RingAttention: chunk_size must be int >=1, got {chunk_size!r}")
        self.chunk_size = chunk_size
        self.causal = bool(causal)

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        scale: float | None = None,
    ) -> Tensor:
        return ring_attention(
            q,
            k,
            v,
            chunk_size=self.chunk_size,
            causal=self.causal,
            mask=mask,
            scale=scale,
        )

    def __repr__(self) -> str:
        return f"RingAttention(chunk_size={self.chunk_size}, causal={self.causal})"


__all__ = ["ring_attention", "RingAttention"]
