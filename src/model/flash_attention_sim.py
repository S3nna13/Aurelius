"""Flash Attention simulation: tiled/chunked attention in pure PyTorch.

This is a reference implementation of Flash Attention's tiled computation
algorithm (Dao et al. 2022) in pure Python/PyTorch — NOT the actual CUDA
kernel, but the same block-wise online-softmax algorithm that produces
numerically equivalent results to standard O(N²) attention.

Key idea: instead of materialising the full (T, T) attention matrix, we
process Q in tiles of size ``block_size`` and, for each Q-tile, stream
through K/V in tiles, maintaining running (max, denominator, output)
accumulators via the log-sum-exp online update.

Components
----------
FlashAttnConfig        — dataclass with tiling / masking settings.
standard_attention     — reference O(N²) attention for correctness checks.
tiled_attention        — tiled block computation (the Flash Attention pattern).
FlashAttentionSim      — nn.Module wrapping tiled_attention with Q/K/V projections.
MultiHeadFlashAttn     — nn.Module adding residual connection + LayerNorm.
compare_attention_outputs — runs both impls and returns max absolute difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FlashAttnConfig:
    """Configuration for the Flash Attention simulation.

    Attributes:
        block_size:  Number of query (and key/value) rows per tile.
        causal:      If True apply causal (autoregressive) masking.
        dropout_p:   Attention dropout probability (applied during training).
        scale:       Attention scaling factor.  None → 1/sqrt(d_head).
    """

    block_size: int = 64
    causal: bool = True
    dropout_p: float = 0.0
    scale: float | None = None


# ---------------------------------------------------------------------------
# Reference implementation (O(N²))
# ---------------------------------------------------------------------------


def standard_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
) -> Tensor:
    """Standard scaled dot-product attention (O(N²) memory).

    Computes  softmax(Q K^T / sqrt(d)) V  without any tiling.

    Args:
        Q:     Query tensor  (B, H, T_q, d) or (B, T_q, d).
        K:     Key tensor    (B, H, T_k, d) or (B, T_k, d).
        V:     Value tensor  (B, H, T_k, d) or (B, T_k, d).
        mask:  Optional boolean mask of shape broadcastable to
               (..., T_q, T_k).  True positions are *masked out*
               (set to −∞ before softmax).
        scale: Scaling factor.  None → 1/sqrt(d).

    Returns:
        Output tensor of the same shape as Q.
    """
    d = Q.size(-1)
    s = scale if scale is not None else 1.0 / math.sqrt(d)

    # scores: (..., T_q, T_k)
    scores = s * torch.matmul(Q.float(), K.float().transpose(-2, -1))

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # (..., T_q, T_k)
    out = torch.matmul(weights, V.float())  # (..., T_q, d)
    return out.to(Q.dtype)


# ---------------------------------------------------------------------------
# Tiled implementation (Flash Attention online-softmax pattern)
# ---------------------------------------------------------------------------


def tiled_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    causal: bool = True,
    block_size: int = 64,
    scale: float | None = None,
) -> Tensor:
    """Tiled block computation simulating Flash Attention's online softmax.

    Processes Q in tiles of *block_size* rows.  For every Q-tile the full
    K/V sequence is streamed tile-by-tile while maintaining running
    statistics (max m, denominator l, output O) via the log-sum-exp online
    update.  This avoids materialising the full (T, T) attention matrix.

    The result is numerically equivalent to ``standard_attention`` (within
    floating-point rounding; typically atol < 1e-5 in float32).

    Args:
        Q:          Query tensor  (B, H, T, d) — 4-D expected.
        K:          Key tensor    (B, H, T, d).
        V:          Value tensor  (B, H, T, d).
        causal:     Apply autoregressive causal mask.
        block_size: Tile height/width (number of tokens per block).
        scale:      Attention scale.  None → 1/sqrt(d).

    Returns:
        Output tensor (B, H, T, d), same dtype as Q.
    """
    B, H, T, d = Q.shape
    device = Q.device
    s = scale if scale is not None else 1.0 / math.sqrt(d)

    # Promote to float32 for numerical stability
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()

    # Final output accumulator
    O = torch.zeros(B, H, T, d, dtype=torch.float32, device=device)  # noqa: E741

    for q_start in range(0, T, block_size):
        q_end = min(q_start + block_size, T)
        Tq = q_end - q_start

        Qb = Qf[:, :, q_start:q_end, :]  # (B, H, Tq, d)

        # Running accumulators for this Q-tile
        m = torch.full((B, H, Tq, 1), float("-inf"), dtype=torch.float32, device=device)
        item = torch.zeros((B, H, Tq, 1), dtype=torch.float32, device=device)
        acc = torch.zeros((B, H, Tq, d), dtype=torch.float32, device=device)

        q_pos = torch.arange(q_start, q_end, device=device)  # (Tq,)

        for kv_start in range(0, T, block_size):
            kv_end = min(kv_start + block_size, T)

            # Causal optimisation: skip entirely-future KV tiles
            if causal and kv_start >= q_end:
                break

            Kb = Kf[:, :, kv_start:kv_end, :]  # (B, H, Tkv, d)
            Vb = Vf[:, :, kv_start:kv_end, :]  # (B, H, Tkv, d)

            # scores: (B, H, Tq, Tkv)
            scores = s * torch.matmul(Qb, Kb.transpose(-2, -1))

            if causal:
                kv_pos = torch.arange(kv_start, kv_end, device=device)  # (Tkv,)
                # future_mask[i, j] = True when kv_pos[j] > q_pos[i]
                future_mask = kv_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # (Tq, Tkv)
                scores = scores.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # ---- online softmax update --------------------------------
            # chunk max
            m_new_chunk = scores.max(dim=-1, keepdim=True).values  # (B, H, Tq, 1)
            m_new = torch.maximum(m, m_new_chunk)

            # stabilised exp for this chunk
            exp_s = torch.exp(scores - m_new)  # (B, H, Tq, Tkv)
            chunk_sum = exp_s.sum(dim=-1, keepdim=True)  # (B, H, Tq, 1)

            # correction factor for previous accumulator
            corr = torch.exp(m - m_new)  # (B, H, Tq, 1)

            item = corr * item + chunk_sum
            acc = corr * acc + torch.matmul(exp_s, Vb)
            m = m_new
            # -----------------------------------------------------------

        # Normalise; clamp prevents divide-by-zero on all-masked rows
        O[:, :, q_start:q_end, :] = acc / item.clamp(min=1e-12)

    return O.to(Q.dtype)  # noqa: E741


# ---------------------------------------------------------------------------
# nn.Module: FlashAttentionSim
# ---------------------------------------------------------------------------


class FlashAttentionSim(nn.Module):
    """Multi-head attention using the tiled Flash Attention simulation.

    Projects input x into Q, K, V with separate bias-free Linear layers,
    applies ``tiled_attention`` internally, then projects the result back
    to *d_model* with an output projection.

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads.  Must divide *d_model* evenly.
        config:  ``FlashAttnConfig`` controlling block size, causality, etc.
    """

    def __init__(self, d_model: int, n_heads: int, config: FlashAttnConfig) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.config = config

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Compute multi-head flash attention.

        Args:
            x: Input tensor (B, T, d_model).

        Returns:
            Output tensor (B, T, d_model).
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        # Project and reshape to (B, H, T, Dh)
        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        out = tiled_attention(
            Q,
            K,
            V,
            causal=self.config.causal,
            block_size=self.config.block_size,
            scale=self.config.scale,
        )  # (B, H, T, Dh)

        # Reshape back: (B, H, T, Dh) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# nn.Module: MultiHeadFlashAttn
# ---------------------------------------------------------------------------


class MultiHeadFlashAttn(nn.Module):
    """``FlashAttentionSim`` wrapped with a residual connection and LayerNorm.

    Implements the standard pre-norm transformer sub-layer pattern::

        x = x + FlashAttentionSim(LayerNorm(x))

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        config:  ``FlashAttnConfig``.
    """

    def __init__(self, d_model: int, n_heads: int, config: FlashAttnConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = FlashAttentionSim(d_model, n_heads, config)

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm residual attention.

        Args:
            x: Input tensor (B, T, d_model).

        Returns:
            Output tensor (B, T, d_model).
        """
        return x + self.attn(self.norm(x))


# ---------------------------------------------------------------------------
# Utility: numerical comparison
# ---------------------------------------------------------------------------


def compare_attention_outputs(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    block_size: int = 64,
    causal: bool = True,
) -> float:
    """Run both standard and tiled attention and return max absolute difference.

    Useful for verifying that ``tiled_attention`` is numerically equivalent
    to ``standard_attention``.  The returned value should be < 1e-4 for
    typical float32 inputs.

    Args:
        Q:          Query tensor  (B, H, T, d).
        K:          Key tensor    (B, H, T, d).
        V:          Value tensor  (B, H, T, d).
        block_size: Tile size used for ``tiled_attention``.
        causal:     Whether to apply causal masking in both implementations.

    Returns:
        Maximum absolute element-wise difference between the two outputs
        (a Python float).
    """
    Q.size(-1)
    T = Q.size(-2)

    # Build causal mask for standard_attention if requested
    mask: Tensor | None = None
    if causal:
        # mask[i, j] = True when j > i  (future positions)
        idx = torch.arange(T, device=Q.device)
        mask = idx.unsqueeze(0) > idx.unsqueeze(1)  # (T, T)

    std_out = standard_attention(Q, K, V, mask=mask)
    tiled_out = tiled_attention(Q, K, V, causal=causal, block_size=block_size)

    return (std_out.float() - tiled_out.float()).abs().max().item()
