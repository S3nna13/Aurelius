"""Sliding Window Attention (SWA) for the Aurelius LLM.

Each token attends only to the W most recent tokens in its local window,
reducing attention complexity from O(N^2) to O(N*W).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SWAConfig:
    d_model: int = 64
    n_heads: int = 4
    window_size: int = 128
    causal: bool = True
    dropout_p: float = 0.0


# ---------------------------------------------------------------------------
# Mask builder
# ---------------------------------------------------------------------------


def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> Tensor:
    """Return an additive float mask of shape (seq_len, seq_len).

    Values:
        0.0   — token i is allowed to attend to token j
        -1e9  — token i is blocked from attending to token j

    Causal mode (causal=True):
        Position i attends to j if  i - window_size <= j <= i.

    Non-causal / bidirectional mode (causal=False):
        Position i attends to j if  |i - j| <= window_size // 2.
    """
    i_idx = torch.arange(seq_len).unsqueeze(1)  # (T, 1)
    j_idx = torch.arange(seq_len).unsqueeze(0)  # (1, T)

    if causal:
        # j must not be in the future AND must be within the window
        attend = (j_idx <= i_idx) & (j_idx >= i_idx - window_size)
    else:
        half = window_size // 2
        attend = (j_idx - i_idx).abs() <= half

    # Convert bool attend mask → additive float mask
    mask = torch.where(attend, torch.zeros(seq_len, seq_len), torch.full((seq_len, seq_len), -1e9))
    return mask


# ---------------------------------------------------------------------------
# Functional sliding-window attention
# ---------------------------------------------------------------------------


def sliding_window_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    window_size: int,
    causal: bool = True,
    scale: float | None = None,
) -> Tensor:
    """Scaled dot-product attention with a sliding window mask.

    Args:
        Q: (B, H, T, d_head)
        K: (B, H, T, d_head)
        V: (B, H, T, d_head)
        window_size: number of past tokens each position may attend to
        causal: if True apply causal constraint in addition to window
        scale: optional scale factor; defaults to 1/sqrt(d_head)

    Returns:
        (B, H, T, d_head)
    """
    B, H, T, d_head = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    # Build mask on same device/dtype
    mask = build_sliding_window_mask(T, window_size, causal=causal).to(
        device=Q.device, dtype=Q.dtype
    )

    # scores: (B, H, T, T)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores = scores + mask  # additive mask — blocked positions become -1e9

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


# ---------------------------------------------------------------------------
# SlidingWindowAttention module
# ---------------------------------------------------------------------------


class SlidingWindowAttention(nn.Module):
    """Multi-head attention with sliding window restriction.

    Uses separate q/k/v/out linear projections (no bias).
    """

    def __init__(self, config: SWAConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, (  # noqa: S101
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.window_size = config.window_size
        self.causal = config.causal
        self.dropout_p = config.dropout_p

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        H, d_h = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)
        K = self.k_proj(x).view(B, T, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d_h).transpose(1, 2)

        out = sliding_window_attention(
            Q,
            K,
            V,
            window_size=self.window_size,
            causal=self.causal,
        )  # (B, H, T, d_h)

        out = out.transpose(1, 2).contiguous().view(B, T, H * d_h)  # (B, T, d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SWABlock: pre-norm + SWA + residual
# ---------------------------------------------------------------------------


class SWABlock(nn.Module):
    """Transformer block with pre-LayerNorm + SlidingWindowAttention + residual."""

    def __init__(self, config: SWAConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = SlidingWindowAttention(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) — residual-connected output
        """
        return x + self.attn(self.norm(x))


# ---------------------------------------------------------------------------
# Diagnostic utility
# ---------------------------------------------------------------------------


def compare_swa_vs_full_attention(
    x: Tensor,
    swa: SlidingWindowAttention,
    window_size: int,
) -> float:
    """Compare SWA output to full causal attention when window covers all tokens.

    For sequences where window_size >= seq_len, SWA is equivalent to full
    causal attention.  Uses the same projection weights.

    Returns:
        Maximum absolute difference between SWA and full-attention outputs.
    """
    B, T, d_model = x.shape
    H = swa.n_heads
    d_h = swa.d_head
    scale = 1.0 / math.sqrt(d_h)

    with torch.no_grad():
        Q = swa.q_proj(x).view(B, T, H, d_h).transpose(1, 2)
        K = swa.k_proj(x).view(B, T, H, d_h).transpose(1, 2)
        V = swa.v_proj(x).view(B, T, H, d_h).transpose(1, 2)

        # SWA output
        swa_out = sliding_window_attention(
            Q, K, V, window_size=window_size, causal=True, scale=scale
        )

        # Full causal mask (no window restriction — window = T covers everything)
        full_out = sliding_window_attention(Q, K, V, window_size=T, causal=True, scale=scale)

        diff = (swa_out - full_out).abs().max().item()
    return diff
