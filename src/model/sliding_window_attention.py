"""
Sliding Window Attention (SWA) for Aurelius LLM.

Each token attends only within a local window of size W, giving
O(T * W) complexity instead of O(T^2) full attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SWAConfig:
    """Configuration for Sliding Window Attention."""

    d_model: int = 512
    n_heads: int = 8
    window_size: int = 256
    causal: bool = True
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Mask builder
# ---------------------------------------------------------------------------


def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> Tensor:
    """Build a (T, T) additive attention mask for sliding-window attention.

    Args:
        seq_len:     Sequence length T.
        window_size: Number of tokens in the local window (W).  Each position i
                     considers positions j where |i - j| <= window_size // 2.
        causal:      If True, also mask future positions (j > i).

    Returns:
        Float tensor of shape (T, T) with values 0.0 (attend) or -inf (block).
    """
    half = window_size // 2
    i_idx = torch.arange(seq_len).unsqueeze(1)  # (T, 1)
    j_idx = torch.arange(seq_len).unsqueeze(0)  # (1, T)

    # Within-window condition: |i - j| <= half
    in_window = (i_idx - j_idx).abs() <= half  # (T, T) bool

    if causal:
        # Also require j <= i (no attending to future tokens)
        not_future = j_idx <= i_idx  # (T, T) bool
        attend = in_window & not_future
    else:
        attend = in_window

    # Build float mask: 0.0 where we attend, -inf where we block
    mask = torch.zeros(seq_len, seq_len)
    mask[~attend] = float("-inf")
    return mask


# ---------------------------------------------------------------------------
# Functional sliding-window attention
# ---------------------------------------------------------------------------


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> Tensor:
    """Scaled dot-product attention with a sliding-window mask.

    Args:
        q, k, v:     Each (B, T, d_head).
        window_size: Local window size W.
        causal:      Whether to apply causal masking.
        dropout_p:   Dropout probability applied to attention weights.

    Returns:
        Output tensor of shape (B, T, d_head).
    """
    B, T, d_head = q.shape
    scale = math.sqrt(d_head)

    # Compute raw attention scores: (B, T, T)
    scores = torch.bmm(q, k.transpose(1, 2)) / scale  # (B, T, T)

    # Build and apply the sliding-window mask
    mask = build_sliding_window_mask(T, window_size, causal=causal)
    mask = mask.to(dtype=scores.dtype, device=scores.device)
    scores = scores + mask.unsqueeze(0)  # broadcast over batch

    # Softmax — rows that are entirely -inf (e.g. T=0 edge cases) stay -inf
    attn_weights = torch.softmax(scores, dim=-1)  # (B, T, T)

    # Replace any NaN produced by softmax of all-inf rows with 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    out = torch.bmm(attn_weights, v)  # (B, T, d_head)
    return out


# ---------------------------------------------------------------------------
# Module: SlidingWindowAttention
# ---------------------------------------------------------------------------


class SlidingWindowAttention(nn.Module):
    """Multi-head sliding-window self-attention.

    Args:
        config: SWAConfig instance.
    """

    def __init__(self, config: SWAConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, (  # noqa: S101
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.config = config
        self.d_head = config.d_model // config.n_heads
        self.n_heads = config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout_p = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        H = self.n_heads
        d_head = self.d_head

        # Project and reshape to (B, H, T, d_head)
        q = self.q_proj(x).view(B, T, H, d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = self.k_proj(x).view(B, T, H, d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d_head).transpose(1, 2)

        # Apply sliding-window attention per head
        # Merge batch and head dims so sliding_window_attention sees (B*H, T, d_head)
        q_flat = q.reshape(B * H, T, d_head)
        k_flat = k.reshape(B * H, T, d_head)
        v_flat = v.reshape(B * H, T, d_head)

        out_flat = sliding_window_attention(
            q_flat,
            k_flat,
            v_flat,
            window_size=self.config.window_size,
            causal=self.config.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B*H, T, d_head)

        # Merge heads back
        out = out_flat.view(B, H, T, d_head).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Module: SWABlock
# ---------------------------------------------------------------------------


class SWABlock(nn.Module):
    """Transformer block using sliding-window attention.

    Architecture (pre-norm residual):
        y = x + Attention(LN1(x))
        z = y + FFN(LN2(y))

    FFN: d_model -> 4*d_model -> d_model with GELU activation.

    Args:
        config: SWAConfig instance.
    """

    def __init__(self, config: SWAConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = SlidingWindowAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)

        ffn_hidden = 4 * config.d_model
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, config.d_model),
        )

        if config.dropout > 0.0:
            self.drop = nn.Dropout(config.dropout)
        else:
            self.drop = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        # Pre-norm attention residual
        x = x + self.drop(self.attn(self.ln1(x)))
        # Pre-norm FFN residual
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Utility: count attended positions
# ---------------------------------------------------------------------------


def count_attended_positions(seq_len: int, window_size: int, causal: bool) -> int:
    """Count total (i, j) pairs where token j is attended by token i.

    Args:
        seq_len:     Sequence length T.
        window_size: Window size W.
        causal:      Whether causal masking is applied.

    Returns:
        Total number of attended (i, j) pairs.
    """
    half = window_size // 2
    total = 0
    for i in range(seq_len):
        if causal:
            # j must satisfy: j <= i  AND  |i - j| <= half  => i - half <= j <= i
            j_lo = max(0, i - half)
            j_hi = i  # inclusive
            total += j_hi - j_lo + 1
        else:
            # j must satisfy: |i - j| <= half => i - half <= j <= i + half
            j_lo = max(0, i - half)
            j_hi = min(seq_len - 1, i + half)
            total += j_hi - j_lo + 1
    return total
