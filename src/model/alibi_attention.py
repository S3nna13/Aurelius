"""ALiBi (Attention with Linear Biases) attention module.

Replaces positional embeddings with a linear bias added directly to attention
scores: bias[h, i, j] = -slopes[h] * |i - j|.

Reference: Press et al., "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation" (2021), https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ALiBiConfig:
    """Configuration for ALiBi attention."""

    d_model: int = 64
    n_heads: int = 8
    causal: bool = True
    max_seq_len: int = 2048


# ---------------------------------------------------------------------------
# Helper: slopes
# ---------------------------------------------------------------------------

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute per-head ALiBi slopes.

    For a power-of-2 number of heads the slopes are:
        m_i = 2^(-8*i/n_heads)  for i = 1 .. n_heads

    For non-power-of-2 heads, we compute slopes for the nearest power-of-2
    that is >= n_heads, then fill any remaining slots by interpolating between
    neighbouring power-of-2 values (following the original paper's recipe).

    Args:
        n_heads: Number of attention heads.

    Returns:
        Tensor of shape (n_heads,) with positive, decreasing slopes.
    """

    def _slopes_for_power_of_2(n: int) -> torch.Tensor:
        # i runs from 1 to n inclusive
        i = torch.arange(1, n + 1, dtype=torch.float32)
        return torch.pow(2.0, -8.0 * i / n)

    # Check if n_heads is a power of 2
    if n_heads & (n_heads - 1) == 0:
        return _slopes_for_power_of_2(n_heads)

    # Find the nearest power of 2 that is >= n_heads
    nearest_pow2 = 2 ** math.ceil(math.log2(n_heads))

    # Slopes for nearest_pow2
    slopes_full = _slopes_for_power_of_2(nearest_pow2)

    # We have nearest_pow2 // 2 < n_heads < nearest_pow2.
    # Fill the first n_heads slots: take all nearest_pow2//2 slopes, then
    # pad with every-other slope from the full set (interpolation).
    half = nearest_pow2 // 2
    slopes_half = _slopes_for_power_of_2(half)

    # Interleaved "extra" slopes between neighbouring pairs
    extra_slopes = slopes_full[torch.arange(0, half, dtype=torch.long) * 2 + 1]

    # Concatenate and take first n_heads
    slopes = torch.cat([slopes_half, extra_slopes])[:n_heads]
    return slopes


# ---------------------------------------------------------------------------
# Helper: bias matrix
# ---------------------------------------------------------------------------

def get_relative_positions(seq_len: int) -> torch.Tensor:
    """Return (T, T) matrix of |i - j| distances.

    Args:
        seq_len: Sequence length T.

    Returns:
        Long tensor of shape (T, T).
    """
    idx = torch.arange(seq_len)
    # (T, 1) - (1, T) -> (T, T)
    return (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()


def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    slopes: Optional[torch.Tensor] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Build the (n_heads, T, T) ALiBi additive bias matrix.

    bias[h, i, j] = -slopes[h] * |i - j|

    For causal attention, positions j > i are set to -inf so the model cannot
    attend to future tokens.

    Args:
        n_heads: Number of attention heads.
        seq_len: Sequence length T.
        slopes: Optional pre-computed slopes tensor of shape (n_heads,).
                If None, slopes are computed via ``get_alibi_slopes``.
        causal: If True, mask future positions with -inf.

    Returns:
        Float tensor of shape (n_heads, T, T).
    """
    if slopes is None:
        slopes = get_alibi_slopes(n_heads)

    # (T, T) distance matrix
    rel_pos = get_relative_positions(seq_len).float()  # (T, T)

    # slopes: (n_heads,) -> (n_heads, 1, 1) for broadcasting
    bias = -slopes.view(n_heads, 1, 1) * rel_pos.unsqueeze(0)  # (n_heads, T, T)

    if causal:
        # Build upper-triangle mask (j > i means future)
        future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        # Apply -inf to future positions
        bias = bias.masked_fill(future_mask.unsqueeze(0), float("-inf"))

    return bias


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class ALiBiAttention(nn.Module):
    """Multi-head self-attention with ALiBi positional bias.

    Uses no positional embeddings; instead adds a head-specific linear bias
    to the raw attention logits before softmax.

    Args:
        config: ``ALiBiConfig`` with model hyper-parameters.
    """

    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
            )

        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        # Projections — no bias, following common LLM practice
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Pre-compute and cache ALiBi slopes and bias matrix
        slopes = get_alibi_slopes(config.n_heads)  # (n_heads,)
        self.register_buffer("slopes", slopes, persistent=True)

        alibi_bias = build_alibi_bias(
            config.n_heads,
            config.max_seq_len,
            slopes=slopes,
            causal=config.causal,
        )  # (n_heads, max_seq_len, max_seq_len)
        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ALiBi self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, C = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Raw attention scores: (B, n_heads, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add ALiBi bias — slice to current seq_len
        # alibi_bias shape: (n_heads, max_seq_len, max_seq_len)
        bias_slice = self.alibi_bias[:, :T, :T]  # (n_heads, T, T)
        attn_scores = attn_scores + bias_slice.unsqueeze(0)  # broadcast over B

        # Softmax (future positions already -inf if causal)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)

        # Merge heads: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class ALiBiBlock(nn.Module):
    """Transformer block: LayerNorm -> ALiBiAttention -> residual.

    Args:
        config: ``ALiBiConfig`` with model hyper-parameters.
    """

    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = ALiBiAttention(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention with residual.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        return x + self.attn(self.norm(x))
