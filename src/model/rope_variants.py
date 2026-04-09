"""RoPE variants: ALiBi, T5-relative position bias, dynamic NTK scaling, and position interpolation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PositionConfig:
    max_seq_len: int = 4096
    n_heads: int = 8
    alibi_bias_max: float = 8.0       # ALiBi max slope
    t5_num_buckets: int = 32          # T5 relative attention buckets
    t5_max_distance: int = 128
    rope_base: float = 10000.0
    dynamic_scale_factor: float = 1.0


def build_alibi_slopes(n_heads: int) -> Tensor:
    """Compute ALiBi per-head slopes.

    Slopes: m_h = 2^(-8 * h / n_heads) for h in 1..n_heads.

    Returns:
        Tensor of shape (n_heads,).
    """
    slopes = torch.tensor(
        [2 ** (-8.0 * h / n_heads) for h in range(1, n_heads + 1)],
        dtype=torch.float32,
    )
    return slopes


def compute_alibi_bias(seq_len: int, n_heads: int, alibi_slopes: Tensor) -> Tensor:
    """Compute ALiBi attention bias matrix.

    For each head h: bias[h, i, j] = -|i - j| * slopes[h], with upper
    triangular positions (future tokens) masked to -inf for causal attention.

    Args:
        seq_len:       Sequence length T.
        n_heads:       Number of attention heads H.
        alibi_slopes:  Per-head slopes, shape (H,).

    Returns:
        Tensor of shape (1, n_heads, seq_len, seq_len).
    """
    # Relative positions: (T, T) — value is -(j - i) distances
    positions = torch.arange(seq_len, device=alibi_slopes.device)
    # distance matrix: dist[i, j] = |i - j|
    dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()  # (T, T)

    # bias[h, i, j] = -dist[i, j] * slopes[h]
    # slopes: (H,) -> (H, 1, 1)
    bias = -dist.unsqueeze(0) * alibi_slopes.view(n_heads, 1, 1)  # (H, T, T)

    # Causal mask: mask future positions (j > i) to -inf
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=alibi_slopes.device, dtype=torch.bool), diagonal=1)
    bias = bias.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

    return bias.unsqueeze(0)  # (1, H, T, T)


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi positional bias (no positional embeddings)."""

    def __init__(self, d_model: int, n_heads: int, config: PositionConfig) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.config = config

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        slopes = build_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        """
        Args:
            x:      Input tensor of shape (B, T, d_model).
            causal: Whether to apply causal masking.

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        H = self.n_heads
        D = self.head_dim

        # QKV projections and reshape to (B, H, T, D)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Scaled dot-product scores: (B, H, T, T)
        scale = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Add ALiBi bias
        alibi = compute_alibi_bias(T, H, self.slopes)  # (1, H, T, T)
        scores = scores + alibi

        # If not causal, undo the -inf masking that compute_alibi_bias applied
        if not causal:
            # Recompute without causal mask
            positions = torch.arange(T, device=x.device)
            dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
            alibi_no_causal = -dist.unsqueeze(0) * self.slopes.view(H, 1, 1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale + alibi_no_causal.unsqueeze(0)

        # Softmax and weighted sum
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, D)

        # Reshape to (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


def t5_relative_position_bucket(
    relative_position: Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> Tensor:
    """Bucket relative positions for T5 bias.

    First half of buckets: exact positions (small distances).
    Second half: logarithmic spacing (large distances).
    Handles both positive and negative relative positions.

    Args:
        relative_position: Tensor of shape (query_len, key_len) with values i - j.
        num_buckets:        Total number of buckets.
        max_distance:       Maximum distance for logarithmic range.

    Returns:
        Bucket indices, shape (query_len, key_len), values in [0, num_buckets).
    """
    ret = torch.zeros_like(relative_position)
    n = -relative_position  # negate: positive means past (key behind query)

    num_buckets //= 2
    # Add num_buckets to the bucket index for negative relative positions
    ret += (n < 0).long() * num_buckets
    n = n.abs()

    # Half the buckets for exact small distances
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # Logarithmic spacing for larger distances
    val_if_large = max_exact + (
        torch.log(n.float().clamp(min=1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long().clamp(max=num_buckets - 1)

    ret += torch.where(is_small, n, val_if_large)
    return ret.clamp(0, 2 * num_buckets - 1)


class T5RelativeAttentionBias(nn.Module):
    """T5-style relative position bias table (learnable)."""

    def __init__(self, config: PositionConfig) -> None:
        super().__init__()
        self.config = config
        # Learnable bias table: (num_buckets, n_heads)
        self.bias_table = nn.Embedding(config.t5_num_buckets, config.n_heads)

    def forward(self, query_len: int, key_len: int) -> Tensor:
        """Compute T5 relative attention bias.

        Args:
            query_len: Query sequence length.
            key_len:   Key sequence length.

        Returns:
            Bias tensor of shape (1, n_heads, query_len, key_len).
        """
        device = self.bias_table.weight.device

        query_pos = torch.arange(query_len, device=device)  # (Q,)
        key_pos = torch.arange(key_len, device=device)       # (K,)

        # Relative positions: (Q, K), value = i - j
        relative_position = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)  # (Q, K)

        # Bucket indices
        buckets = t5_relative_position_bucket(
            relative_position,
            num_buckets=self.config.t5_num_buckets,
            max_distance=self.config.t5_max_distance,
        )  # (Q, K)

        # Lookup bias table: (Q, K, n_heads)
        bias = self.bias_table(buckets)

        # Reshape to (1, n_heads, Q, K)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, Q, K)
        return bias


def dynamic_ntk_scaling(
    seq_len: int,
    base: float,
    d_model: int,
    scale_factor: float = 1.0,
) -> Tensor:
    """Compute RoPE frequencies with dynamic NTK scaling.

    If seq_len exceeds baseline (original_max via scale_factor), the base
    frequency is scaled: base_new = base * (scale_factor * seq_len / original_max)^(d/(d-2)).

    When scale_factor == 1.0, applies no extra scaling (seq_len treated as original max).

    Args:
        seq_len:       Current sequence length.
        base:          RoPE base frequency (e.g. 10000.0).
        d_model:       Model / head dimension (must be even).
        scale_factor:  Dynamic scale multiplier; values > 1 trigger NTK rescaling.

    Returns:
        Frequency theta values, shape (d_model // 2,).
    """
    if scale_factor > 1.0:
        # Scale the base: new_base = base * scale_factor^(d/(d-2))
        scaled_base = base * (scale_factor ** (d_model / (d_model - 2)))
    else:
        scaled_base = base

    half_dim = d_model // 2
    i = torch.arange(0, half_dim, dtype=torch.float32)
    freqs = 1.0 / (scaled_base ** (2.0 * i / d_model))
    return freqs  # (d_model // 2,)


def interpolate_rope_positions(positions: Tensor, scale: float) -> Tensor:
    """Position interpolation for context extension (Chen et al.).

    Scale down positions: positions / scale.

    Args:
        positions: Position indices, shape (T,).
        scale:     Scale factor (> 1 compresses positions into original range).

    Returns:
        Scaled positions, shape (T,), float tensor.
    """
    return positions.float() / scale
