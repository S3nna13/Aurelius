"""Alternative positional encodings: ALiBi, Sinusoidal, Learned, T5 relative, and Sandwich."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ALiBi helpers
# ---------------------------------------------------------------------------

def get_alibi_slopes(n_heads: int) -> Tensor:
    """Compute ALiBi slopes for each attention head.

    For a power-of-2 number of heads, slope for head i (1-indexed) is:
        m_i = 2^(-8/n_heads)^i  =  (2^(-8/n_heads))^i

    For non-power-of-2 heads: compute slopes for the nearest power of 2,
    then interpolate to fill the remaining slots.

    Returns:
        Tensor of shape (n_heads,) with values in (0, 1).
    """
    def _slopes_for_pow2(n: int) -> Tensor:
        base = 2 ** (-8 / n)
        return torch.tensor([base ** i for i in range(1, n + 1)], dtype=torch.float32)

    # Check if n_heads is a power of 2
    if n_heads & (n_heads - 1) == 0:
        return _slopes_for_pow2(n_heads)

    # Nearest power of 2 >= n_heads
    nearest_pow2 = 2 ** math.ceil(math.log2(n_heads))
    slopes_pow2 = _slopes_for_pow2(nearest_pow2)  # (nearest_pow2,)

    # Also get slopes for nearest_pow2 // 2
    half_slopes = _slopes_for_pow2(nearest_pow2 // 2)  # (nearest_pow2 // 2,)

    # Take every other slope from the full set to interpolate
    extra_slopes = slopes_pow2[0::2]  # geometric midpoints

    # Combine: first use all half_slopes, then fill remainder with extra_slopes
    n_half = nearest_pow2 // 2
    n_extra = n_heads - n_half
    slopes = torch.cat([half_slopes, extra_slopes[:n_extra]])
    return slopes


def compute_alibi_bias(seq_len: int, slopes: Tensor) -> Tensor:
    """Compute the ALiBi causal bias matrix.

    For each head h and positions (i, j):
        bias[h, i, j] = -slopes[h] * |i - j|   if j <= i  (causal positions)
        bias[h, i, j] = 0                         if j > i  (future positions, zeroed)

    Args:
        seq_len: Sequence length T.
        slopes: (n_heads,) tensor of ALiBi slopes.

    Returns:
        Tensor of shape (n_heads, seq_len, seq_len).
    """
    n_heads = slopes.shape[0]
    device = slopes.device

    # Build distance matrix: dist[i, j] = |i - j|
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (T, T)

    # Causal mask: only positions j <= i contribute a bias; j > i gets 0
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # (T, T)
    causal_dist = dist * causal_mask  # future positions zero

    # slopes: (n_heads,) -> broadcast to (n_heads, T, T)
    bias = -slopes.view(n_heads, 1, 1) * causal_dist.unsqueeze(0)
    return bias


# ---------------------------------------------------------------------------
# ALiBi Attention module
# ---------------------------------------------------------------------------

class ALiBiAttention(nn.Module):
    """Self-attention with ALiBi positional bias.

    Args:
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        slopes = get_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute self-attention with ALiBi causal bias.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project to Q, K, V and split heads
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_heads, T, T)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add ALiBi bias: (n_heads, T, T) -> broadcast over batch
        alibi_bias = compute_alibi_bias(T, self.slopes)  # (n_heads, T, T)
        scores = scores + alibi_bias.unsqueeze(0)  # (B, n_heads, T, T)

        # Causal mask: future positions set to -inf
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Sinusoidal encoding
# ---------------------------------------------------------------------------

def sinusoidal_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> Tensor:
    """Classic fixed sinusoidal positional encoding (Vaswani et al. 2017).

    PE[pos, 2i]   = sin(pos / base^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / base^(2i / d_model))

    Args:
        seq_len: Number of positions.
        d_model: Embedding dimension (must be even for alternating sin/cos).
        base: The base for the geometric progression (default 10000).

    Returns:
        Tensor of shape (seq_len, d_model).
    """
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    # Dimension indices for even positions: 0, 2, 4, ...
    dims = torch.arange(0, d_model, 2, dtype=torch.float32)  # (d_model//2,)
    div_term = base ** (dims / d_model)  # (d_model//2,)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(positions / div_term)
    pe[:, 1::2] = torch.cos(positions / div_term)
    return pe


# ---------------------------------------------------------------------------
# Learned positional encoding
# ---------------------------------------------------------------------------

class LearnedPositionalEncoding(nn.Module):
    """Fully learned positional embeddings via an nn.Embedding table.

    Args:
        max_seq_len: Maximum supported sequence length.
        d_model: Embedding dimension.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int) -> Tensor:
        """Return positional embeddings for positions 0..seq_len-1.

        Args:
            seq_len: Number of positions to return.

        Returns:
            Tensor of shape (seq_len, d_model).
        """
        positions = torch.arange(seq_len, device=self.pe.weight.device)
        return self.pe(positions)


# ---------------------------------------------------------------------------
# T5 relative position bias (bucket indices as float tensor)
# ---------------------------------------------------------------------------

def t5_relative_position_bias(
    n_heads: int,
    seq_len: int,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> Tensor:
    """T5-style relative position bias using log-scale bucketing.

    Relative position r = j - i is bucketed into num_buckets groups:
      - First half of buckets: exact small distances (0 .. num_buckets//2 - 1)
      - Second half: log-scale over distances up to max_distance

    For simplicity this function returns the bucket indices as a float tensor
    (actual learnable head embeddings are omitted).

    Args:
        n_heads: Number of attention heads (bias is broadcast across heads).
        seq_len: Sequence length T.
        num_buckets: Total number of relative position buckets.
        max_distance: Maximum distance covered by log-scale buckets.

    Returns:
        Tensor of shape (n_heads, seq_len, seq_len) with bucket indices cast to float.
    """
    positions = torch.arange(seq_len, dtype=torch.long)
    # (seq_len, seq_len) relative positions: r = j - i
    rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)

    # Causal: only backward (r <= 0) positions; clamp positive to 0
    rel_pos = -torch.clamp(rel_pos, max=0)  # distances are non-negative, causal

    max_exact = num_buckets // 2
    is_small = rel_pos < max_exact

    # Log-scale bucket for larger distances
    log_ratio = math.log(max_distance / max_exact) if max_distance > max_exact else 1.0
    val_if_large = max_exact + (
        torch.log(rel_pos.float().clamp(min=1) / max_exact)
        / log_ratio
        * (num_buckets - max_exact)
    ).long().clamp(max=num_buckets - 1)

    buckets = torch.where(is_small, rel_pos, val_if_large)  # (T, T)

    # Expand to (n_heads, T, T) and return as float
    buckets = buckets.unsqueeze(0).expand(n_heads, -1, -1).float()
    return buckets


# ---------------------------------------------------------------------------
# Sandwich positional encoding
# ---------------------------------------------------------------------------

class SandwichPositionalEncoding(nn.Module):
    """Sandwich encoding: average of sinusoidal and learned positional encodings.

    Args:
        max_seq_len: Maximum supported sequence length.
        d_model: Embedding dimension.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.learned = LearnedPositionalEncoding(max_seq_len, d_model)

    def forward(self, seq_len: int) -> Tensor:
        """Return blended positional encoding.

        Args:
            seq_len: Number of positions.

        Returns:
            Tensor of shape (seq_len, d_model).
        """
        device = self.learned.pe.weight.device
        sin_enc = sinusoidal_encoding(seq_len, self.d_model).to(device)
        learned_enc = self.learned(seq_len)
        return (sin_enc + learned_enc) / 2
