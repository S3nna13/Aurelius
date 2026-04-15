"""RoPE (Rotary Position Embedding) caching for the Aurelius LLM project.

Pre-computes and stores cos/sin tables to avoid recomputation every forward
pass.  The cache is built once at construction time and sliced per-batch.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RopeCacheConfig:
    """Configuration for the RoPE cos/sin cache.

    Attributes:
        d_head: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to pre-compute.
        base: Frequency base (default 10000.0).
        dtype: String dtype for the cache tensors ("float32" or "float16").
    """

    d_head: int = 64
    max_seq_len: int = 2048
    base: float = 10000.0
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def build_cos_sin_cache(
    d_head: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cosine and sine tables for RoPE.

    For dimension index i in [0, d_head//2):
        theta_i = 1 / (base ^ (2*i / d_head))
        angle[pos, i] = pos * theta_i

    Args:
        d_head: Head dimension (must be even).
        max_seq_len: Number of positions to pre-compute.
        base: Frequency base.

    Returns:
        Tuple (cos_cache, sin_cache) each of shape (max_seq_len, d_head // 2).
    """
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    half = d_head // 2

    # theta_i = 1 / (base^(2*i / d_head)) for i in [0, half)
    i = torch.arange(0, half, dtype=torch.float32)         # (half,)
    theta = 1.0 / (base ** (2.0 * i / d_head))             # (half,)

    # positions
    t = torch.arange(max_seq_len, dtype=torch.float32)     # (max_seq_len,)

    # angles[pos, i] = pos * theta_i
    angles = torch.outer(t, theta)                          # (max_seq_len, half)

    cos_cache = angles.cos()                                # (max_seq_len, half)
    sin_cache = angles.sin()                                # (max_seq_len, half)

    return cos_cache, sin_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by 90 degrees in each consecutive pair.

    Given x of shape (..., d), split into [x1 | x2] where each half has
    size d//2, and return [-x2 | x1].

    Args:
        x: Tensor of shape (..., d) where d is even.

    Returns:
        Tensor of same shape as x.
    """
    d = x.shape[-1]
    assert d % 2 == 0, "Last dimension must be even for rotate_half"
    x1 = x[..., : d // 2]   # first half
    x2 = x[..., d // 2 :]   # second half
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_with_cache(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Apply RoPE using pre-computed cos/sin tables.

    Rotation formula (real-valued):
        [x1, x2]_rotated = [x1*cos - x2*sin, x1*sin + x2*cos]

    which is equivalent to:
        x_rotated = x * cos + rotate_half(x) * sin

    where rotate_half(x) = [-x2, x1].

    Args:
        x: Input tensor of shape (..., T, d_head).
        cos: Cosine cache of shape (T, d_head//2).
        sin: Sine cache of shape (T, d_head//2).
        seq_dim: Which dimension of x is the sequence dimension (default 1).

    Returns:
        Rotated tensor of same shape as x.
    """
    T = x.shape[seq_dim]
    d_head = x.shape[-1]
    half = d_head // 2

    # Slice to actual sequence length
    cos_t = cos[:T]   # (T, half)
    sin_t = sin[:T]   # (T, half)

    # Expand cos/sin to full d_head by repeating each half: [cos|cos]  [sin|sin]
    # This matches the split [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    cos_full = torch.cat([cos_t, cos_t], dim=-1)   # (T, d_head)
    sin_full = torch.cat([sin_t, sin_t], dim=-1)   # (T, d_head)

    # Broadcast over all leading dims: x is (..., T, d_head)
    return x * cos_full + rotate_half(x) * sin_full


# ---------------------------------------------------------------------------
# RopeCache class
# ---------------------------------------------------------------------------

class RopeCache:
    """Stores pre-computed cos/sin RoPE tables and applies them on demand.

    Not an nn.Module so it can be shared without affecting parameter counts.
    The buffers are plain tensors (no gradient tracking).
    """

    def __init__(self, config: RopeCacheConfig) -> None:
        self.config = config
        torch_dtype = getattr(torch, config.dtype)

        cos_cache, sin_cache = build_cos_sin_cache(
            d_head=config.d_head,
            max_seq_len=config.max_seq_len,
            base=config.base,
        )

        # Store as plain tensors (no grad)
        self.cos: torch.Tensor = cos_cache.to(torch_dtype)
        self.sin: torch.Tensor = sin_cache.to(torch_dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) sliced to seq_len positions.

        Args:
            seq_len: Number of positions to return (<= max_seq_len).

        Returns:
            Tuple (cos[:seq_len], sin[:seq_len]) each of shape
            (seq_len, d_head // 2).
        """
        return self.cos[:seq_len], self.sin[:seq_len]

    def apply(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply cached RoPE to x.

        Args:
            x: Input tensor of shape (B, T, d_head).
            seq_len: Number of positions to use (usually T).

        Returns:
            Rotated tensor of same shape as x.
        """
        cos_t, sin_t = self.get(seq_len)
        # Move cache to same device/dtype as x if needed
        cos_t = cos_t.to(device=x.device, dtype=x.dtype)
        sin_t = sin_t.to(device=x.device, dtype=x.dtype)
        return apply_rotary_with_cache(x, cos_t, sin_t)


# ---------------------------------------------------------------------------
# CachedRoPEAttention nn.Module
# ---------------------------------------------------------------------------

class CachedRoPEAttention(nn.Module):
    """Multi-head self-attention with cached RoPE applied to Q and K.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        config: RopeCacheConfig for cache construction.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        config: RopeCacheConfig,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.config = config

        # Linear projections — no bias (common in modern LLMs)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Build RoPE cache
        self.rope = RopeCache(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention with RoPE applied to Q and K.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape

        # Project
        q = self.q_proj(x)   # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, T, n_heads, d_head) then transpose to
        # (B, n_heads, T, d_head) for attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K — reshape to (B*n_heads, T, d_head) for
        # apply_rotary_with_cache (seq_dim=1)
        Bh = B * self.n_heads
        q = q.reshape(Bh, T, self.d_head)
        k = k.reshape(Bh, T, self.d_head)

        cos_t, sin_t = self.rope.get(T)
        cos_t = cos_t.to(device=x.device, dtype=x.dtype)
        sin_t = sin_t.to(device=x.device, dtype=x.dtype)

        q = apply_rotary_with_cache(q, cos_t, sin_t)
        k = apply_rotary_with_cache(k, cos_t, sin_t)

        # Restore shape (B, n_heads, T, d_head)
        q = q.view(B, self.n_heads, T, self.d_head)
        k = k.view(B, self.n_heads, T, self.d_head)

        # Scaled dot-product attention (causal)
        scale = math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attn_scores = attn_scores + causal_mask
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, v)   # (B, n_heads, T, d_head)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(attn_out)
