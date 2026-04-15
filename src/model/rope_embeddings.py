"""Rotary Position Embeddings (RoPE) for the Aurelius LLM project.

RoPE encodes position by rotating query/key vectors in 2D subspaces using
complex-valued frequencies, enabling relative position awareness without
explicit position encodings.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embeddings.

    Attributes:
        d_head: Dimension of each attention head.
        max_seq_len: Maximum sequence length to pre-compute frequencies for.
        base: Base for frequency computation (default 10000.0).
        scale: Positional scaling factor for NTK/YaRN-lite style extension.
               Divides the angle, effectively increasing the effective context.
    """
    d_head: int = 64
    max_seq_len: int = 2048
    base: float = 10000.0
    scale: float = 1.0


def compute_freqs_cis(
    d_head: int,
    max_seq_len: int,
    base: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute complex RoPE frequency tensor.

    For each 2D subspace i in [0, d_head//2):
        theta_i = 1 / (base ^ (2i / d_head))
        freqs_cis[t, i] = exp(j * t * theta_i / scale)

    Args:
        d_head: Head dimension (must be even).
        max_seq_len: Number of positions to compute.
        base: Frequency base.
        scale: Position scaling factor (>1 stretches context window).

    Returns:
        Complex tensor of shape (max_seq_len, d_head // 2).
    """
    # theta_i = 1 / (base^(2i / d_head)) for i in [0, d_head//2)
    i = torch.arange(0, d_head, 2, dtype=torch.float32)  # (d_head//2,)
    theta = 1.0 / (base ** (i / d_head))                  # (d_head//2,)

    # positions t in [0, max_seq_len)
    t = torch.arange(max_seq_len, dtype=torch.float32)    # (max_seq_len,)

    # angles[t, i] = t * theta_i / scale
    angles = torch.outer(t, theta) / scale                 # (max_seq_len, d_head//2)

    # freqs_cis = exp(j * angles) = cos(angles) + j*sin(angles)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex64

    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor x using complex multiplication.

    Reshapes the last dimension of x into complex pairs, multiplies by
    freqs_cis (broadcasting over batch and heads), then flattens back.

    Args:
        x: Real tensor of shape (B, T, n_heads, d_head).
        freqs_cis: Complex tensor of shape (max_seq_len, d_head // 2).
                   Only the first T positions are used.

    Returns:
        Rotated tensor of same shape as x: (B, T, n_heads, d_head).
    """
    B, T, n_heads, d_head = x.shape
    assert d_head % 2 == 0, "d_head must be even for RoPE"

    # Reshape to complex: (B, T, n_heads, d_head//2)
    x_complex = torch.view_as_complex(x.float().reshape(B, T, n_heads, d_head // 2, 2))

    # freqs_cis: (T, d_head//2) -> broadcast over B and n_heads
    # Reshape to (1, T, 1, d_head//2) for broadcasting
    fc = freqs_cis[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_head//2)

    # Complex multiply applies the rotation
    x_rotated = x_complex * fc  # (B, T, n_heads, d_head//2)

    # Convert back to real and restore original shape
    x_out = torch.view_as_real(x_rotated).reshape(B, T, n_heads, d_head)

    return x_out.to(x.dtype)


def build_rope_cache(config: RoPEConfig) -> torch.Tensor:
    """Convenience function: build RoPE frequency cache from a RoPEConfig.

    Args:
        config: RoPEConfig instance.

    Returns:
        Complex tensor of shape (max_seq_len, d_head // 2).
    """
    return compute_freqs_cis(
        d_head=config.d_head,
        max_seq_len=config.max_seq_len,
        base=config.base,
        scale=config.scale,
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Alternative real-valued RoPE rotation: rotate by 90 degrees in each pair.

    Splits last dimension into two halves [x1 | x2] and returns [-x2 | x1].
    This is the standard "rotate_half" used in the real-valued RoPE formulation.

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


def apply_rotary_emb_real(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using the real-valued (cos/sin) formulation.

    Rotation formula:
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

    Args:
        q: Query tensor of shape (..., d_head).
        k: Key tensor of shape (..., d_head).
        cos: Cosine tensor of shape (T, d_head). Must match sequence dim.
        sin: Sine tensor of shape (T, d_head). Must match sequence dim.

    Returns:
        Tuple of (q_rotated, k_rotated), each with same shape as input.
    """
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings.

    Projects input to Q, K, V, applies RoPE to Q and K (not V),
    then computes causal scaled dot-product attention manually.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        config: RoPEConfig for frequency computation.
    """

    def __init__(self, d_model: int, n_heads: int, config: RoPEConfig) -> None:
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

        # Pre-compute and register frequency cache
        freqs = compute_freqs_cis(
            d_head=self.d_head,
            max_seq_len=config.max_seq_len,
            base=config.base,
            scale=config.scale,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute RoPE attention.

        Args:
            x: Input tensor of shape (B, T, d_model).
            freqs_cis: Optional pre-computed complex frequencies of shape
                       (max_seq_len, d_head // 2). If None, uses cached freqs.

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape

        if freqs_cis is None:
            freqs_cis = self.freqs_cis  # type: ignore[assignment]

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, T, n_heads, d_head) for RoPE application
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        # Apply RoPE to Q and K (not V)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Rearrange to (B, n_heads, T, d_head) for attention computation
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (manual, causal)
        scale = math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # Causal mask: upper triangle (future positions) set to -inf
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attn_scores = attn_scores + causal_mask

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, T, T)

        # Weighted sum of values
        attn_out = torch.matmul(attn_weights, v)  # (B, n_heads, T, d_head)

        # Merge heads: (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(attn_out)
