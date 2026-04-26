"""
src/model/xpos_attention.py

XPos attention with extrapolatable positional embeddings.
Extends RoPE with learnable per-head exponential decay that allows
length extrapolation without retraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class XPosConfig:
    d_model: int
    n_heads: int
    head_dim: int
    base_theta: float = 10000.0
    scale_base: int = 512
    use_xpos: bool = True

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (  # noqa: S101
            f"d_model ({self.d_model}) must equal n_heads * head_dim "
            f"({self.n_heads} * {self.head_dim})"
        )


def _build_xpos_scale(
    seq_len: int,
    head_dim: int,
    scale_base: int,
    device: torch.device,
) -> Tensor:
    """Build per-position xpos scale factors.

    Returns:
        Float tensor of shape (seq_len, head_dim).
        Even indices hold scale, odd indices hold 1/scale.
    """
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    # Scalar scale per position
    scale = (positions + scale_base / 2.0) / scale_base  # (seq_len,)

    # Interleave [scale, 1/scale] along head_dim
    half = head_dim // 2
    scale_expanded = scale.unsqueeze(1).expand(seq_len, half)  # (seq_len, half)
    inv_scale = 1.0 / scale_expanded

    # Stack and reshape: (seq_len, half, 2) -> (seq_len, head_dim)
    result = torch.stack([scale_expanded, inv_scale], dim=-1)  # (seq_len, half, 2)
    return result.reshape(seq_len, head_dim)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the second half of the last dimension by negating and swapping."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _build_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build standard RoPE cosine/sine tables.

    Returns:
        cos, sin each of shape (seq_len, head_dim).
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    # (seq_len, half)
    freqs = torch.outer(positions, inv_freq)
    # Repeat to cover full head_dim
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return freqs.cos(), freqs.sin()


def apply_xpos_rope(
    q: Tensor,
    k: Tensor,
    seq_len: int,
    theta: float,
    scale_base: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE with XPos scale to query and key tensors.

    Args:
        q: Query tensor of shape (B, n_heads, S, head_dim).
        k: Key tensor of shape (B, n_kv_heads, S, head_dim).
        seq_len: Sequence length.
        theta: RoPE base theta.
        scale_base: XPos scale base.
        device: Computation device.

    Returns:
        (q_rot, k_rot) with RoPE + xpos applied.
        q is multiplied by xpos scale, k by 1/xpos scale.
    """
    head_dim = q.shape[-1]

    cos, sin = _build_rope_cos_sin(seq_len, head_dim, theta, device)
    # (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Standard RoPE
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin

    # XPos scale
    xscale = _build_xpos_scale(seq_len, head_dim, scale_base, device)
    # (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    xscale = xscale.unsqueeze(0).unsqueeze(0)

    q_rot = q_rot * xscale
    k_rot = k_rot * (1.0 / xscale)

    return q_rot, k_rot


class XPosAttention(nn.Module):
    """Grouped-Query Attention with XPos positional embeddings.

    Args:
        config: XPosConfig with model dimensions and XPos parameters.
        n_kv_heads: Number of key/value heads. Defaults to n_heads (full MHA).
    """

    def __init__(self, config: XPosConfig, n_kv_heads: int | None = None) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else config.n_heads
        self.base_theta = config.base_theta
        self.scale_base = config.scale_base
        self.use_xpos = config.use_xpos

        assert self.n_heads % self.n_kv_heads == 0, (  # noqa: S101
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.kv_groups = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, S, d_model).

        Returns:
            Output tensor of shape (B, S, d_model).
        """
        B, S, _ = x.shape
        device = x.device

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_xpos:
            q, k = apply_xpos_rope(q, k, S, self.base_theta, self.scale_base, device)
        else:
            # Standard RoPE fallback
            cos, sin = _build_rope_cos_sin(S, self.head_dim, self.base_theta, device)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

        # Expand kv heads for GQA
        if self.kv_groups > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.kv_groups, S, self.head_dim)
            k = k.reshape(B, self.n_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.kv_groups, S, self.head_dim)
            v = v.reshape(B, self.n_heads, S, self.head_dim)

        # Causal mask
        causal_mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
        attn_bias = torch.zeros(S, S, device=device, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale + attn_bias
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)  # (B, n_heads, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.out_proj(out)
