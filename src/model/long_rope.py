"""LongRoPE: dynamic position encoding extrapolation for long-context inference (Ding et al., 2024)."""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LongRoPEConfig:
    head_dim: int = 32
    base_theta: float = 10000.0
    max_train_len: int = 4096  # training context length
    target_len: int = 32768  # desired inference length
    n_rescale_factors: int = 16  # per-dimension rescale (LongRoPE uses different scale per dim)
    lambda_min: float = 1.0  # minimum rescale factor
    lambda_max: float = 8.0  # maximum rescale factor


def compute_longrope_freqs(config: LongRoPEConfig) -> torch.Tensor:
    """Compute LongRoPE non-uniform position rescaling frequencies.

    Standard RoPE: theta_i = base^(-2i/d) for i in 0..d/2
    LongRoPE: for each dimension i, compute rescale factor lambda_i:
      - Linearly interpolate from lambda_min to lambda_max based on dimension index
      - Higher-frequency dimensions (small i) get smaller lambda (less stretch)
      - Lower-frequency dimensions (large i) get larger lambda (more stretch)
    Effective frequency: freq_i = theta_i / lambda_i

    Returns:
        Tensor of shape (head_dim // 2,) — the per-dimension frequencies.
    """
    half_dim = config.head_dim // 2

    # Dimension indices: 0, 1, ..., half_dim - 1
    i = torch.arange(0, half_dim, dtype=torch.float32)

    # Base RoPE frequencies: base^(-2i/d)
    base_freqs = 1.0 / (config.base_theta ** (2 * i / config.head_dim))

    # Linear interpolation of lambda from lambda_min (i=0) to lambda_max (i=half_dim-1)
    # Higher-frequency dims (small i) -> smaller lambda (less stretch)
    # Lower-frequency dims  (large i) -> larger lambda (more stretch)
    if half_dim == 1:
        t = torch.zeros(1)
    else:
        t = i / (half_dim - 1)  # 0 -> 1

    lambdas = config.lambda_min + t * (config.lambda_max - config.lambda_min)

    # Effective frequency per dimension
    freqs = base_freqs / lambdas

    return freqs  # shape (head_dim // 2,)


def build_longrope_cos_sin(
    seq_len: int,
    config: LongRoPEConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build position-dependent cos/sin tables for long sequences.

    Args:
        seq_len: Sequence length T.
        config: LongRoPEConfig.

    Returns:
        (cos, sin) each of shape (T, head_dim).
    """
    freqs = compute_longrope_freqs(config)  # (D/2,)
    positions = torch.arange(seq_len, dtype=torch.float32)  # (T,)

    # Outer product: (T, D/2)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

    # Duplicate to full head_dim: (T, D)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles), torch.sin(angles)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half, negate second half, swap: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_longrope_rotation(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to x.

    Args:
        x:   Shape (B, n_heads, T, D).
        cos: Shape (T, D) — broadcast over B and n_heads.
        sin: Shape (T, D) — broadcast over B and n_heads.

    Returns:
        Rotated tensor of same shape as x.
    """
    # Broadcast cos/sin over batch and heads: (1, 1, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class LongRoPEAttention(nn.Module):
    """Attention with LongRoPE position encodings."""

    def __init__(self, d_model: int, n_heads: int, config: LongRoPEConfig) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = config.head_dim
        self.config = config

        self.q_proj = nn.Linear(d_model, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, d_model, bias=False)

        # Pre-compute cos/sin tables for max_train_len and target_len
        cos_train, sin_train = build_longrope_cos_sin(config.max_train_len, config)
        cos_target, sin_target = build_longrope_cos_sin(config.target_len, config)

        self.register_buffer("cos_train", cos_train)  # (max_train_len, head_dim)
        self.register_buffer("sin_train", sin_train)
        self.register_buffer("cos_target", cos_target)  # (target_len, head_dim)
        self.register_buffer("sin_target", sin_target)

    def forward(
        self,
        x: torch.Tensor,
        seq_len_override: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Shape (B, T, D).
            seq_len_override: Optional override for sequence length selection.

        Returns:
            Output tensor of shape (B, T, D).
        """
        B, T, _ = x.shape
        effective_len = seq_len_override if seq_len_override is not None else T

        # Select appropriate cos/sin table
        use_len = min(effective_len, self.config.target_len)

        if use_len <= self.config.max_train_len:
            cos = self.cos_train[:T]
            sin = self.sin_train[:T]
        else:
            cos = self.cos_target[:T]
            sin = self.sin_target[:T]

        # Project
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply LongRoPE rotation to Q and K
        q = apply_longrope_rotation(q, cos, sin)
        k = apply_longrope_rotation(k, cos, sin)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape: (B, n_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


def extrapolation_quality(
    model_freqs: torch.Tensor,
    longrope_freqs: torch.Tensor,
) -> dict[str, float]:
    """Compare standard RoPE vs LongRoPE frequencies.

    Args:
        model_freqs:    Standard RoPE frequencies, shape (D/2,).
        longrope_freqs: LongRoPE frequencies, shape (D/2,).

    Returns:
        Dict with keys: mean_rescale, max_rescale, freq_spread.
    """
    # Rescale factors: how much each dimension was stretched
    rescale = model_freqs / longrope_freqs.clamp(min=1e-12)

    mean_rescale = rescale.mean().item()
    max_rescale = rescale.max().item()
    freq_spread = (longrope_freqs.max() - longrope_freqs.min()).item()

    return {
        "mean_rescale": mean_rescale,
        "max_rescale": max_rescale,
        "freq_spread": freq_spread,
    }
