"""Position interpolation: extend RoPE context length at inference time.

Implements linear position interpolation and NTK-aware scaling for extending
RoPE to longer contexts at inference time, complementary to YaRN (yarn_rope.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PositionInterpolationConfig:
    """Configuration for position interpolation context extension."""

    original_max_len: int = 8192
    target_max_len: int = 32768
    method: str = "linear"  # "linear" | "ntk" | "dynamic_ntk"
    ntk_alpha: float = 1.0  # NTK scaling factor


def linear_interpolate_positions(
    seq_len: int,
    original_max_len: int,
    target_max_len: int,
) -> Tensor:
    """Scale positions so they fit within original_max_len.

    interpolated_pos[i] = i * (original_max_len / target_max_len)

    Args:
        seq_len: Number of positions to generate.
        original_max_len: Original training context length.
        target_max_len: Target extended context length.

    Returns:
        (seq_len,) float tensor of interpolated positions.
    """
    scale = original_max_len / target_max_len
    positions = torch.arange(seq_len, dtype=torch.float32) * scale
    return positions


def ntk_rope_base(
    base: float,
    head_dim: int,
    original_max_len: int,
    target_max_len: int,
    alpha: float = 1.0,
) -> float:
    """Compute NTK-aware RoPE base frequency.

    new_base = base * (alpha * target_max_len / original_max_len) ^ (head_dim / (head_dim - 2))

    Args:
        base: Original RoPE base frequency (e.g., 10000.0).
        head_dim: Attention head dimension.
        original_max_len: Original training context length.
        target_max_len: Target extended context length.
        alpha: NTK scaling factor.

    Returns:
        New base frequency as float.
    """
    scale_ratio = alpha * target_max_len / original_max_len
    exponent = head_dim / (head_dim - 2)
    return base * (scale_ratio**exponent)


def build_rope_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    positions: Tensor | None = None,
) -> Tensor:
    """Build RoPE frequency matrix.

    theta_i = base^(-2i/head_dim) for i in [0, head_dim/2)
    freqs = outer(positions, theta)  — (seq_len, head_dim/2)

    Returns a real tensor of shape (seq_len, head_dim) where
    element [t, 2k] = cos(positions[t] * theta_k) and
    element [t, 2k+1] = sin(positions[t] * theta_k).

    Args:
        head_dim: Attention head dimension (must be even).
        seq_len: Number of positions.
        base: RoPE base frequency.
        positions: Optional custom positions of shape (seq_len,). If None,
                   uses 0, 1, 2, ..., seq_len-1.

    Returns:
        Real tensor of shape (seq_len, head_dim).
    """
    half_dim = head_dim // 2
    # theta_i = base^(-2i/head_dim)
    i = torch.arange(0, half_dim, dtype=torch.float32)
    theta = base ** (-2.0 * i / head_dim)  # (half_dim,)

    if positions is None:
        positions = torch.arange(seq_len, dtype=torch.float32)

    # angles: (seq_len, half_dim)
    angles = torch.outer(positions, theta)

    cos_vals = angles.cos()  # (seq_len, half_dim)
    sin_vals = angles.sin()  # (seq_len, half_dim)

    # Interleave cos/sin into (seq_len, head_dim)
    freqs = torch.zeros(seq_len, head_dim, dtype=torch.float32)
    freqs[:, 0::2] = cos_vals
    freqs[:, 1::2] = sin_vals

    return freqs


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary position embeddings to x.

    Args:
        x: (B, H, T, head_dim)
        freqs: (T, head_dim) real tensor where [:, 2k] = cos, [:, 2k+1] = sin

    Returns:
        (B, H, T, head_dim) with RoPE applied.
    """
    B, H, T, D = x.shape
    half_dim = D // 2

    # Extract cos and sin from interleaved freqs
    cos = freqs[:, 0::2]  # (T, half_dim)
    sin = freqs[:, 1::2]  # (T, half_dim)

    # Reshape for broadcasting: (1, 1, T, half_dim)
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)

    # Split x into first and second halves along head_dim
    x1 = x[..., :half_dim]  # (B, H, T, half_dim)
    x2 = x[..., half_dim:]  # (B, H, T, half_dim)

    # 2D rotation: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


class ContextLengthExtender:
    """Extends a model's context length via position interpolation.

    Patches model.freqs_cis with an interpolated version for the target
    context length, then restores the original after use.
    """

    def __init__(self, model, config: PositionInterpolationConfig) -> None:
        self.model = model
        self.config = config
        self._original_freqs = None

    def patch_rope(self) -> None:
        """Replace model.freqs_cis with interpolated version for target_max_len."""
        # Save original
        self._original_freqs = self.model.freqs_cis

        cfg = self.config
        head_dim = self._original_freqs.shape[-1]
        # freqs_cis is complex: shape (seq_len, head_dim//2)
        # We need to figure out real head_dim from complex tensor
        if self._original_freqs.is_complex():
            real_head_dim = head_dim * 2
        else:
            real_head_dim = head_dim

        target_len = cfg.target_max_len

        if cfg.method == "linear":
            positions = linear_interpolate_positions(
                target_len, cfg.original_max_len, cfg.target_max_len
            )
            new_freqs = self._build_complex_freqs(real_head_dim, target_len, positions=positions)
        elif cfg.method == "ntk":
            new_base = ntk_rope_base(
                10000.0, real_head_dim, cfg.original_max_len, cfg.target_max_len, cfg.ntk_alpha
            )
            new_freqs = self._build_complex_freqs(real_head_dim, target_len, base=new_base)
        elif cfg.method == "dynamic_ntk":
            scale = dynamic_ntk_scale(target_len, cfg.original_max_len, real_head_dim)
            new_base = 10000.0 * scale
            new_freqs = self._build_complex_freqs(real_head_dim, target_len, base=new_base)
        else:
            raise ValueError(f"Unknown method: {cfg.method!r}")

        # Move to same device as model
        new_freqs = new_freqs.to(self._original_freqs.device)
        self.model.freqs_cis = new_freqs

    def _build_complex_freqs(
        self,
        head_dim: int,
        seq_len: int,
        base: float = 10000.0,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Build complex freqs_cis tensor matching model's expected format."""
        half_dim = head_dim // 2
        i = torch.arange(0, half_dim, dtype=torch.float32)
        theta = base ** (-2.0 * i / head_dim)

        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.float32)

        angles = torch.outer(positions, theta)  # (seq_len, half_dim)
        return torch.polar(torch.ones_like(angles), angles)  # complex64

    def restore_rope(self) -> None:
        """Restore original freqs_cis."""
        if self._original_freqs is not None:
            self.model.freqs_cis = self._original_freqs
            self._original_freqs = None

    def generate_extended(self, input_ids: Tensor) -> Tensor:
        """Generate with patched RoPE for extended context.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            logits (B, T, vocab_size).
        """
        self.patch_rope()
        try:
            with torch.no_grad():
                loss, logits, past_key_values = self.model(input_ids)
        finally:
            self.restore_rope()
        return logits


def dynamic_ntk_scale(
    seq_len: int,
    original_max_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> float:
    """Dynamic NTK scaling factor based on current sequence length.

    scale = max(1.0, (seq_len / original_max_len)^(head_dim/(head_dim-2)))

    Args:
        seq_len: Current sequence length.
        original_max_len: Original training context length.
        head_dim: Attention head dimension.
        base: RoPE base frequency (unused, kept for API consistency).

    Returns:
        Scaling factor as float (>= 1.0).
    """
    ratio = seq_len / original_max_len
    exponent = head_dim / (head_dim - 2)
    return max(1.0, ratio**exponent)
