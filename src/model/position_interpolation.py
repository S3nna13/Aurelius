"""Position interpolation techniques for extending RoPE context length.

Implements linear, NTK-aware, YaRN, and dynamic NTK position scaling,
distinct from the NTK-only approach in ntk_rope.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class PIConfig:
    """Configuration for position interpolation methods."""

    method: str = "linear"          # "linear" | "ntk" | "yarn" | "dynamic"
    scale_factor: float = 4.0       # context extension factor
    base_theta: float = 10000.0     # RoPE base frequency
    original_max_len: int = 2048    # original training context length
    extended_max_len: int = 8192    # target extended context length


def compute_rope_freqs(head_dim: int, theta: float = 10000.0) -> Tensor:
    """Compute standard RoPE frequencies.

    Formula: freq_i = 1 / theta^(2i / D) for i in range(D // 2).

    Args:
        head_dim: Head dimension D (must be even).
        theta:    Base frequency (default 10000.0).

    Returns:
        Frequency tensor of shape (D // 2,).
    """
    half_dim = head_dim // 2
    i = torch.arange(0, half_dim, dtype=torch.float32)
    freqs = 1.0 / (theta ** (2.0 * i / head_dim))
    return freqs  # (D//2,)


def linear_position_interpolation(freqs: Tensor, scale_factor: float) -> Tensor:
    """Linear position interpolation: scale down frequencies to extend context.

    Equivalent to dividing positional indices by scale_factor, which maps
    extended positions into the original training range.

    Args:
        freqs:        Standard RoPE frequencies, shape (D//2,).
        scale_factor: Context extension ratio (e.g. 4.0 for 4x context).

    Returns:
        Scaled frequencies of the same shape (D//2,).
    """
    return freqs / scale_factor


def ntk_aware_interpolation(
    head_dim: int,
    scale_factor: float,
    base_theta: float,
) -> Tensor:
    """NTK-aware position interpolation via scaled base theta.

    Scales the RoPE base: theta' = theta * scale_factor^(D / (D - 2)).
    This avoids high-frequency collapse by redistributing the extension
    across all frequency dimensions.

    Args:
        head_dim:     Head dimension D (must be even).
        scale_factor: Context extension ratio.
        base_theta:   Original RoPE base frequency.

    Returns:
        Frequency tensor of shape (D//2,).
    """
    scaled_theta = base_theta * (scale_factor ** (head_dim / (head_dim - 2)))
    return compute_rope_freqs(head_dim, theta=scaled_theta)


def yarn_interpolation(
    head_dim: int,
    scale_factor: float,
    base_theta: float,
    alpha: float = 1.0,
    beta: float = 32.0,
    original_max_len: int = 2048,
) -> Tensor:
    """YaRN interpolation: blend linear and NTK per frequency dimension.

    For each frequency component i, the wavelength determines the blend:
      - wavelength > original_max_len / beta → low-frequency: use NTK (no scaling)
      - wavelength < original_max_len / alpha → high-frequency: use linear (scale down)
      - otherwise: smooth interpolation between the two

    Args:
        head_dim:         Head dimension D (must be even).
        scale_factor:     Context extension ratio.
        base_theta:       Original RoPE base frequency.
        alpha:            High-frequency threshold parameter.
        beta:             Low-frequency threshold parameter.
        original_max_len: Original maximum sequence length.

    Returns:
        Blended frequency tensor of shape (D//2,).
    """
    standard_freqs = compute_rope_freqs(head_dim, theta=base_theta)
    linear_freqs = linear_position_interpolation(standard_freqs, scale_factor)
    ntk_freqs = ntk_aware_interpolation(head_dim, scale_factor, base_theta)

    half_dim = head_dim // 2
    blended = torch.empty(half_dim, dtype=torch.float32)

    low_thresh = original_max_len / beta   # wavelength above this → use NTK (original)
    high_thresh = original_max_len / alpha  # wavelength below this → use linear

    for idx in range(half_dim):
        freq = standard_freqs[idx].item()
        # wavelength = 2π / freq  (freq > 0 always)
        wavelength = (2.0 * math.pi) / freq if freq != 0.0 else float("inf")

        if wavelength > low_thresh:
            # Low frequency: preserve original NTK freq (no scaling)
            blended[idx] = ntk_freqs[idx]
        elif wavelength < high_thresh:
            # High frequency: apply linear scaling
            blended[idx] = linear_freqs[idx]
        else:
            # Middle: smooth linear interpolation between linear and NTK
            # t=0 → low_thresh (NTK), t=1 → high_thresh (linear)
            t = (low_thresh - wavelength) / (low_thresh - high_thresh + 1e-8)
            t = max(0.0, min(1.0, t))
            blended[idx] = (1.0 - t) * ntk_freqs[idx] + t * linear_freqs[idx]

    return blended


def dynamic_ntk_interpolation(
    head_dim: int,
    seq_len: int,
    original_max_len: int,
    base_theta: float,
) -> Tensor:
    """Dynamic NTK interpolation: compute scale_factor at runtime from seq_len.

    scale_factor = max(1, seq_len / original_max_len).
    When seq_len <= original_max_len, no scaling is applied (standard RoPE).

    Args:
        head_dim:         Head dimension D (must be even).
        seq_len:          Current sequence length.
        original_max_len: Original training context length.
        base_theta:       Original RoPE base frequency.

    Returns:
        Frequency tensor of shape (D//2,).
    """
    scale_factor = max(1.0, seq_len / original_max_len)
    return ntk_aware_interpolation(head_dim, scale_factor, base_theta)


def apply_rope_with_freqs(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply RoPE rotations using precomputed frequencies.

    For each position t and each frequency pair, rotates x by angle t * freq.

    Args:
        x:     Input tensor of shape (B, H, T, D).
        freqs: Frequency tensor of shape (D//2,).

    Returns:
        Rotated tensor of the same shape (B, H, T, D).
    """
    B, H, T, D = x.shape
    half_dim = D // 2

    # Build angles: (T, D//2), then cos/sin
    positions = torch.arange(T, device=x.device, dtype=x.dtype)
    angles = torch.outer(positions, freqs.to(x.device, x.dtype))  # (T, D//2)

    cos = angles.cos()  # (T, D//2)
    sin = angles.sin()  # (T, D//2)

    # Reshape for broadcasting: (1, 1, T, D//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split into pairs
    x1 = x[..., :half_dim]   # (B, H, T, D//2)
    x2 = x[..., half_dim:]   # (B, H, T, D//2)

    # Rotate: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


class PositionInterpolator:
    """Applies position interpolation to attention head tensors.

    Supports linear, NTK, YaRN, and dynamic NTK methods as configured by PIConfig.
    """

    def __init__(self, config: PIConfig, head_dim: int) -> None:
        self.config = config
        self.head_dim = head_dim

    def get_freqs(self, seq_len: int) -> Tensor:
        """Return interpolated frequencies for the given sequence length.

        Args:
            seq_len: Current sequence length.

        Returns:
            Frequency tensor of shape (head_dim // 2,).
        """
        cfg = self.config
        method = cfg.method

        if method == "linear":
            base_freqs = compute_rope_freqs(self.head_dim, theta=cfg.base_theta)
            return linear_position_interpolation(base_freqs, cfg.scale_factor)
        elif method == "ntk":
            return ntk_aware_interpolation(self.head_dim, cfg.scale_factor, cfg.base_theta)
        elif method == "yarn":
            return yarn_interpolation(
                self.head_dim,
                cfg.scale_factor,
                cfg.base_theta,
                original_max_len=cfg.original_max_len,
            )
        elif method == "dynamic":
            return dynamic_ntk_interpolation(
                self.head_dim, seq_len, cfg.original_max_len, cfg.base_theta
            )
        else:
            raise ValueError(f"Unknown position interpolation method: {method!r}")

    def apply(self, x: Tensor) -> Tensor:
        """Apply position encoding to input tensor.

        Args:
            x: Input tensor of shape (B, H, T, D).

        Returns:
            Position-encoded tensor of the same shape.
        """
        _B, _H, T, _D = x.shape
        freqs = self.get_freqs(T)
        return apply_rope_with_freqs(x, freqs)
