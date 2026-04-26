"""Position interpolation for extending RoPE context windows.

Implements linear position interpolation, NTK-aware scaling, and YaRN for
extending RoPE-based models to longer contexts at inference time.

References:
    Chen et al. 2023, "Extending Context Window of Large Language Models via
    Positional Interpolation"
    Peng et al. 2023, "YaRN: Efficient Context Window Extension of Large Language Models"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PIConfig:
    """Configuration for position interpolation context extension."""

    d_head: int = 64
    original_max_seq_len: int = 2048
    extended_max_seq_len: int = 8192
    base: float = 10000.0
    method: str = "linear"  # "linear", "ntk", or "yarn"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _standard_freqs_cis(
    d_head: int,
    seq_len: int,
    base: float = 10000.0,
    positions: Tensor | None = None,
) -> Tensor:
    """Compute complex RoPE frequencies.

    Args:
        d_head: Head dimension (must be even).
        seq_len: Sequence length.
        base: RoPE base frequency.
        positions: Optional (seq_len,) position tensor; defaults to 0..seq_len-1.

    Returns:
        Complex64 tensor of shape (seq_len, d_head // 2).
    """
    d_head // 2
    # theta_i = 1 / base^(2i/d_head)
    theta = 1.0 / (
        base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
    )  # (half_dim,)

    if positions is None:
        positions = torch.arange(seq_len, dtype=torch.float32)

    freqs = torch.outer(positions, theta)  # (seq_len, half_dim)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


# ---------------------------------------------------------------------------
# Scale factor computation
# ---------------------------------------------------------------------------


def compute_scale_factor(config: PIConfig) -> float:
    """Compute the position scaling factor for the given interpolation method.

    For linear: scale = extended / original.
    For NTK/YaRN: scale = (extended / original)^(d_head/(d_head-2)).

    Returns:
        Scale factor >= 1.0.
    """
    ratio = config.extended_max_seq_len / config.original_max_seq_len
    if config.method == "linear":
        return ratio
    else:
        # NTK and YaRN use the same scale formula
        exponent = config.d_head / (config.d_head - 2)
        return ratio**exponent


# ---------------------------------------------------------------------------
# Interpolation functions
# ---------------------------------------------------------------------------


def interpolate_freqs_cis(
    freqs_cis: Tensor,
    target_seq_len: int,
    scale_factor: float,
) -> Tensor:
    """Extend freqs_cis to target_seq_len by dividing positions by scale_factor.

    Args:
        freqs_cis: (original_seq_len, d_head//2) complex tensor.
        target_seq_len: Desired output sequence length.
        scale_factor: Position scaling factor (>= 1.0).

    Returns:
        Complex64 tensor of shape (target_seq_len, d_head//2).
    """
    half_dim = freqs_cis.shape[1]
    half_dim * 2

    # Recover per-dimension theta from existing freqs_cis using a single position
    # theta_i = angle(freqs_cis[1, i]) / 1  (position 1 gives theta directly)
    angles_at_1 = freqs_cis[1].angle() if freqs_cis.shape[0] > 1 else freqs_cis[0].angle()
    theta = angles_at_1  # (half_dim,)

    # New positions: 0, 1/scale, 2/scale, ..., (T-1)/scale
    positions = torch.arange(target_seq_len, dtype=torch.float32) / scale_factor
    freqs = torch.outer(positions, theta)
    return torch.polar(torch.ones_like(freqs), freqs)


def ntk_aware_freqs_cis(
    d_head: int,
    seq_len: int,
    base: float,
    scale: float,
) -> Tensor:
    """NTK-aware interpolation: modifies the base frequency.

    NTK scales the base as: base_new = base * scale^(d_head/(d_head-2))
    then computes standard freqs_cis with the new base.

    Args:
        d_head: Head dimension.
        seq_len: Target sequence length.
        base: Original RoPE base.
        scale: Scale factor (>= 1.0).

    Returns:
        Complex64 tensor of shape (seq_len, d_head//2).
    """
    exponent = d_head / (d_head - 2)
    new_base = base * (scale**exponent)
    return _standard_freqs_cis(d_head, seq_len, base=new_base)


def yarn_freqs_cis(
    d_head: int,
    seq_len: int,
    base: float,
    scale: float,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Tensor:
    """YaRN interpolation: blends interpolated and extrapolated frequencies.

    High-frequency dimensions (short wavelength) use interpolation (divide by scale).
    Low-frequency dimensions (long wavelength) use extrapolation (no scaling).
    Mid-range is blended.

    Args:
        d_head: Head dimension.
        seq_len: Target sequence length.
        base: Original RoPE base.
        scale: Scale factor.
        beta_fast: Wavelength ratio threshold for high-freq boundary.
        beta_slow: Wavelength ratio threshold for low-freq boundary.

    Returns:
        Complex64 tensor of shape (seq_len, d_head//2).
    """
    d_head // 2

    # Per-dimension theta (original)
    dim_idx = torch.arange(0, d_head, 2, dtype=torch.float32)
    theta_orig = 1.0 / (base ** (dim_idx / d_head))  # (half_dim,)

    # Wavelength for each dim: lambda = 2*pi / theta
    wavelengths = 2.0 * math.pi / theta_orig  # (half_dim,)

    # Blend factor: low wavelength (high freq) → interpolate (alpha=1)
    #               high wavelength (low freq) → extrapolate (alpha=0)
    alpha = torch.clamp(
        (wavelengths / seq_len - 1.0 / beta_fast) / (1.0 / beta_slow - 1.0 / beta_fast),
        0.0,
        1.0,
    )  # (half_dim,)

    # Interpolated theta: theta / scale
    theta_interp = theta_orig / scale
    # Extrapolated theta: theta unchanged
    theta_extrap = theta_orig

    # Blended theta
    theta_yarn = alpha * theta_interp + (1.0 - alpha) * theta_extrap  # (half_dim,)

    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta_yarn)
    return torch.polar(torch.ones_like(freqs), freqs)


def build_interpolated_freqs_cis(config: PIConfig) -> Tensor:
    """Build freqs_cis for extended_max_seq_len using the configured method.

    Returns:
        Complex64 tensor of shape (extended_max_seq_len, d_head//2).
    """
    scale = compute_scale_factor(config)

    if config.method == "linear":
        positions = torch.arange(config.extended_max_seq_len, dtype=torch.float32) / scale
        theta = 1.0 / (
            config.base ** (torch.arange(0, config.d_head, 2, dtype=torch.float32) / config.d_head)
        )
        freqs = torch.outer(positions, theta)
        return torch.polar(torch.ones_like(freqs), freqs)
    elif config.method == "ntk":
        return ntk_aware_freqs_cis(config.d_head, config.extended_max_seq_len, config.base, scale)
    elif config.method == "yarn":
        return yarn_freqs_cis(config.d_head, config.extended_max_seq_len, config.base, scale)
    else:
        raise ValueError(f"Unknown method: '{config.method}'")


# ---------------------------------------------------------------------------
# PositionInterpolator
# ---------------------------------------------------------------------------


class PositionInterpolator:
    """High-level interface for position interpolation.

    Caches computed freqs_cis to avoid redundant computation.

    Args:
        config: PIConfig controlling the interpolation method and dimensions.
    """

    def __init__(self, config: PIConfig) -> None:
        self.config = config
        self._cache: dict[int, Tensor] = {}

    def get_freqs_cis(self, seq_len: int) -> Tensor:
        """Return freqs_cis for the given sequence length.

        Uses cached result if available.

        Args:
            seq_len: Target sequence length.

        Returns:
            Complex64 tensor of shape (seq_len, d_head//2).
        """
        if seq_len in self._cache:
            return self._cache[seq_len]

        cfg = self.config
        if seq_len <= cfg.original_max_seq_len:
            result = _standard_freqs_cis(cfg.d_head, seq_len, cfg.base)
        else:
            scale = compute_scale_factor(cfg)
            if cfg.method == "linear":
                positions = torch.arange(seq_len, dtype=torch.float32) / scale
                theta = 1.0 / (
                    cfg.base ** (torch.arange(0, cfg.d_head, 2, dtype=torch.float32) / cfg.d_head)
                )
                freqs = torch.outer(positions, theta)
                result = torch.polar(torch.ones_like(freqs), freqs)
            elif cfg.method == "ntk":
                result = ntk_aware_freqs_cis(cfg.d_head, seq_len, cfg.base, scale)
            elif cfg.method == "yarn":
                result = yarn_freqs_cis(cfg.d_head, seq_len, cfg.base, scale)
            else:
                raise ValueError(f"Unknown method: '{cfg.method}'")

        self._cache[seq_len] = result
        return result

    def extend_context(self, original_freqs_cis: Tensor, new_max_seq_len: int) -> Tensor:
        """Extend existing freqs_cis to a longer context.

        Args:
            original_freqs_cis: (orig_len, d_head//2) complex tensor.
            new_max_seq_len: Target sequence length.

        Returns:
            Complex64 tensor of shape (new_max_seq_len, d_head//2).
        """
        scale = compute_scale_factor(self.config)
        return interpolate_freqs_cis(original_freqs_cis, new_max_seq_len, scale)
