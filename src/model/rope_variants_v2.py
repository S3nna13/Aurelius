"""RoPE variants: standard, linear scaled, NTK-aware, and YaRN rotary position embeddings.

Covers alternative RoPE formulas for long-context extension:
  - standard:       original RoPE (Su et al., 2021)
  - linear_scaled:  divide freqs by scale_factor (Chen et al., 2023)
  - dynamic_ntk:    NTK-aware scaling (bloc97, 2023)
  - yarn:           YaRN interpolation (Peng et al., 2023)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

_VALID_ROPE_TYPES = {"standard", "linear_scaled", "dynamic_ntk", "yarn"}


@dataclass
class RoPEConfig:
    """Configuration for RoPE variant selection and parameters."""

    d_head: int = 64
    base: float = 10000.0
    max_seq_len: int = 2048
    rope_type: str = "standard"  # one of _VALID_ROPE_TYPES
    scale_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    def __post_init__(self) -> None:
        if self.rope_type not in _VALID_ROPE_TYPES:
            raise ValueError(
                f"rope_type must be one of {_VALID_ROPE_TYPES}, got '{self.rope_type}'"
            )
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {self.d_head}")


# ---------------------------------------------------------------------------
# Frequency computation functions
# ---------------------------------------------------------------------------


def compute_standard_freqs(d_head: int, base: float = 10000.0) -> Tensor:
    """Compute standard RoPE inverse frequencies.

    freqs[i] = 1 / base^(2i / d_head)  for i in 0 .. d_head//2 - 1

    Args:
        d_head: Head dimension (must be even).
        base:   Base for the geometric sequence (default 10000).

    Returns:
        Tensor of shape (d_head//2,) with positive, decreasing frequencies.
    """
    half = d_head // 2
    i = torch.arange(0, half, dtype=torch.float32)
    freqs = 1.0 / (base ** (2.0 * i / d_head))
    return freqs


def compute_linear_scaled_freqs(d_head: int, base: float, scale_factor: float) -> Tensor:
    """Compute linearly scaled RoPE frequencies for context extension.

    Divides standard frequencies by scale_factor, effectively compressing the
    position space so that positions beyond the original training length remain
    in-distribution.

    Args:
        d_head:       Head dimension (must be even).
        base:         RoPE base frequency.
        scale_factor: Factor by which to divide frequencies (> 1 extends context).

    Returns:
        Tensor of shape (d_head//2,).
    """
    freqs = compute_standard_freqs(d_head, base)
    return freqs / scale_factor


def compute_ntk_freqs(d_head: int, base: float, scale_factor: float, max_seq_len: int) -> Tensor:
    """Compute NTK-aware scaled RoPE frequencies.

    Rescales the base: new_base = base * scale_factor^(d_head / (d_head - 2)),
    then recomputes standard frequencies with new_base.

    Args:
        d_head:       Head dimension (must be even).
        base:         Original RoPE base.
        scale_factor: Scaling factor (> 1 extends context via NTK adjustment).
        max_seq_len:  Target max sequence length (used to inform scale_factor).

    Returns:
        Tensor of shape (d_head//2,).
    """
    new_base = base * (scale_factor ** (d_head / (d_head - 2)))
    return compute_standard_freqs(d_head, new_base)


def compute_yarn_freqs(
    d_head: int,
    base: float,
    scale_factor: float,
    beta_fast: float,
    beta_slow: float,
) -> Tensor:
    """Compute YaRN (Yet another RoPE extensioN) frequencies.

    Interpolates per frequency dimension between linear scaling (high-frequency
    dims) and NTK scaling (low-frequency dims) based on wavelength thresholds
    determined by beta_fast and beta_slow.

    For each dimension i:
      wavelength = 2 * pi / freq_i
      if wavelength < 2*pi / beta_fast  -> use NTK (no interpolation for fast dims)
      if wavelength > 2*pi / beta_slow  -> use linear scaling
      else                              -> linear blend

    Args:
        d_head:       Head dimension (must be even).
        base:         Original RoPE base.
        scale_factor: Context extension scale factor.
        beta_fast:    Fast frequency threshold (wavelength cutoff, smaller = more NTK).
        beta_slow:    Slow frequency threshold (wavelength cutoff, larger = more linear).

    Returns:
        Tensor of shape (d_head//2,).
    """
    d_head // 2

    freqs_std = compute_standard_freqs(d_head, base)  # (d_head//2,)
    freqs_lin = compute_linear_scaled_freqs(d_head, base, scale_factor)
    freqs_ntk = compute_ntk_freqs(d_head, base, scale_factor, 0)

    # Wavelength per frequency dimension: lambda_i = 2*pi / freq_i
    wavelengths = 2.0 * math.pi / freqs_std  # (d_head//2,)

    # Threshold wavelengths
    lambda_fast = 2.0 * math.pi / beta_fast  # below this -> NTK
    lambda_slow = 2.0 * math.pi / beta_slow  # above this -> linear

    # Blend factor alpha in [0, 1]: 0 = NTK, 1 = linear
    alpha = (wavelengths - lambda_fast) / (lambda_slow - lambda_fast + 1e-8)
    alpha = alpha.clamp(0.0, 1.0)

    yarn_freqs = (1.0 - alpha) * freqs_ntk + alpha * freqs_lin
    return yarn_freqs


# ---------------------------------------------------------------------------
# Rotation matrix and application
# ---------------------------------------------------------------------------


def build_rotation_matrix(freqs: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
    """Build cosine and sine rotation tables from frequency tensor.

    Computes outer product of positions [0, seq_len) with freqs to get angles,
    then takes cos and sin.

    Args:
        freqs:   Frequency tensor of shape (d_head//2,).
        seq_len: Number of positions to compute.

    Returns:
        Tuple of (cos, sin), each of shape (seq_len, d_head//2).
    """
    positions = torch.arange(seq_len, dtype=torch.float32, device=freqs.device)
    angles = torch.outer(positions, freqs)  # (seq_len, d_head//2)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Splits x into pairs along the last dimension and rotates each pair by the
    corresponding angle: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos].

    Args:
        x:   Input tensor of shape (B, T, d_head).
        cos: Cosine table of shape (T, d_head//2).
        sin: Sine table of shape (T, d_head//2).

    Returns:
        Rotated tensor of shape (B, T, d_head).
    """
    B, T, d = x.shape
    half = d // 2

    x1 = x[..., :half]  # (B, T, d//2)
    x2 = x[..., half:]  # (B, T, d//2)

    # cos/sin: (T, d//2) -> broadcast over batch
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return torch.cat([rotated_x1, rotated_x2], dim=-1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def get_rope_freqs(config: RoPEConfig) -> Tensor:
    """Dispatch to the correct frequency function based on config.rope_type.

    Args:
        config: RoPEConfig specifying type and all parameters.

    Returns:
        Frequency tensor of shape (d_head//2,).
    """
    if config.rope_type == "standard":
        return compute_standard_freqs(config.d_head, config.base)
    elif config.rope_type == "linear_scaled":
        return compute_linear_scaled_freqs(config.d_head, config.base, config.scale_factor)
    elif config.rope_type == "dynamic_ntk":
        return compute_ntk_freqs(
            config.d_head, config.base, config.scale_factor, config.max_seq_len
        )
    elif config.rope_type == "yarn":
        return compute_yarn_freqs(
            config.d_head,
            config.base,
            config.scale_factor,
            config.yarn_beta_fast,
            config.yarn_beta_slow,
        )
    else:
        raise ValueError(f"Unknown rope_type: '{config.rope_type}'")


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------


class RoPEVariant(nn.Module):
    """Rotary position embedding module supporting multiple RoPE variants.

    Pre-computes and registers cos/sin tables as buffers for efficiency.

    Args:
        config: RoPEConfig specifying variant and all hyperparameters.
    """

    def __init__(self, config: RoPEConfig) -> None:
        super().__init__()
        self.config = config
        freqs = get_rope_freqs(config)
        cos, sin = build_rotation_matrix(freqs, config.max_seq_len)
        self.register_buffer("cos_cached", cos)  # (max_seq_len, d_head//2)
        self.register_buffer("sin_cached", sin)  # (max_seq_len, d_head//2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Input of shape (B, T, d_head) where T <= max_seq_len.

        Returns:
            Rotated tensor of shape (B, T, d_head).
        """
        T = x.shape[1]
        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]
        return apply_rope(x, cos, sin)

    def extend_to(self, new_max_len: int) -> None:
        """Recompute and update cos/sin buffers to support longer sequences.

        Args:
            new_max_len: New maximum sequence length (must be >= current max_seq_len).
        """
        self.config = RoPEConfig(
            d_head=self.config.d_head,
            base=self.config.base,
            max_seq_len=new_max_len,
            rope_type=self.config.rope_type,
            scale_factor=self.config.scale_factor,
            yarn_beta_fast=self.config.yarn_beta_fast,
            yarn_beta_slow=self.config.yarn_beta_slow,
        )
        freqs = get_rope_freqs(self.config)
        cos, sin = build_rotation_matrix(freqs, new_max_len)
        self.cos_cached = cos.to(self.cos_cached.device)
        self.sin_cached = sin.to(self.sin_cached.device)
