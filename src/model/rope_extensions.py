"""Extended RoPE variants: YaRN, LongRoPE, and Dynamic NTK scaling.

All implementations use pure native PyTorch — no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RoPEConfig:
    """Configuration for RoPE and its extended variants."""

    head_dim: int
    base: float = 10000.0
    max_seq_len: int = 4096
    scale_factor: float = 1.0          # used by LinearScaledRoPE / NTKScaledRoPE
    original_max_len: int = 4096       # context length the model was trained with


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dimension of *x* by 90 degrees.

    Splits the last dimension in two equal halves [x1, x2] and returns
    [-x2, x1], which is equivalent to multiplying by the rotation matrix
    [[0, -1], [1, 0]] applied block-wise.

    Args:
        x: Tensor of shape (..., d) where d is even.

    Returns:
        Rotated tensor of the same shape.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseRoPE(nn.Module):
    """Classic Rotary Position Embedding (RoPE).

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021).
    """

    def __init__(self, config: RoPEConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Core frequency computation
    # ------------------------------------------------------------------

    def _inv_freqs(self, device: torch.device) -> Tensor:
        """Return inverse frequencies (head_dim // 2,) using the configured base."""
        head_dim = self.config.head_dim
        # theta_i = 1 / base^(2i / head_dim)  for i in 0 .. head_dim//2 - 1
        i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.config.base ** (i / head_dim))

    def get_freqs(self, seq_len: int, device: torch.device) -> Tensor:
        """Compute RoPE cosine/sine frequency table.

        Args:
            seq_len: Number of positions to compute.
            device:  Target device.

        Returns:
            Tensor of shape (seq_len, head_dim // 2) containing the angular
            positions (m * theta_i) — callers should take cos/sin of these.
        """
        inv_freqs = self._inv_freqs(device)                          # (D/2,)
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # (T,)
        # outer product: (T, D/2)
        freqs = torch.outer(positions, inv_freqs)
        return freqs

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, x: Tensor, seq_len: int) -> Tensor:
        """Apply RoPE to query or key tensor using rotate_half style.

        Args:
            x:       Tensor of shape (B, T, n_heads, head_dim).
            seq_len: Sequence length (used to build the frequency table).

        Returns:
            Rotated tensor of the same shape as *x*.
        """
        device = x.device
        freqs = self.get_freqs(seq_len, device)          # (T, D/2)
        # Build full (T, D) angle by duplicating: [theta, theta]
        freqs_full = torch.cat([freqs, freqs], dim=-1)   # (T, D)
        # Reshape for broadcasting: (1, T, 1, D)
        cos = freqs_full.cos().unsqueeze(0).unsqueeze(2)
        sin = freqs_full.sin().unsqueeze(0).unsqueeze(2)
        return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Linear (position-interpolation) scaled RoPE
# ---------------------------------------------------------------------------

class LinearScaledRoPE(BaseRoPE):
    """RoPE with linear position interpolation.

    Divides each position index by *scale_factor* before computing the
    rotation angles, effectively stretching the positional space to cover
    a longer context without retraining.

    Reference: "Extending Context Window of Large Language Models via
    Positional Interpolation" (Chen et al., 2023).
    """

    def get_freqs(self, seq_len: int, device: torch.device) -> Tensor:
        inv_freqs = self._inv_freqs(device)                          # (D/2,)
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        # Scale positions down so they stay within the original trained range.
        positions = positions / self.config.scale_factor
        freqs = torch.outer(positions, inv_freqs)
        return freqs


# ---------------------------------------------------------------------------
# NTK-aware scaled RoPE
# ---------------------------------------------------------------------------

class NTKScaledRoPE(BaseRoPE):
    """NTK-aware RoPE: rescales the base frequency rather than positions.

    Rescaling the base avoids the interpolation quality degradation seen with
    linear scaling while still extending the effective context window.

    The new base is: base' = base * scale_factor^(head_dim / (head_dim - 2))

    Reference: "Scaling Laws of RoPE-based Extrapolation" / Reddit NTK post.
    """

    def _ntk_base(self) -> float:
        cfg = self.config
        return cfg.base * (cfg.scale_factor ** (cfg.head_dim / (cfg.head_dim - 2)))

    def get_freqs(self, seq_len: int, device: torch.device) -> Tensor:
        head_dim = self.config.head_dim
        ntk_base = self._ntk_base()
        i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freqs = 1.0 / (ntk_base ** (i / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        return torch.outer(positions, inv_freqs)


# ---------------------------------------------------------------------------
# YaRN RoPE
# ---------------------------------------------------------------------------

class YaRNRoPE(BaseRoPE):
    """YaRN: Yet Another RoPE extensioN.

    Mixes NTK-scaled and linear-scaled frequencies per frequency dimension
    using a smooth ramp function, giving the best of both worlds.

    For each frequency dimension i:
      - Compute the "wavelength" lambda_i = 2*pi / theta_i
      - Compute d_i = original_max_len / lambda_i  (how many full rotations
        the original context makes at this frequency)
      - Ramp: r_i = clip((d_i - alpha) / (beta - alpha), 0, 1)
        * r_i = 0 -> high-frequency dimension: use linear interpolation
        * r_i = 1 -> low-frequency dimension:  use NTK scaling
      - freq_i = r_i * ntk_freq_i + (1 - r_i) * linear_freq_i

    Reference: "YaRN: Efficient Context Window Extension of Large Language
    Models" (Peng et al., 2023).
    """

    def __init__(self, config: RoPEConfig, alpha: float = 1.0, beta: float = 32.0) -> None:
        super().__init__(config)
        self.alpha = alpha
        self.beta = beta

    def get_freqs(self, seq_len: int, device: torch.device) -> Tensor:
        cfg = self.config
        head_dim = cfg.head_dim

        i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)

        # Base inverse frequencies (theta_i = base^(-2i/D))
        base_inv_freqs = 1.0 / (cfg.base ** (i / head_dim))           # (D/2,)

        # NTK-scaled inverse frequencies
        ntk_base = cfg.base * (cfg.scale_factor ** (head_dim / (head_dim - 2)))
        ntk_inv_freqs = 1.0 / (ntk_base ** (i / head_dim))            # (D/2,)

        # Linear-scaled inverse frequencies (divide positions by scale_factor)
        # Equivalent to keeping inv_freqs the same but positions are scaled —
        # here we encode it as effective_inv_freqs = base_inv_freqs / scale_factor
        linear_inv_freqs = base_inv_freqs / cfg.scale_factor           # (D/2,)

        # Wavelength of each frequency: lambda_i = 2*pi / theta_i
        wavelengths = (2.0 * math.pi) / base_inv_freqs                 # (D/2,)

        # d_i: number of wavelengths that fit in original_max_len
        d = cfg.original_max_len / wavelengths                         # (D/2,)

        # Ramp r_i in [0, 1]: 0 = linear, 1 = NTK
        r = ((d - self.alpha) / (self.beta - self.alpha)).clamp(0.0, 1.0)  # (D/2,)

        # Mixed inverse frequencies
        mixed_inv_freqs = r * ntk_inv_freqs + (1.0 - r) * linear_inv_freqs  # (D/2,)

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        return torch.outer(positions, mixed_inv_freqs)


# ---------------------------------------------------------------------------
# Dynamic NTK RoPE
# ---------------------------------------------------------------------------

class DynamicNTKRoPE(BaseRoPE):
    """Dynamic NTK RoPE: adjusts base on-the-fly based on actual sequence length.

    When seq_len exceeds original_max_len the base is rescaled proportionally
    so the effective context window expands to cover seq_len without any
    pre-configuration.  For sequences within the original context window,
    the standard base is used unchanged.

    scale = (seq_len / original_max_len)^(head_dim / (head_dim - 2))
    new_base = base * scale   (only when seq_len > original_max_len)
    """

    def get_freqs(self, seq_len: int, device: torch.device) -> Tensor:
        cfg = self.config
        head_dim = cfg.head_dim

        if seq_len > cfg.original_max_len:
            scale = (seq_len / cfg.original_max_len) ** (head_dim / (head_dim - 2))
            effective_base = cfg.base * scale
        else:
            effective_base = cfg.base

        i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freqs = 1.0 / (effective_base ** (i / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        return torch.outer(positions, inv_freqs)
