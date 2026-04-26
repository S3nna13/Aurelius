"""RoPE and context-length extension variants (v2).

Implements standard RoPE plus NTK-aware, YaRN, and Dynamic NTK scaling.
Pure native PyTorch — no external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Standard RoPE
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Standard Rotary Position Embedding (RoPE).

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", 2021.

    Args:
        head_dim: Dimension of each attention head (must be even).
        base:     Base for geometric frequency sequence.
        max_seq_len: Maximum sequence length (used for optional precompute).
    """

    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute theta_i = 1 / base^(2i/d) for i in 0..d//2-1
        # Shape: (head_dim // 2,)
        head_dim // 2
        i = torch.arange(0, head_dim, 2, dtype=torch.float32)  # [0, 2, 4, ...]
        inv_freqs = 1.0 / (base ** (i / head_dim))
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)

        self._cached_seq_len: int = -1
        self._cached_freqs: Tensor | None = None

    def _compute_freqs(self, seq_len: int) -> Tensor:
        """Compute complex exponential frequency table.

        Returns:
            Tensor of shape (seq_len, head_dim // 2) with complex values
            exp(i * theta * pos).
        """
        device = self.inv_freqs.device
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        # (seq_len, head_dim//2) — outer product of positions and inv_freqs
        angles = torch.outer(positions, self.inv_freqs)
        # Convert to complex exponentials: exp(i * angle)
        freqs_complex = torch.polar(torch.ones_like(angles), angles)
        return freqs_complex

    def _get_freqs(self, seq_len: int) -> Tensor:
        if self._cached_seq_len == seq_len and self._cached_freqs is not None:
            return self._cached_freqs
        freqs = self._compute_freqs(seq_len)
        self._cached_seq_len = seq_len
        self._cached_freqs = freqs
        return freqs

    def _apply_rope(self, x: Tensor, freqs: Tensor) -> Tensor:
        """Apply rotary embeddings to tensor x using complex multiplication.

        Args:
            x:     Tensor of shape (B, H, T, D_head).
            freqs: Complex tensor of shape (T, D_head // 2).

        Returns:
            Rotated tensor of same shape as x.
        """
        B, H, T, D = x.shape
        # Reshape to complex: (B, H, T, D//2, 2) -> complex (B, H, T, D//2)
        x_c = torch.view_as_complex(x.reshape(B, H, T, D // 2, 2))
        # freqs: (T, D//2) -> broadcast to (1, 1, T, D//2)
        freqs_bc = freqs.unsqueeze(0).unsqueeze(0)
        # Multiply in complex space
        x_rot_c = x_c * freqs_bc
        # Convert back to real: (B, H, T, D//2, 2) -> (B, H, T, D)
        x_rot = torch.view_as_real(x_rot_c).flatten(-2)
        return x_rot

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor of shape (B, H, T, D_head).
            k: Key tensor of shape (B, H, T, D_head).

        Returns:
            Tuple (q_rot, k_rot) with rotary embeddings applied.
        """
        seq_len = q.shape[2]
        freqs = self._get_freqs(seq_len)
        q_rot = self._apply_rope(q, freqs)
        k_rot = self._apply_rope(k, freqs)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# NTK-aware scaled RoPE
# ---------------------------------------------------------------------------


class NTKRoPE(RotaryEmbedding):
    """NTK-aware scaled RoPE for context-length extension.

    Instead of interpolating positions, the base frequency is scaled up
    so that the effective context window expands to scale_factor * original.

    Modified base: base_new = base * scale_factor^(head_dim / (head_dim - 2))

    Reference: "Scaling Laws of RoPE-based Extrapolation" / NTK-aware scaling.

    Args:
        head_dim:     Dimension of each attention head.
        base:         Original RoPE base.
        scale_factor: Context-extension factor (e.g. 8.0 for 8x extension).
    """

    def __init__(self, head_dim: int, base: float = 10000.0, scale_factor: float = 8.0) -> None:
        # Compute scaled base before calling super().__init__
        base_new = base * (scale_factor ** (head_dim / (head_dim - 2)))
        super().__init__(head_dim=head_dim, base=base_new)
        self._original_base = base
        self._scale_factor = scale_factor

    def context_window_extension(self) -> float:
        """Return the theoretical context extension factor."""
        return self._scale_factor


# ---------------------------------------------------------------------------
# YaRN RoPE
# ---------------------------------------------------------------------------


class YaRNRoPE(nn.Module):
    """YaRN: Yet Another RoPE extensioN.

    Uses frequency-range-dependent interpolation: high-frequency dimensions
    are left unchanged while low-frequency dimensions are scaled.

    Reference: Peng et al., "YaRN: Efficient Context Window Extension of
    Large Language Models", 2023.

    Args:
        head_dim:    Dimension of each attention head.
        base:        Original RoPE base.
        scale_factor: Context scaling factor.
        beta_fast:   High-frequency threshold (small wavelength, no change).
        beta_slow:   Low-frequency threshold (large wavelength, scale).
        mscale:      Magnitude scaling coefficient.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        scale_factor: float = 8.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 0.1,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scale_factor = scale_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale

        self._cached_seq_len: int = -1
        self._cached_freqs: Tensor | None = None

    def _ramp_fn(self, wavelength: float) -> float:
        """Compute interpolation coefficient (ramp) for a given wavelength.

        Returns a value in [0, 1]:
          - 1.0 → pure original (high-frequency, short wavelength)
          - 0.0 → pure scaled (low-frequency, long wavelength)

        Args:
            wavelength: 2π / theta_i for dimension i.

        Returns:
            Float interpolation coefficient in [0, 1].
        """
        # Reference scale: original training context length approximated via base
        # Use base_scale = scale_factor as the reference period multiplier
        base_scale = self.scale_factor

        # Boundaries in wavelength space
        low_freq_wavelen = base_scale * 2.0 * math.pi / self.beta_slow
        high_freq_wavelen = base_scale * 2.0 * math.pi / self.beta_fast

        if wavelength < high_freq_wavelen:
            # Short wavelength → high frequency → no change (ramp = 1)
            return 1.0
        elif wavelength > low_freq_wavelen:
            # Long wavelength → low frequency → scale (ramp = 0)
            return 0.0
        else:
            # Linear ramp between boundaries
            return (low_freq_wavelen - wavelength) / (low_freq_wavelen - high_freq_wavelen)

    def _compute_freqs(self, seq_len: int) -> Tensor:
        """Compute YaRN-modified complex frequency table.

        Returns:
            Complex tensor of shape (seq_len, head_dim // 2).
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        half = self.head_dim // 2

        # Base inverse frequencies
        idx = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        base_inv_freqs = 1.0 / (self.base ** (idx / self.head_dim))  # (half,)

        # Build blended inverse frequencies dimension by dimension
        blended = torch.empty(half, dtype=torch.float32, device=device)
        mag_scale = 1.0 + self.mscale * math.log(self.scale_factor)

        for i in range(half):
            theta_i = base_inv_freqs[i].item()
            wavelength = (2.0 * math.pi) / theta_i if theta_i > 0.0 else float("inf")
            ramp = self._ramp_fn(wavelength)

            # Scaled (interpolated) frequency: divide by scale_factor
            scaled_theta_i = theta_i / self.scale_factor
            # Blend: ramp=1 → original, ramp=0 → scaled
            blended[i] = ramp * theta_i + (1.0 - ramp) * scaled_theta_i

        # Apply magnitude correction
        blended = blended * mag_scale

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, blended)
        freqs_complex = torch.polar(torch.ones_like(angles), angles)
        return freqs_complex

    def _get_freqs(self, seq_len: int) -> Tensor:
        if self._cached_seq_len == seq_len and self._cached_freqs is not None:
            return self._cached_freqs
        freqs = self._compute_freqs(seq_len)
        self._cached_seq_len = seq_len
        self._cached_freqs = freqs
        return freqs

    def _apply_rope(self, x: Tensor, freqs: Tensor) -> Tensor:
        B, H, T, D = x.shape
        x_c = torch.view_as_complex(x.reshape(B, H, T, D // 2, 2))
        freqs_bc = freqs.unsqueeze(0).unsqueeze(0)
        x_rot_c = x_c * freqs_bc
        return torch.view_as_real(x_rot_c).flatten(-2)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply YaRN RoPE to query and key tensors.

        Args:
            q: Query tensor of shape (B, H, T, D_head).
            k: Key tensor of shape (B, H, T, D_head).

        Returns:
            Tuple (q_rot, k_rot).
        """
        seq_len = q.shape[2]
        freqs = self._get_freqs(seq_len)
        q_rot = self._apply_rope(q, freqs)
        k_rot = self._apply_rope(k, freqs)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Dynamic NTK RoPE
# ---------------------------------------------------------------------------


class DynamicNTKRoPE(nn.Module):
    """Dynamic NTK RoPE: adapts scale at runtime based on sequence length.

    When seq_len exceeds max_position_embeddings, the base is rescaled
    proportionally so the effective context window covers the longer sequence.
    For sequences within the original limit, standard RoPE is used.

    Args:
        head_dim:               Dimension of each attention head.
        base:                   Original RoPE base.
        max_position_embeddings: Original training context length.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        # Cache: map seq_len -> complex freqs tensor
        self._cache: dict[int, Tensor] = {}

    def _compute_freqs_for_scale(self, seq_len: int, scale_factor: float) -> Tensor:
        """Compute complex frequency table for given seq_len and scale."""
        # NTK-scaled base when scale_factor > 1, otherwise standard base
        if scale_factor > 1.0:
            effective_base = self.base * (scale_factor ** (self.head_dim / (self.head_dim - 2)))
        else:
            effective_base = self.base

        idx = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freqs = 1.0 / (effective_base ** (idx / self.head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freqs)
        freqs_complex = torch.polar(torch.ones_like(angles), angles)
        return freqs_complex

    def _get_freqs(self, seq_len: int) -> Tensor:
        if seq_len in self._cache:
            return self._cache[seq_len]

        if seq_len > self.max_position_embeddings:
            scale_factor = (seq_len / self.max_position_embeddings) ** (
                self.head_dim / (self.head_dim - 2)
            )
        else:
            scale_factor = 1.0

        freqs = self._compute_freqs_for_scale(seq_len, scale_factor)
        self._cache[seq_len] = freqs
        return freqs

    def _apply_rope(self, x: Tensor, freqs: Tensor) -> Tensor:
        B, H, T, D = x.shape
        x_c = torch.view_as_complex(x.reshape(B, H, T, D // 2, 2))
        freqs_bc = freqs.unsqueeze(0).unsqueeze(0)
        x_rot_c = x_c * freqs_bc
        return torch.view_as_real(x_rot_c).flatten(-2)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply Dynamic NTK RoPE to query and key tensors.

        Automatically adapts scale based on actual sequence length.

        Args:
            q: Query tensor of shape (B, H, T, D_head).
            k: Key tensor of shape (B, H, T, D_head).

        Returns:
            Tuple (q_rot, k_rot).
        """
        seq_len = q.shape[2]
        freqs = self._get_freqs(seq_len)
        q_rot = self._apply_rope(q, freqs)
        k_rot = self._apply_rope(k, freqs)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# RoPE Analyzer
# ---------------------------------------------------------------------------


class RoPEAnalyzer:
    """Analyze properties of RoPE and its variants.

    Provides utilities to measure effective context length, frequency
    distribution, and rotation matrices.
    """

    def __init__(self) -> None:
        pass

    def effective_context_length(self, rope_module: nn.Module, max_seq_len: int = 4096) -> int:
        """Estimate effective context length by measuring attention score decay.

        Computes the cosine similarity between a query at position 0 and
        keys at increasing positions.  Returns the first position where
        mean cosine similarity drops below 0.5.

        Args:
            rope_module: A module with a ``forward(q, k)`` method.
            max_seq_len: Upper bound on positions to test.

        Returns:
            Effective context length (int).
        """
        # Detect head_dim from module attribute
        head_dim: int
        if hasattr(rope_module, "head_dim"):
            head_dim = rope_module.head_dim
        else:
            head_dim = 64  # fallback

        torch.manual_seed(0)
        # Single query at position 0, keys at all positions
        q = torch.randn(1, 1, max_seq_len, head_dim)
        k = torch.randn(1, 1, max_seq_len, head_dim)

        with torch.no_grad():
            q_rot, k_rot = rope_module.forward(q, k)

        # Reference: query at position 0
        q0 = q_rot[:, :, 0:1, :]  # (1, 1, 1, D)
        # Cosine similarity with each key position
        q0_norm = q0 / (q0.norm(dim=-1, keepdim=True) + 1e-8)
        k_norm = k_rot / (k_rot.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = (q0_norm * k_norm).sum(dim=-1).squeeze()  # (max_seq_len,)

        threshold = 0.5
        for pos in range(max_seq_len):
            if cos_sim[pos].item() < threshold:
                return pos

        return max_seq_len

    def frequency_distribution(self, head_dim: int, base: float = 10000.0) -> dict:
        """Analyze the frequency distribution of standard RoPE.

        Args:
            head_dim: Attention head dimension.
            base:     RoPE base frequency.

        Returns:
            Dictionary with keys: min_freq, max_freq, n_high_freq, n_low_freq.
            High-frequency: wavelength < head_dim (many rotations per context).
            Low-frequency:  wavelength >= head_dim.
        """
        idx = torch.arange(0, head_dim, 2, dtype=torch.float64)
        inv_freqs = 1.0 / (base ** (idx / head_dim))  # theta_i values

        wavelengths = (2.0 * math.pi) / inv_freqs  # lambda_i = 2pi / theta_i

        min_freq = inv_freqs.min().item()
        max_freq = inv_freqs.max().item()

        n_high_freq = int((wavelengths < head_dim).sum().item())
        n_low_freq = int((wavelengths >= head_dim).sum().item())

        return {
            "min_freq": min_freq,
            "max_freq": max_freq,
            "n_high_freq": n_high_freq,
            "n_low_freq": n_low_freq,
        }

    def rotation_matrix(self, rope_module: nn.Module, seq_pos: int) -> Tensor:
        """Compute the (D, D) rotation matrix applied at a given sequence position.

        Constructs the rotation matrix by applying the RoPE to the standard
        basis vectors of R^D.

        Args:
            rope_module: Module with ``forward(q, k)`` interface.
            seq_pos:     Position index to evaluate (0-indexed).

        Returns:
            Tensor of shape (D, D).
        """
        if hasattr(rope_module, "head_dim"):
            head_dim = rope_module.head_dim
        else:
            head_dim = 64

        # We need to evaluate the rotation at a specific position.
        # Build identity matrix padded to full sequence of length seq_pos+1.
        total_len = seq_pos + 1

        # Query = identity at position seq_pos, zeros elsewhere
        # Shape: (1, 1, total_len, head_dim)
        torch.zeros(1, 1, total_len, head_dim)
        torch.zeros(1, 1, total_len, head_dim)

        # Fill the target position with identity basis vectors one column at a time
        rot_cols = []
        for col in range(head_dim):
            q_col = torch.zeros(1, 1, total_len, head_dim)
            q_col[0, 0, seq_pos, col] = 1.0
            k_col = torch.zeros(1, 1, total_len, head_dim)

            with torch.no_grad():
                q_rot, _ = rope_module.forward(q_col, k_col)

            rot_cols.append(q_rot[0, 0, seq_pos, :])  # (D,)

        # Stack columns to get rotation matrix: R[col, :] = rotated basis vector
        # R is (head_dim, head_dim); transpose to get (D, D) where R @ e_i = col_i
        R = torch.stack(rot_cols, dim=0).T  # (D, D)
        return R
