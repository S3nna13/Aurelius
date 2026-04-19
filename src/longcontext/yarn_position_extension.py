"""Standalone YaRN rotary-position-extension utility.

Implements the YaRN (Yet another RoPE extensioN) method from
Peng et al. 2023, "YaRN: Efficient Context Window Extension of Large
Language Models" (arXiv:2309.00071).

This is a STANDALONE helper library. ``src/model/rope.py`` is frozen and
already handles YaRN inline for the core model path; this module exists
so that other long-context components (benchmarks, ablation tooling,
standalone probes, etc.) can construct YaRN-style rotary caches without
having to import or depend on ``src.model``.

Public surface (see ``__init__.py``):

    YarnConfig                 -- dataclass of YaRN hyperparameters
    yarn_inv_freq              -- adjusted inverse frequencies
    yarn_linear_ramp_mask      -- smooth NTK-by-parts transition region
    yarn_mscale                -- per-position attention scale factor
    build_yarn_rotary_cache    -- (cos, sin) rotary cache of shape [S, D]
    apply_rotary               -- standard RoPE application, reusable

The math follows the paper:

* NTK-by-parts: for each frequency band, interpolate between
  "extrapolation" (keep inv_freq as-is) for high-frequency bands
  (wavelengths <= original_max_seq_len / beta_fast) and
  "interpolation" (divide inv_freq by scaling_factor) for
  low-frequency bands (wavelengths >= original_max_seq_len / beta_slow),
  with a linear ramp between them.
* Attention mscale: a temperature factor
  ``mscale = 1 + mscale_factor * log(scaling_factor)`` is folded into
  the cos/sin so that attention logits stay calibrated after stretching.

Dependencies: pure PyTorch. No coupling to ``src.model`` or foreign
ML frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import math

import torch
from torch import Tensor


@dataclass
class YarnConfig:
    """Hyperparameters for a YaRN rotary-position extension.

    Attributes
    ----------
    head_dim:
        Per-head embedding dimension. Must be even (RoPE pairs adjacent
        components).
    rope_theta:
        Base ``theta`` of the original RoPE (``10000`` for vanilla RoPE,
        ``500000`` for Llama-3-style long-context bases).
    original_max_seq_len:
        Sequence length the model was *originally* trained on. This is
        the reference length the ramp is computed against.
    scaling_factor:
        Target extension factor, i.e. the ratio
        ``new_max_seq_len / original_max_seq_len``. Must be ``>= 1``.
        A value of ``1.0`` is a no-op (YaRN reduces to plain RoPE).
    beta_fast:
        Number of rotations after which a frequency band is considered
        "fast" (extrapolated, not rescaled). Default 32 per the paper.
    beta_slow:
        Number of rotations below which a frequency band is considered
        "slow" (interpolated, rescaled by ``scaling_factor``). Default 1.
    mscale_factor:
        Coefficient in ``mscale = 1 + mscale_factor * log(scaling)``.
        The paper uses ``0.1`` for Llama-family models.
    """

    head_dim: int
    rope_theta: float = 500_000.0
    original_max_seq_len: int = 8192
    scaling_factor: float = 4.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale_factor: float = 0.1

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be a positive even integer, got {self.head_dim}"
            )
        if self.scaling_factor < 1.0:
            raise ValueError(
                f"scaling_factor must be >= 1.0, got {self.scaling_factor}"
            )
        if self.original_max_seq_len <= 0:
            raise ValueError(
                f"original_max_seq_len must be positive, got {self.original_max_seq_len}"
            )
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.beta_fast <= self.beta_slow:
            raise ValueError(
                f"beta_fast ({self.beta_fast}) must be > beta_slow ({self.beta_slow})"
            )


# ---------------------------------------------------------------------------
# Core YaRN math
# ---------------------------------------------------------------------------


def _find_correction_dim(
    num_rotations: float, head_dim: int, base: float, original_max_seq_len: int
) -> float:
    """Invert ``wavelength(dim) = original_max_seq_len / num_rotations``.

    For RoPE, the wavelength of frequency band ``i`` (of the
    ``head_dim/2`` bands) is ``2*pi * base^(2i/head_dim)``. Setting this
    equal to ``original_max_seq_len / num_rotations`` and solving for
    ``i`` gives the fractional band index at which exactly
    ``num_rotations`` rotations occur over the original context window.
    """
    return (
        head_dim
        * math.log(original_max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _find_correction_range(config: YarnConfig) -> tuple[int, int]:
    """Return the ``(low, high)`` band indices bounding the ramp region.

    Bands ``< low`` (high frequency / short wavelength) are extrapolated;
    bands ``> high`` (low frequency / long wavelength) are interpolated;
    bands in ``[low, high]`` get a linear ramp between the two regimes.
    """
    low = math.floor(
        _find_correction_dim(
            config.beta_fast,
            config.head_dim,
            config.rope_theta,
            config.original_max_seq_len,
        )
    )
    high = math.ceil(
        _find_correction_dim(
            config.beta_slow,
            config.head_dim,
            config.rope_theta,
            config.original_max_seq_len,
        )
    )
    # Clamp to a valid range for head_dim/2 bands.
    low = max(low, 0)
    high = min(high, config.head_dim // 2 - 1)
    if high <= low:
        high = low + 1
    return low, high


def yarn_linear_ramp_mask(
    config: YarnConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Smooth transition mask over frequency bands, shape ``[head_dim//2]``.

    Returns values in ``[0, 1]``:

    * ``0`` -> extrapolate (high-frequency band, keep original inv_freq)
    * ``1`` -> interpolate (low-frequency band, rescale by
      ``1/scaling_factor``)
    * values in between -> linear ramp

    Note: by convention in this module, ``mask=0`` means
    "extrapolation" and ``mask=1`` means "interpolation", and the
    blended inverse frequency is
    ``inv_freq * ((1 - mask) + mask / scaling_factor)``.
    """
    low, high = _find_correction_range(config)
    half = config.head_dim // 2
    idx = torch.arange(half, device=device, dtype=dtype)
    # Linear ramp from low to high; clamp to [0, 1].
    ramp = (idx - low) / max(high - low, 1)
    return torch.clamp(ramp, 0.0, 1.0)


def yarn_inv_freq(
    config: YarnConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the YaRN-adjusted inverse frequencies, shape ``[head_dim//2]``.

    Blends the "extrapolation" inverse frequencies (plain RoPE) with the
    "interpolation" ones (PI, inv_freq / scaling_factor) using the
    linear ramp from :func:`yarn_linear_ramp_mask`.
    """
    half = config.head_dim // 2
    idx = torch.arange(0, half, device=device, dtype=dtype)
    # Plain RoPE frequencies: theta^(-2i/head_dim).
    inv_freq_extrap = 1.0 / (
        config.rope_theta ** (2.0 * idx / config.head_dim)
    )
    inv_freq_interp = inv_freq_extrap / config.scaling_factor
    mask = yarn_linear_ramp_mask(config, device=device, dtype=dtype)
    # mask=0 -> extrap, mask=1 -> interp.
    inv_freq = inv_freq_extrap * (1.0 - mask) + inv_freq_interp * mask
    return inv_freq


def _mscale_scalar(config: YarnConfig) -> float:
    """The paper's scalar mscale: 1 + mscale_factor * log(scaling)."""
    if config.scaling_factor <= 1.0:
        return 1.0
    return 1.0 + config.mscale_factor * math.log(float(config.scaling_factor))


def yarn_mscale(
    config: YarnConfig,
    position: Union[int, Tensor],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the attention-temperature scale at ``position``.

    For positions within the original training window
    (``position < original_max_seq_len``) the scale is ~1.0; for
    positions beyond it the scale rises toward the paper's constant
    ``mscale = 1 + mscale_factor * log(scaling_factor)`` via a smooth
    transition. This matches the intent of the YaRN mscale: keep
    in-distribution attention unchanged while re-calibrating logits on
    the extrapolated region.
    """
    base = 1.0
    peak = _mscale_scalar(config)

    if isinstance(position, Tensor):
        pos = position.to(device=device, dtype=dtype)
    else:
        pos = torch.tensor(float(position), device=device, dtype=dtype)

    orig = float(config.original_max_seq_len)
    # Smoothstep transition: 0 at position == orig, 1 at position == scaling*orig.
    extended = float(config.scaling_factor) * orig
    denom = max(extended - orig, 1.0)
    t = torch.clamp((pos - orig) / denom, 0.0, 1.0)
    # Smoothstep for a gentle interior gradient.
    smooth = t * t * (3.0 - 2.0 * t)
    return base + (peak - base) * smooth


# ---------------------------------------------------------------------------
# Rotary cache construction & application
# ---------------------------------------------------------------------------


def build_yarn_rotary_cache(
    config: YarnConfig,
    seq_len: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Return ``(cos, sin)`` rotary caches, each of shape ``[seq_len, head_dim]``.

    The per-band mscale temperature is folded directly into the cos/sin
    amplitudes (a scalar ``mscale``, following the paper's practical
    recipe), which saves an explicit multiply in
    :func:`apply_rotary`. Callers that need the raw (unscaled) cos/sin
    should rebuild with ``mscale_factor=0.0`` in the config.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    inv_freq = yarn_inv_freq(config, device=device, dtype=dtype)  # [D/2]
    t = torch.arange(seq_len, device=device, dtype=dtype)  # [S]
    # freqs[s, i] = t[s] * inv_freq[i]
    freqs = torch.outer(t, inv_freq)  # [S, D/2]
    # Duplicate to reach head_dim: [f0, f1, ..., f0, f1, ...] (Llama convention).
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]

    mscale = _mscale_scalar(config)
    cos = emb.cos() * mscale
    sin = emb.sin() * mscale
    # Clamp is a no-op mathematically (cos/sin in [-1,1], mscale ~1) but
    # guards numerically against 1 + eps overshoot on large scaling.
    cos = torch.clamp(cos, -float(max(mscale, 1.0)), float(max(mscale, 1.0)))
    sin = torch.clamp(sin, -float(max(mscale, 1.0)), float(max(mscale, 1.0)))
    return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    """Llama-style rotate-half: split last dim in two halves and swap/negate."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to ``x`` using precomputed ``cos``/``sin``.

    Parameters
    ----------
    x:
        Tensor of shape ``[..., S, D]`` (e.g. ``[B, H, S, D]``).
    cos, sin:
        Tensors of shape ``[S, D]`` (as returned by
        :func:`build_yarn_rotary_cache`) or broadcastable to ``x``.

    Returns a tensor of the same shape and dtype as ``x``.
    """
    if x.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"head_dim mismatch: x has D={x.shape[-1]}, cos has D={cos.shape[-1]}"
        )
    if x.shape[-2] > cos.shape[-2]:
        raise ValueError(
            f"seq_len mismatch: x has S={x.shape[-2]}, cos has S={cos.shape[-2]}"
        )

    s = x.shape[-2]
    cos_s = cos[:s].to(dtype=x.dtype, device=x.device)
    sin_s = sin[:s].to(dtype=x.dtype, device=x.device)
    # Broadcast over leading dims.
    while cos_s.dim() < x.dim():
        cos_s = cos_s.unsqueeze(0)
        sin_s = sin_s.unsqueeze(0)
    return x * cos_s + _rotate_half(x) * sin_s


__all__ = [
    "YarnConfig",
    "yarn_inv_freq",
    "yarn_linear_ramp_mask",
    "yarn_mscale",
    "build_yarn_rotary_cache",
    "apply_rotary",
]
