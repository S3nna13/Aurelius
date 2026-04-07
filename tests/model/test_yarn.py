"""Tests for YaRN RoPE context extension."""
import torch
import pytest

from src.model.config import AureliusConfig
from src.model.attention import precompute_rope_frequencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEAD_DIM = 64
SEQ_LEN = 32
THETA = 500_000.0


def _freqs(scaling_type="none", factor=1.0, **kwargs):
    """Convenience wrapper around precompute_rope_frequencies."""
    return precompute_rope_frequencies(
        head_dim=HEAD_DIM,
        max_seq_len=SEQ_LEN,
        theta=THETA,
        rope_scaling_type=scaling_type,
        rope_scaling_factor=factor,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_yarn_frequencies_shape():
    """YaRN produces a complex freq tensor of the correct shape."""
    freqs = _freqs(
        scaling_type="yarn",
        factor=16.0,
        yarn_original_max_seq_len=8192,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
    )
    assert freqs.shape == (SEQ_LEN, HEAD_DIM // 2), (
        f"Expected ({SEQ_LEN}, {HEAD_DIM // 2}), got {freqs.shape}"
    )
    assert freqs.is_complex(), "Frequency tensor must be complex-valued"


def test_yarn_no_scaling_unchanged():
    """rope_scaling_type='none' must produce identical output to vanilla RoPE."""
    # Vanilla call (no kwargs at all)
    freqs_vanilla = precompute_rope_frequencies(
        head_dim=HEAD_DIM,
        max_seq_len=SEQ_LEN,
        theta=THETA,
    )
    # Explicit "none"
    freqs_none = _freqs(scaling_type="none", factor=1.0)

    assert torch.allclose(freqs_vanilla, freqs_none), (
        "rope_scaling_type='none' must be identical to the vanilla (no-scaling) call"
    )


def test_yarn_ntk_shifts_base():
    """NTK scaling must produce frequencies different from unscaled RoPE."""
    freqs_base = _freqs(scaling_type="none", factor=1.0)
    freqs_ntk = _freqs(scaling_type="ntk", factor=16.0)

    assert not torch.allclose(freqs_base, freqs_ntk), (
        "NTK scaling should change frequencies relative to unscaled RoPE"
    )


def test_yarn_midband_blend():
    """YaRN mid-band frequencies must be strictly between linear and NTK values.

    We test the raw (pre-complex-exponentiation) frequencies by calling the
    helper with max_seq_len=1 and extracting the first position's angle
    divided by the position index (which equals the raw freq for pos=1).

    With factor=16, beta_fast=32, beta_slow=1 and head_dim=64:
      high_freq_thresh = 2π*32/16 ≈ 12.57
      low_freq_thresh  = 2π*1/16  ≈  0.39
    Dimension 0 (freq=1.0, lambda=2π≈6.28) and dimension 2 fall in the mid-band.
    Their blended frequencies must be strictly between linear and NTK values.
    """
    import math

    factor = 16.0
    theta = 500_000.0
    head_dim = HEAD_DIM

    # Compute raw frequencies for each scaling type by reading the per-dim freq
    # directly: at position p=1, angle = p * freq = freq.
    def raw_freqs(stype, **kw):
        fc = precompute_rope_frequencies(
            head_dim=head_dim, max_seq_len=2, theta=theta,
            rope_scaling_type=stype, rope_scaling_factor=factor, **kw,
        )
        # pos=1: angle = freq (since outer(1, freqs) = freqs)
        return fc[1].angle()

    f_linear = raw_freqs("linear")
    f_ntk = raw_freqs("ntk")
    f_yarn = raw_freqs("yarn", yarn_original_max_seq_len=8192,
                        yarn_beta_fast=32.0, yarn_beta_slow=1.0)

    # YaRN must differ from both endpoints
    assert not torch.allclose(f_yarn, f_linear), "YaRN should differ from pure linear"
    assert not torch.allclose(f_yarn, f_ntk), "YaRN should differ from pure NTK"

    # For mid-band dims, |yarn - ntk| < |linear - ntk| (yarn is between the two)
    dist_yarn_to_ntk = (f_yarn - f_ntk).abs()
    dist_linear_to_ntk = (f_linear - f_ntk).abs()
    # At least one mid-band dim: YaRN is closer to NTK than linear is
    # (blend=0 would match NTK exactly; blend=1 would match linear exactly)
    mid_band_dims_exist = (dist_yarn_to_ntk < dist_linear_to_ntk).any()
    assert mid_band_dims_exist, (
        "No dimension shows YaRN blending between linear and NTK — mid-band blend may be dead code"
    )


def test_yarn_config_fields():
    """AureliusConfig exposes all 5 new YaRN fields with correct defaults."""
    cfg = AureliusConfig()

    assert hasattr(cfg, "rope_scaling_type"), "Missing field: rope_scaling_type"
    assert hasattr(cfg, "rope_scaling_factor"), "Missing field: rope_scaling_factor"
    assert hasattr(cfg, "yarn_original_max_seq_len"), "Missing field: yarn_original_max_seq_len"
    assert hasattr(cfg, "yarn_beta_fast"), "Missing field: yarn_beta_fast"
    assert hasattr(cfg, "yarn_beta_slow"), "Missing field: yarn_beta_slow"

    assert cfg.rope_scaling_type == "none", (
        f"Default rope_scaling_type should be 'none', got {cfg.rope_scaling_type!r}"
    )
    assert cfg.rope_scaling_factor == 1.0, (
        f"Default rope_scaling_factor should be 1.0, got {cfg.rope_scaling_factor}"
    )
    assert cfg.yarn_original_max_seq_len == 8192, (
        f"Default yarn_original_max_seq_len should be 8192, got {cfg.yarn_original_max_seq_len}"
    )
    assert cfg.yarn_beta_fast == 32.0, (
        f"Default yarn_beta_fast should be 32.0, got {cfg.yarn_beta_fast}"
    )
    assert cfg.yarn_beta_slow == 1.0, (
        f"Default yarn_beta_slow should be 1.0, got {cfg.yarn_beta_slow}"
    )
