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
