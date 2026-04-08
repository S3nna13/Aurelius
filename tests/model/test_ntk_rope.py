import pytest
import torch
import math
from src.model.ntk_rope import (
    NTKRoPEConfig, ntk_scaled_theta, ntk_rope_frequencies, dynamic_ntk_frequencies
)

def test_ntk_scaled_theta_increases():
    """Scaled theta must be >= original theta for scale_factor >= 1."""
    theta = ntk_scaled_theta(500_000.0, 4.0, 128)
    assert theta > 500_000.0

def test_ntk_scaled_theta_scale1_unchanged():
    """scale_factor=1.0 should return original theta (1^anything = 1)."""
    theta = ntk_scaled_theta(500_000.0, 1.0, 128)
    assert theta == pytest.approx(500_000.0)

def test_ntk_frequencies_shape():
    freqs = ntk_rope_frequencies(head_dim=64, max_seq_len=128, scale_factor=2.0)
    assert freqs.shape == (128, 32)
    assert freqs.is_complex()

def test_ntk_frequencies_unit_magnitude():
    freqs = ntk_rope_frequencies(head_dim=64, max_seq_len=32, scale_factor=1.0)
    magnitudes = freqs.abs()
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)

def test_dynamic_ntk_no_scale_within_limit():
    """Within original context, scale_factor=1.0 → same as standard RoPE."""
    from src.model.ntk_rope import ntk_rope_frequencies
    freqs_dynamic = dynamic_ntk_frequencies(64, current_seq_len=100, original_max_seq_len=200)
    freqs_standard = ntk_rope_frequencies(64, 100, scale_factor=1.0)
    assert torch.allclose(freqs_dynamic.abs(), freqs_standard.abs(), atol=1e-5)

def test_dynamic_ntk_scales_beyond_limit():
    freqs = dynamic_ntk_frequencies(64, current_seq_len=400, original_max_seq_len=100)
    assert freqs.shape == (400, 32)

def test_ntk_frequencies_match_formula():
    """Verify theta scaling formula: new_theta = theta * scale^(dim/(dim-2))."""
    head_dim = 64
    scale = 4.0
    theta = 10_000.0
    expected_theta = theta * (scale ** (head_dim / (head_dim - 2)))
    actual = ntk_scaled_theta(theta, scale, head_dim)
    assert actual == pytest.approx(expected_theta, rel=1e-6)
