"""Tests for src/compression/gradient_compression.py — 8+ tests."""
import pytest
import torch
from src.compression.gradient_compression import (
    GradCompressMethod,
    GradCompressionConfig,
    GradientCompressor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grad_4x4():
    torch.manual_seed(0)
    return torch.randn(4, 4)


@pytest.fixture
def top_k_compressor():
    cfg = GradCompressionConfig(method=GradCompressMethod.TOP_K, sparsity=0.75)
    return GradientCompressor(cfg)


@pytest.fixture
def sign_compressor():
    cfg = GradCompressionConfig(method=GradCompressMethod.SIGN_SGD)
    return GradientCompressor(cfg)


# ---------------------------------------------------------------------------
# 1. TOP_K
# ---------------------------------------------------------------------------

def test_top_k_keeps_correct_count(top_k_compressor, grad_4x4):
    values, indices = top_k_compressor.compress(grad_4x4)
    n = grad_4x4.numel()  # 16
    expected_k = max(1, int(round(n * (1 - 0.75))))  # 4
    assert values.numel() == expected_k
    assert indices.numel() == expected_k


def test_top_k_selects_largest_magnitudes(top_k_compressor, grad_4x4):
    values, indices = top_k_compressor.compress(grad_4x4)
    flat = grad_4x4.reshape(-1)
    # All kept values should be among the top-k by magnitude
    kept_mags = values.abs()
    min_kept = kept_mags.min().item()
    # All discarded values should have smaller or equal magnitude
    mask = torch.ones(flat.numel(), dtype=torch.bool)
    mask[indices] = False
    discarded_mags = flat[mask].abs()
    if discarded_mags.numel() > 0:
        assert discarded_mags.max().item() <= min_kept + 1e-6


def test_decompress_top_k_restores_shape(top_k_compressor, grad_4x4):
    values, indices = top_k_compressor.compress(grad_4x4)
    out = top_k_compressor.decompress(values, indices, grad_4x4.shape)
    assert out.shape == grad_4x4.shape


def test_decompress_top_k_zeros_pruned(top_k_compressor, grad_4x4):
    values, indices = top_k_compressor.compress(grad_4x4)
    out = top_k_compressor.decompress(values, indices, grad_4x4.shape)
    flat_out = out.reshape(-1)
    kept_set = set(indices.tolist())
    for i in range(flat_out.numel()):
        if i not in kept_set:
            assert flat_out[i].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. RANDOM_K
# ---------------------------------------------------------------------------

def test_random_k_count(grad_4x4):
    cfg = GradCompressionConfig(method=GradCompressMethod.RANDOM_K, sparsity=0.5)
    c = GradientCompressor(cfg)
    values, indices = c.compress(grad_4x4)
    expected_k = max(1, int(round(16 * 0.5)))
    assert values.numel() == expected_k


def test_random_k_decompress_shape(grad_4x4):
    cfg = GradCompressionConfig(method=GradCompressMethod.RANDOM_K, sparsity=0.5)
    c = GradientCompressor(cfg)
    values, indices = c.compress(grad_4x4)
    out = c.decompress(values, indices, grad_4x4.shape)
    assert out.shape == grad_4x4.shape


# ---------------------------------------------------------------------------
# 3. THRESHOLD
# ---------------------------------------------------------------------------

def test_threshold_keeps_large_values():
    cfg = GradCompressionConfig(method=GradCompressMethod.THRESHOLD, threshold=0.5)
    c = GradientCompressor(cfg)
    # Craft a tensor where we know which values exceed the threshold
    g = torch.tensor([[0.0, 0.6, 0.1, 0.8], [0.0, 0.0, 0.9, 0.2]], dtype=torch.float32)
    values, indices = c.compress(g)
    kept_vals = values.abs()
    assert (kept_vals > 0.5).all()


def test_threshold_fallback_to_max_when_none_pass():
    cfg = GradCompressionConfig(method=GradCompressMethod.THRESHOLD, threshold=100.0)
    c = GradientCompressor(cfg)
    g = torch.ones(4, 4) * 0.1
    values, indices = c.compress(g)
    # Should not be empty (fallback keeps the single max)
    assert values.numel() >= 1


# ---------------------------------------------------------------------------
# 4. SIGN_SGD
# ---------------------------------------------------------------------------

def test_sign_sgd_all_elements_returned(sign_compressor, grad_4x4):
    values, indices = sign_compressor.compress(grad_4x4)
    assert values.numel() == grad_4x4.numel()
    assert indices.numel() == grad_4x4.numel()


def test_sign_sgd_values_are_signs(sign_compressor, grad_4x4):
    values, _ = sign_compressor.compress(grad_4x4)
    # All values should be -1, 0, or 1
    assert (values.abs() <= 1.0).all()
    flat = grad_4x4.reshape(-1)
    expected_signs = flat.sign()
    assert torch.allclose(values, expected_signs)


# ---------------------------------------------------------------------------
# 5. Enum values
# ---------------------------------------------------------------------------

def test_enum_values():
    assert GradCompressMethod.TOP_K == "top_k"
    assert GradCompressMethod.SIGN_SGD == "sign_sgd"
