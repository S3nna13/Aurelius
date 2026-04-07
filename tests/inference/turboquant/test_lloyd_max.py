"""Tests for Lloyd-Max codebook for Beta(2,2)."""
import torch
import pytest
from src.inference.turboquant.lloyd_max import compute_lloyd_max_codebook, _CODEBOOK_CACHE


def test_codebook_length():
    cb = compute_lloyd_max_codebook(8)
    assert len(cb) == 8


def test_codebook_sorted():
    cb = compute_lloyd_max_codebook(16)
    assert (cb[1:] >= cb[:-1]).all(), "Codebook must be sorted ascending"


def test_codebook_in_unit_interval():
    cb = compute_lloyd_max_codebook(32)
    assert cb.min() >= 0.0
    assert cb.max() <= 1.0


def test_codebook_returns_tensor():
    cb = compute_lloyd_max_codebook(4)
    assert isinstance(cb, torch.Tensor)
    assert cb.dtype == torch.float32


def test_codebook_cached():
    """Second call returns the same object (module-level cache)."""
    _CODEBOOK_CACHE.clear()
    cb1 = compute_lloyd_max_codebook(8)
    cb2 = compute_lloyd_max_codebook(8)
    assert cb1 is cb2


def test_codebook_deterministic():
    """Same n_codes always gives same codebook."""
    _CODEBOOK_CACHE.clear()
    cb1 = compute_lloyd_max_codebook(16)
    _CODEBOOK_CACHE.clear()
    cb2 = compute_lloyd_max_codebook(16)
    assert torch.allclose(cb1, cb2, atol=1e-6)


def test_codebook_symmetric():
    """Beta(2,2) is symmetric around 0.5, so codebook should be roughly symmetric."""
    cb = compute_lloyd_max_codebook(8)
    # Mirror: codebook[i] + codebook[n-1-i] should be close to 1.0
    mirror_sum = cb + cb.flip(0)
    assert torch.allclose(mirror_sum, torch.ones(8), atol=1e-4), \
        f"Codebook not symmetric: {mirror_sum}"


def test_different_sizes_independent():
    cb8 = compute_lloyd_max_codebook(8)
    cb16 = compute_lloyd_max_codebook(16)
    assert len(cb8) == 8
    assert len(cb16) == 16
