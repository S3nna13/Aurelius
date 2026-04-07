"""Tests for TurboQuant compressor and KV backend."""
import torch
import pytest
from src.inference.turboquant.compressor import TurboQuantCompressor, CompressedKV
from src.inference.turboquant.kv_backend import CompressedKVCache


@pytest.fixture
def compressor():
    return TurboQuantCompressor(dim=32, n_codes=16, sketch_dim=16, seed=0)


def test_compress_returns_compressedkv(compressor):
    x = torch.randn(2, 4, 32)
    ckv = compressor.compress(x)
    assert isinstance(ckv, CompressedKV)


def test_compress_polar_state_present(compressor):
    x = torch.randn(2, 4, 32)
    ckv = compressor.compress(x)
    assert ckv.polar_state is not None
    assert ckv.polar_state.codes.shape == (2, 4, 32)


def test_compress_residual_sketch_present(compressor):
    x = torch.randn(2, 4, 32)
    ckv = compressor.compress(x)
    assert ckv.residual_signs.shape == (2, 4, 16)
    assert ckv.residual_norms.shape == (2, 4)


def test_decompress_shape(compressor):
    x = torch.randn(3, 8, 32)
    ckv = compressor.compress(x)
    x_hat = compressor.decompress(ckv)
    assert x_hat.shape == x.shape


def test_kv_backend_update_and_get():
    cache = CompressedKVCache(n_layers=2, head_dim=32, n_kv_heads=2, n_codes=16, sketch_dim=16)
    k = torch.randn(1, 4, 2, 32)
    v = torch.randn(1, 4, 2, 32)
    cache.update(0, k, v)
    result = cache.get_decompressed(0)
    assert result is not None
    k_hat, v_hat = result
    assert k_hat.shape == k.shape
    assert v_hat.shape == v.shape


def test_kv_backend_empty_before_update():
    cache = CompressedKVCache(n_layers=2, head_dim=32, n_kv_heads=2, n_codes=16, sketch_dim=16)
    assert cache.is_empty(0)
    assert cache.is_empty(1)


def test_kv_backend_not_empty_after_update():
    cache = CompressedKVCache(n_layers=2, head_dim=32, n_kv_heads=2, n_codes=16, sketch_dim=16)
    k = torch.randn(1, 2, 2, 32)
    v = torch.randn(1, 2, 2, 32)
    cache.update(0, k, v)
    assert not cache.is_empty(0)
    assert cache.is_empty(1)  # layer 1 still empty


def test_kv_backend_clear():
    cache = CompressedKVCache(n_layers=2, head_dim=32, n_kv_heads=2, n_codes=16, sketch_dim=16)
    k = torch.randn(1, 2, 2, 32)
    v = torch.randn(1, 2, 2, 32)
    cache.update(0, k, v)
    cache.clear()
    assert cache.is_empty(0)


def test_turboquant_exports():
    from src.inference.turboquant import (
        compute_lloyd_max_codebook, PolarQuant, PolarQuantState,
        QJLSketch, TurboQuantCompressor, CompressedKV, CompressedKVCache,
    )
