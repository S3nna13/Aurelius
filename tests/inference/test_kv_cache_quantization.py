"""Tests for KV-cache quantization helpers."""

import pytest
import torch

from src.inference.kv_cache_quantization import (
    dequantize_kv_cache,
    kv_cache_quantization_error,
    quantize_kv_cache,
)


def make_cache():
    key = torch.randn(2, 5, 3, 4)
    value = torch.randn(2, 5, 3, 4)
    return key, value


def test_quantize_kv_cache_preserves_shape():
    key, value = make_cache()
    cache = quantize_kv_cache(key, value)
    assert cache.key_q.shape == key.shape
    assert cache.value_q.shape == value.shape


def test_quantize_kv_cache_outputs_int8():
    key, value = make_cache()
    cache = quantize_kv_cache(key, value)
    assert cache.key_q.dtype == torch.int8
    assert cache.value_q.dtype == torch.int8


def test_dequantize_kv_cache_round_trip_is_close():
    key, value = make_cache()
    cache = quantize_kv_cache(key, value)
    key_hat, value_hat = dequantize_kv_cache(cache)
    assert torch.allclose(key_hat, key, atol=0.05)
    assert torch.allclose(value_hat, value, atol=0.05)


def test_quantization_error_is_small():
    key, value = make_cache()
    key_err, value_err = kv_cache_quantization_error(key, value)
    assert key_err.item() < 0.05
    assert value_err.item() < 0.05


def test_quantization_handles_zero_cache():
    zeros = torch.zeros(1, 2, 3, 4)
    cache = quantize_kv_cache(zeros, zeros)
    key_hat, value_hat = dequantize_kv_cache(cache)
    assert torch.equal(key_hat, zeros)
    assert torch.equal(value_hat, zeros)


def test_quantize_kv_cache_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        quantize_kv_cache(torch.randn(1, 2, 3, 4), torch.randn(1, 3, 3, 4))


def test_quantize_kv_cache_rejects_bad_rank():
    with pytest.raises(ValueError):
        quantize_kv_cache(torch.randn(2, 3, 4), torch.randn(2, 3, 4))
