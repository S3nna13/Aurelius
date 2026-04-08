"""Tests for src/inference/kv_quant.py.

Covers quantize/dequantize accuracy, dtype checks, KV cache round-trip,
and memory savings estimation.
"""
from __future__ import annotations

import torch
import pytest

from src.inference.kv_quant import KVQuantizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, SEQ_LEN, N_KV_HEADS, HEAD_DIM = 2, 64, 4, 64
N_LAYERS = 12


@pytest.fixture(scope="module")
def sample_tensor() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, SEQ_LEN, N_KV_HEADS, HEAD_DIM)


@pytest.fixture(scope="module")
def kv_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    k = torch.randn(B, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
    v = torch.randn(B, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
    return k, v


# ---------------------------------------------------------------------------
# Quantize / dequantize core tests
# ---------------------------------------------------------------------------

class TestKVQuantizerCore:

    def test_quantize_shape(self, sample_tensor):
        """Quantized tensor has the same shape as input (last dim padded to group multiple)."""
        quantizer = KVQuantizer(bits=8, group_size=64, symmetric=True)
        quantized, scales, _ = quantizer.quantize(sample_tensor)
        # Last dim: HEAD_DIM=64 == group_size -> no padding -> same shape
        assert quantized.shape == sample_tensor.shape, (
            f"Expected shape {sample_tensor.shape}, got {quantized.shape}"
        )

    def test_quantize_int8_dtype(self, sample_tensor):
        """Symmetric quantization stores as int8; asymmetric as uint8."""
        quantizer_sym = KVQuantizer(bits=8, group_size=64, symmetric=True)
        q_sym, _, _ = quantizer_sym.quantize(sample_tensor)
        assert q_sym.dtype == torch.int8, f"Expected int8, got {q_sym.dtype}"

        quantizer_asym = KVQuantizer(bits=8, group_size=64, symmetric=False)
        q_asym, _, _ = quantizer_asym.quantize(sample_tensor)
        assert q_asym.dtype == torch.uint8, f"Expected uint8, got {q_asym.dtype}"

    def test_dequantize_close(self, sample_tensor):
        """dequantize(quantize(x)) is approximately equal to x within INT8 error."""
        quantizer = KVQuantizer(bits=8, group_size=64, symmetric=True)
        quantized, scales, zp = quantizer.quantize(sample_tensor)
        reconstructed = quantizer.dequantize(
            quantized, scales, zp, original_shape=tuple(sample_tensor.shape)
        )
        # INT8 symmetric: max relative error should be < 2% for standard-normal data
        max_abs = sample_tensor.abs().max().item()
        max_err = (sample_tensor - reconstructed).abs().max().item()
        assert max_err < 0.02 * max_abs, (
            f"Roundtrip error {max_err:.6f} exceeds 2% of max={max_abs:.4f}"
        )

    def test_symmetric_no_zero_point(self, sample_tensor):
        """Symmetric quantization returns zero_points=None."""
        quantizer = KVQuantizer(bits=8, group_size=64, symmetric=True)
        _, _, zero_points = quantizer.quantize(sample_tensor)
        assert zero_points is None, f"Expected None for symmetric ZP, got {zero_points}"

    def test_asymmetric_has_zero_point(self, sample_tensor):
        """Asymmetric quantization returns a zero_points tensor."""
        quantizer = KVQuantizer(bits=8, group_size=64, symmetric=False)
        _, _, zero_points = quantizer.quantize(sample_tensor)
        assert zero_points is not None, "Expected zero_points tensor for asymmetric"
        assert isinstance(zero_points, torch.Tensor), (
            f"zero_points should be a Tensor, got {type(zero_points)}"
        )


# ---------------------------------------------------------------------------
# KV cache API
# ---------------------------------------------------------------------------

class TestKVCacheAPI:

    def test_quantize_kv_cache_keys(self, kv_pair):
        """quantize_kv_cache returns a dict with required keys."""
        k, v = kv_pair
        quantizer = KVQuantizer(bits=8, group_size=64, symmetric=True)
        cache = quantizer.quantize_kv_cache(k, v)
        required_keys = {"k_q", "k_scales", "v_q", "v_scales"}
        missing = required_keys - cache.keys()
        assert not missing, f"Cache dict missing keys: {missing}"

    def test_dequantize_kv_cache_shapes(self, kv_pair):
        """Restored K and V have the original shape."""
        k, v = kv_pair
        original_shape = tuple(k.shape)
        quantizer = KVQuantizer(bits=8, group_size=64, symmetric=True)
        cache = quantizer.quantize_kv_cache(k, v)
        k_out, v_out = quantizer.dequantize_kv_cache(cache)
        assert k_out.shape == original_shape, (
            f"K shape mismatch: expected {original_shape}, got {k_out.shape}"
        )
        assert v_out.shape == original_shape, (
            f"V shape mismatch: expected {original_shape}, got {v_out.shape}"
        )


# ---------------------------------------------------------------------------
# Memory analysis
# ---------------------------------------------------------------------------

class TestMemorySavings:

    def test_memory_saved_mb_savings(self):
        """INT8 quantization yields savings_factor > 1 versus FP16."""
        quantizer = KVQuantizer(bits=8, group_size=128, symmetric=True)
        result = quantizer.memory_saved_mb(
            seq_len=SEQ_LEN,
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            n_layers=N_LAYERS,
        )
        assert "fp16_mb" in result, "Missing 'fp16_mb' key"
        assert "quantized_mb" in result, "Missing 'quantized_mb' key"
        assert "savings_factor" in result, "Missing 'savings_factor' key"
        assert result["savings_factor"] > 1.0, (
            f"Expected savings_factor > 1, got {result['savings_factor']:.4f}"
        )
