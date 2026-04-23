"""Tests for activation compression."""
from __future__ import annotations

import pytest
from src.compression.activation_compression import (
    SparsificationMethod,
    ActivationCompressor,
    ACTIVATION_COMPRESSOR,
)


# --- SparsificationMethod enum ---

def test_sparsification_method_topk():
    assert SparsificationMethod.TOPK == "topk"

def test_sparsification_method_threshold():
    assert SparsificationMethod.THRESHOLD == "threshold"

def test_sparsification_method_random_dropout():
    assert SparsificationMethod.RANDOM_DROPOUT == "random_dropout"


# --- topk_sparsify ---

def test_topk_sparsify_exactly_k_nonzero():
    ac = ActivationCompressor()
    result = ac.topk_sparsify([1.0, -5.0, 3.0, 0.5, 2.0], k=2)
    nonzero = sum(1 for v in result if abs(v) > 1e-8)
    assert nonzero == 2

def test_topk_sparsify_keeps_top_by_abs():
    ac = ActivationCompressor()
    activations = [1.0, -5.0, 3.0, 0.5, 2.0]
    result = ac.topk_sparsify(activations, k=2)
    # -5.0 and 3.0 are top-2 by abs value
    assert result[1] == -5.0
    assert result[2] == 3.0

def test_topk_sparsify_zeroes_rest():
    ac = ActivationCompressor()
    result = ac.topk_sparsify([1.0, 2.0, 3.0, 4.0, 5.0], k=2)
    # Only 2 non-zero
    zero_count = sum(1 for v in result if abs(v) < 1e-8)
    assert zero_count == 3

def test_topk_sparsify_k_equal_len():
    ac = ActivationCompressor()
    activations = [1.0, 2.0, 3.0]
    result = ac.topk_sparsify(activations, k=3)
    assert result == activations

def test_topk_sparsify_k_zero():
    ac = ActivationCompressor()
    result = ac.topk_sparsify([1.0, 2.0, 3.0], k=0)
    assert all(v == 0.0 for v in result)

def test_topk_sparsify_same_length():
    ac = ActivationCompressor()
    activations = [1.0, -2.0, 3.0, 0.1]
    result = ac.topk_sparsify(activations, k=2)
    assert len(result) == len(activations)

def test_topk_sparsify_empty():
    ac = ActivationCompressor()
    assert ac.topk_sparsify([], k=3) == []


# --- threshold_sparsify ---

def test_threshold_sparsify_below_threshold_zeroed():
    ac = ActivationCompressor()
    result = ac.threshold_sparsify([0.1, 0.5, 1.0, 0.3], threshold=0.5)
    assert result[0] == 0.0
    assert result[3] == 0.0

def test_threshold_sparsify_above_threshold_kept():
    ac = ActivationCompressor()
    result = ac.threshold_sparsify([0.1, 0.5, 1.0, 0.3], threshold=0.5)
    assert result[1] == 0.5
    assert result[2] == 1.0

def test_threshold_sparsify_negative_values():
    ac = ActivationCompressor()
    result = ac.threshold_sparsify([-1.0, -0.2, 0.8], threshold=0.5)
    assert result[0] == -1.0
    assert result[1] == 0.0
    assert result[2] == 0.8

def test_threshold_sparsify_same_length():
    ac = ActivationCompressor()
    activations = [1.0, 2.0, 0.1]
    assert len(ac.threshold_sparsify(activations, 0.5)) == len(activations)


# --- quantize_fp8 ---

def test_quantize_fp8_clamped_positive():
    ac = ActivationCompressor()
    result = ac.quantize_fp8([1000.0])
    assert result[0] <= 448.0

def test_quantize_fp8_clamped_negative():
    ac = ActivationCompressor()
    result = ac.quantize_fp8([-1000.0])
    assert result[0] >= -448.0

def test_quantize_fp8_rounded_to_step():
    ac = ActivationCompressor()
    result = ac.quantize_fp8([1.03])
    # nearest 0.0625 to 1.03 is 1.0
    assert abs(result[0] % 0.0625) < 1e-6 or abs(result[0] % 0.0625 - 0.0625) < 1e-6

def test_quantize_fp8_zero():
    ac = ActivationCompressor()
    result = ac.quantize_fp8([0.0])
    assert result[0] == 0.0

def test_quantize_fp8_exact_boundary():
    ac = ActivationCompressor()
    result = ac.quantize_fp8([448.0])
    assert result[0] == 448.0

def test_quantize_fp8_same_length():
    ac = ActivationCompressor()
    values = [1.0, 2.0, 3.0]
    assert len(ac.quantize_fp8(values)) == len(values)

def test_quantize_fp8_small_value():
    ac = ActivationCompressor()
    result = ac.quantize_fp8([0.0625])
    assert abs(result[0] - 0.0625) < 1e-9


# --- sparsity_ratio ---

def test_sparsity_ratio_all_zero():
    ac = ActivationCompressor()
    assert ac.sparsity_ratio([0.0, 0.0, 0.0]) == 1.0

def test_sparsity_ratio_no_zero():
    ac = ActivationCompressor()
    assert ac.sparsity_ratio([1.0, 2.0, 3.0]) == 0.0

def test_sparsity_ratio_half_zero():
    ac = ActivationCompressor()
    ratio = ac.sparsity_ratio([0.0, 0.0, 1.0, 1.0])
    assert abs(ratio - 0.5) < 1e-9

def test_sparsity_ratio_empty():
    ac = ActivationCompressor()
    assert ac.sparsity_ratio([]) == 0.0

def test_sparsity_ratio_near_zero():
    ac = ActivationCompressor()
    ratio = ac.sparsity_ratio([1e-9, 1.0])
    assert ratio == 0.5  # 1e-9 < 1e-8 threshold


# --- compress ---

def test_compress_topk_returns_same_length():
    ac = ActivationCompressor()
    activations = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = ac.compress(activations, method=SparsificationMethod.TOPK, k=3)
    assert len(result) == len(activations)

def test_compress_topk_correct_nonzero():
    ac = ActivationCompressor()
    result = ac.compress([1.0, 2.0, 3.0, 4.0, 5.0], method=SparsificationMethod.TOPK, k=2)
    nonzero = sum(1 for v in result if abs(v) > 1e-8)
    assert nonzero == 2

def test_compress_threshold_method():
    ac = ActivationCompressor()
    result = ac.compress([0.1, 0.8, 0.3], method=SparsificationMethod.THRESHOLD, threshold=0.5)
    assert result[0] == 0.0
    assert result[1] == 0.8

def test_compress_random_dropout_same_length():
    ac = ActivationCompressor()
    activations = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = ac.compress(activations, method=SparsificationMethod.RANDOM_DROPOUT)
    assert len(result) == len(activations)


# --- memory_savings_estimate ---

def test_memory_savings_half_sparsity():
    ac = ActivationCompressor()
    result = ac.memory_savings_estimate(1000, 0.5)
    assert abs(result - 0.5) < 1e-9

def test_memory_savings_zero_sparsity():
    ac = ActivationCompressor()
    result = ac.memory_savings_estimate(1000, 0.0)
    assert abs(result - 1.0) < 1e-9

def test_memory_savings_full_sparsity():
    ac = ActivationCompressor()
    result = ac.memory_savings_estimate(1000, 1.0)
    assert abs(result - 0.0) < 1e-9

def test_memory_savings_zero_size():
    ac = ActivationCompressor()
    result = ac.memory_savings_estimate(0, 0.5)
    assert result == 0.0


# --- ACTIVATION_COMPRESSOR singleton ---

def test_activation_compressor_exists():
    assert ACTIVATION_COMPRESSOR is not None

def test_activation_compressor_is_instance():
    assert isinstance(ACTIVATION_COMPRESSOR, ActivationCompressor)

def test_activation_compressor_default_sparsity():
    assert ACTIVATION_COMPRESSOR.default_sparsity == 0.5
