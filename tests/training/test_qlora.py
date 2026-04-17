"""Tests for QLoRA: Efficient Fine-tuning of Quantized LLMs (arXiv:2305.14314).

Covers:
  - NF4 level properties (count, range, symmetry, monotonicity)
  - NF4Quantizer.quantize / dequantize round-trip fidelity
  - QLoRALinear construction, gradient flow, forward output shape
  - from_linear constructor
  - Determinism, NaN/Inf safety, edge cases
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.qlora import (
    NF4_LEVELS,
    NF4Quantizer,
    QLoRALinear,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(out: int, in_: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(out, in_, generator=g)


def _make_linear(out: int, in_: int, seed: int = 42) -> nn.Linear:
    g = torch.Generator()
    g.manual_seed(seed)
    linear = nn.Linear(in_, out, bias=True)
    with torch.no_grad():
        nn.init.normal_(linear.weight, generator=g)
        nn.init.zeros_(linear.bias)
    return linear


# ---------------------------------------------------------------------------
# Test 1: NF4_LEVELS has exactly 16 values
# ---------------------------------------------------------------------------
def test_nf4_levels_count():
    """NF4_LEVELS must have exactly 16 quantization levels (4-bit → 2^4)."""
    assert NF4_LEVELS.numel() == 16, (
        f"Expected 16 NF4 levels, got {NF4_LEVELS.numel()}"
    )


# ---------------------------------------------------------------------------
# Test 2: NF4_LEVELS are sorted and lie in [-1, 1]
# ---------------------------------------------------------------------------
def test_nf4_levels_range_and_sorted():
    """All NF4 levels must be in [-1, 1] and sorted in ascending order."""
    levels = NF4_LEVELS.float()
    assert levels.min().item() >= -1.0 - 1e-6, "NF4 level below -1"
    assert levels.max().item() <= 1.0 + 1e-6, "NF4 level above +1"
    diffs = levels[1:] - levels[:-1]
    assert (diffs > 0).all(), "NF4 levels are not strictly sorted"


# ---------------------------------------------------------------------------
# Test 3: NF4_LEVELS are approximately symmetric
# ---------------------------------------------------------------------------
def test_nf4_levels_symmetric():
    """NF4 levels should satisfy levels[i] ≈ -levels[15-i] (normal symmetry)."""
    levels = NF4_LEVELS.float()
    for i in range(16):
        a = levels[i].item()
        b = levels[15 - i].item()
        assert abs(a + b) < 0.05, (
            f"Symmetry violated at index {i}: {a:.4f} + {b:.4f} = {a+b:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4: quantize → codes dtype is uint8, values in [0, 15]
# ---------------------------------------------------------------------------
def test_quantize_codes_dtype_and_range():
    """Codes tensor must be uint8 with values in [0, 15]."""
    w = _make_weight(32, 64)
    codes, scales = NF4Quantizer.quantize(w, block_size=64)
    assert codes.dtype == torch.uint8, f"Expected uint8, got {codes.dtype}"
    assert codes.shape == w.shape, f"Codes shape mismatch: {codes.shape} vs {w.shape}"
    assert int(codes.min().item()) >= 0, "Code value below 0"
    assert int(codes.max().item()) <= 15, "Code value above 15"


# ---------------------------------------------------------------------------
# Test 5: scales shape is (n_blocks,)
# ---------------------------------------------------------------------------
def test_quantize_scales_shape():
    """scales must have shape (n_blocks,) where n_blocks = ceil(N / block_size)."""
    out, in_ = 32, 64
    block_size = 64
    w = _make_weight(out, in_)
    _, scales = NF4Quantizer.quantize(w, block_size=block_size)
    N = out * in_
    expected_n_blocks = math.ceil(N / block_size)
    assert scales.shape == (expected_n_blocks,), (
        f"Expected scales shape ({expected_n_blocks},), got {scales.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: round-trip reconstruction error < 0.1 for normal weights in [-1, 1]
# ---------------------------------------------------------------------------
def test_roundtrip_error_small():
    """quantize → dequantize must reconstruct weights with mean abs error < 0.1."""
    torch.manual_seed(7)
    w = torch.randn(64, 128)
    # Normalize to [-1, 1] to satisfy the NF4 assumption
    w = w / w.abs().max()

    codes, scales = NF4Quantizer.quantize(w, block_size=64)
    w_deq = NF4Quantizer.dequantize(codes, scales, original_shape=w.shape, block_size=64)
    mae = (w - w_deq).abs().mean().item()
    assert mae < 0.1, f"Round-trip MAE {mae:.4f} exceeds 0.1 threshold"


# ---------------------------------------------------------------------------
# Test 7: quantize → dequantize low error for normally distributed weights
# ---------------------------------------------------------------------------
def test_roundtrip_normal_dist():
    """For N(0,1) weights, quantize→dequantize should keep RMSE reasonably low."""
    torch.manual_seed(99)
    w = torch.randn(128, 256)
    codes, scales = NF4Quantizer.quantize(w, block_size=64)
    w_deq = NF4Quantizer.dequantize(codes, scales, original_shape=w.shape, block_size=64)
    rmse = ((w - w_deq) ** 2).mean().sqrt().item()
    # NF4 is good for normal weights; expect RMSE well below 0.3 with block_size=64
    assert rmse < 0.3, f"Round-trip RMSE {rmse:.4f} too large for normally distributed weights"


# ---------------------------------------------------------------------------
# Test 8: block_size=1 gives (near-)perfect reconstruction
# ---------------------------------------------------------------------------
def test_roundtrip_block_size_1():
    """With block_size=1, each element is its own block → near-perfect reconstruction."""
    torch.manual_seed(11)
    w = torch.randn(8, 8)
    codes, scales = NF4Quantizer.quantize(w, block_size=1)
    w_deq = NF4Quantizer.dequantize(codes, scales, original_shape=w.shape, block_size=1)
    mae = (w - w_deq).abs().max().item()
    # Each element scaled independently; residual error is only the 4-bit level spacing
    # relative to max(|w|)=scale, which should be tiny since scale absorbs magnitude.
    assert mae < 0.5, f"Per-element quantization gave MAE {mae:.4f}"


# ---------------------------------------------------------------------------
# Test 9: QLoRALinear output shape matches nn.Linear
# ---------------------------------------------------------------------------
def test_qlora_output_shape():
    """QLoRALinear forward must produce the same output shape as nn.Linear."""
    linear = _make_linear(out=32, in_=64)
    qlora = QLoRALinear.from_linear(linear, r=4)
    x = torch.randn(3, 10, 64)
    out = qlora(x)
    assert out.shape == (3, 10, 32), f"Unexpected output shape {out.shape}"


# ---------------------------------------------------------------------------
# Test 10: QLoRALinear base weight is frozen (no gradient)
# ---------------------------------------------------------------------------
def test_qlora_base_weight_frozen():
    """W_codes and W_scales must have requires_grad=False (frozen NF4 base weight)."""
    linear = _make_linear(out=32, in_=64)
    qlora = QLoRALinear.from_linear(linear, r=4)
    assert not qlora.W_codes.requires_grad, "W_codes must be frozen"
    assert not qlora.W_scales.requires_grad, "W_scales must be frozen"


# ---------------------------------------------------------------------------
# Test 11: QLoRALinear adapters A and B are trainable
# ---------------------------------------------------------------------------
def test_qlora_adapters_trainable():
    """A and B must have requires_grad=True."""
    linear = _make_linear(out=32, in_=64)
    qlora = QLoRALinear.from_linear(linear, r=4)
    assert qlora.A.requires_grad, "Adapter A must be trainable"
    assert qlora.B.requires_grad, "Adapter B must be trainable"


# ---------------------------------------------------------------------------
# Test 12: Initial QLoRALinear output ≈ original nn.Linear output
# ---------------------------------------------------------------------------
def test_qlora_initial_output_close_to_original():
    """At init (B=0), QLoRALinear output should approximate the original linear output."""
    torch.manual_seed(0)
    linear = _make_linear(out=64, in_=128)
    qlora = QLoRALinear.from_linear(linear, r=8, block_size=64)
    x = torch.randn(4, 128)
    with torch.no_grad():
        y_orig = linear(x)
        y_qlora = qlora(x)
    mae = (y_orig - y_qlora).abs().mean().item()
    # Only quantization error; should be small relative to typical output magnitude
    assert mae < 1.0, f"Initial output diverges too much from original: MAE={mae:.4f}"


# ---------------------------------------------------------------------------
# Test 13: from_linear output shape and dtype correct
# ---------------------------------------------------------------------------
def test_from_linear_shape_and_type():
    """from_linear must return a QLoRALinear with correct output shape."""
    linear = nn.Linear(128, 64, bias=False)
    qlora = QLoRALinear.from_linear(linear, r=4)
    assert isinstance(qlora, QLoRALinear)
    x = torch.randn(2, 128)
    out = qlora(x)
    assert out.shape == (2, 64)


# ---------------------------------------------------------------------------
# Test 14: Gradient flow through A and B
# ---------------------------------------------------------------------------
def test_gradient_flow_through_adapters():
    """Gradients must flow through both A and B after a backward pass."""
    linear = _make_linear(out=32, in_=64)
    qlora = QLoRALinear.from_linear(linear, r=4)
    x = torch.randn(2, 64, requires_grad=False)
    out = qlora(x)
    loss = out.sum()
    loss.backward()
    assert qlora.A.grad is not None, "No gradient on A"
    assert qlora.B.grad is not None, "No gradient on B"
    assert torch.isfinite(qlora.A.grad).all(), "Non-finite gradient on A"
    assert torch.isfinite(qlora.B.grad).all(), "Non-finite gradient on B"


# ---------------------------------------------------------------------------
# Test 15: No NaN/Inf on normal inputs
# ---------------------------------------------------------------------------
def test_no_nan_inf():
    """Forward pass must not produce NaN or Inf values on normal inputs."""
    linear = _make_linear(out=64, in_=128)
    qlora = QLoRALinear.from_linear(linear, r=8)
    x = torch.randn(4, 128)
    out = qlora(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 16: Determinism under fixed seed
# ---------------------------------------------------------------------------
def test_determinism():
    """Two identical QLoRALinear constructions with the same seed must produce
    identical outputs."""
    def _build_and_run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        linear = _make_linear(out=32, in_=64, seed=seed)
        qlora = QLoRALinear.from_linear(linear, r=4)
        x = torch.randn(2, 64)
        with torch.no_grad():
            return qlora(x)

    out1 = _build_and_run(42)
    out2 = _build_and_run(42)
    assert torch.allclose(out1, out2, atol=1e-6), "QLoRALinear is not deterministic under same seed"
