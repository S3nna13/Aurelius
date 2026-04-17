"""Tests for BitNet b1.58 implementation.

Tiny config: d_model=16, n_heads=2, n_layers=2, vocab_size=16, seq_len=4, batch=2.
Every test runs forward (and backward where relevant) passes.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.bitnet import (
    AbsMaxQuantizer,
    BitLinear,
    BitNetAnalyzer,
    BitNetBlock,
    BitNetModel,
    TernaryQuantizer,
)

# ---------------------------------------------------------------------------
# Tiny config constants
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 2
VOCAB_SIZE = 16
SEQ_LEN = 4
BATCH = 2


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_x(requires_grad: bool = False) -> torch.Tensor:
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    if requires_grad:
        x.requires_grad_(True)
    return x


def make_model() -> BitNetModel:
    return BitNetModel(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        vocab_size=VOCAB_SIZE,
    )


# ---------------------------------------------------------------------------
# TernaryQuantizer tests
# ---------------------------------------------------------------------------

def test_ternary_quantize_values_in_set():
    """Quantized output values must belong to {-1, 0, 1}."""
    q = TernaryQuantizer()
    W = torch.randn(8, 8)
    W_q = q.quantize(W)
    unique = W_q.unique().tolist()
    for v in unique:
        assert v in {-1.0, 0.0, 1.0}, f"Unexpected value {v} in quantized weight"


def test_ternary_quantize_shape_preserved():
    """Output shape must match input shape."""
    q = TernaryQuantizer()
    W = torch.randn(4, D_MODEL)
    assert q.quantize(W).shape == W.shape


def test_ternary_straight_through_forward_approx_quantized():
    """STE forward value should equal quantize(W)."""
    q = TernaryQuantizer()
    W = torch.randn(8, 8)
    W_ste = q.straight_through(W)
    W_q = q.quantize(W)
    # They must be numerically identical (STE forward = quantized)
    assert torch.allclose(W_ste, W_q), "STE forward must equal quantized weight"


def test_ternary_straight_through_gradient_flows():
    """Backward through STE must yield non-zero gradient on W."""
    q = TernaryQuantizer()
    W = nn.Parameter(torch.randn(8, 8))
    W_ste = q.straight_through(W)
    loss = W_ste.sum()
    loss.backward()
    assert W.grad is not None, "Gradient should exist after STE backward"
    assert W.grad.abs().sum().item() > 0, "Gradient should be non-zero"


def test_ternary_bit_width_range():
    """bit_width must be in [0, 1]."""
    q = TernaryQuantizer()
    W = torch.randn(16, 16)
    bw = q.bit_width(q.quantize(W))
    assert 0.0 <= bw <= 1.0, f"bit_width out of range: {bw}"


def test_ternary_bit_width_all_zeros_gives_zero():
    """A tensor of all zeros quantizes to all zeros → bit_width = 0.0."""
    q = TernaryQuantizer()
    W = torch.zeros(8, 8)
    W_q = q.quantize(W)
    bw = q.bit_width(W_q)
    assert bw == 0.0, f"Expected 0.0 bit_width for all-zero weights, got {bw}"


# ---------------------------------------------------------------------------
# AbsMaxQuantizer tests
# ---------------------------------------------------------------------------

def test_absmax_quantize_output_shape_unchanged():
    """Dequantized output must have the same shape as input."""
    aq = AbsMaxQuantizer(n_bits=8)
    x = make_x()
    x_dequant, scale = aq.quantize(x)
    assert x_dequant.shape == x.shape, "Dequantized shape mismatch"


def test_absmax_quantize_scale_shape():
    """Scale must have shape (B, T, 1) for per-token quantization."""
    aq = AbsMaxQuantizer(n_bits=8)
    x = make_x()
    _, scale = aq.quantize(x)
    assert scale.shape == (BATCH, SEQ_LEN, 1), f"Scale shape wrong: {scale.shape}"


def test_absmax_quantization_error_nonnegative():
    """Quantization error must be >= 0."""
    aq = AbsMaxQuantizer(n_bits=8)
    x = make_x()
    err = aq.quantization_error(x)
    assert err >= 0.0, f"Negative quantization error: {err}"


def test_absmax_quantization_error_decreases_with_more_bits():
    """Higher bit-width should give equal or lower quantization error."""
    x = make_x()
    err4 = AbsMaxQuantizer(n_bits=4).quantization_error(x)
    err8 = AbsMaxQuantizer(n_bits=8).quantization_error(x)
    assert err8 <= err4, f"8-bit error {err8} should be <= 4-bit error {err4}"


# ---------------------------------------------------------------------------
# BitLinear tests
# ---------------------------------------------------------------------------

def test_bitlinear_output_shape():
    """BitLinear output shape must be (B, T, out_features)."""
    layer = BitLinear(D_MODEL, D_MODEL)
    x = make_x()
    out = layer(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Shape mismatch: {out.shape}"


def test_bitlinear_weight_grad_after_backward():
    """Weight gradient must exist and be non-zero after a backward pass."""
    layer = BitLinear(D_MODEL, D_MODEL)
    x = make_x()
    loss = layer(x).sum()
    loss.backward()
    assert layer.weight.grad is not None, "No gradient on weight"
    assert layer.weight.grad.abs().sum().item() > 0, "Weight gradient is all zeros"


def test_bitlinear_bias_grad_after_backward():
    """Bias gradient must exist and be non-zero after a backward pass."""
    layer = BitLinear(D_MODEL, D_MODEL, bias=True)
    x = make_x()
    loss = layer(x).sum()
    loss.backward()
    assert layer.bias.grad is not None, "No gradient on bias"
    assert layer.bias.grad.abs().sum().item() > 0, "Bias gradient is all zeros"


def test_bitlinear_weight_remains_float32():
    """The stored weight parameter must remain float32 (quantization is transient)."""
    layer = BitLinear(D_MODEL, D_MODEL)
    assert layer.weight.dtype == torch.float32, \
        f"Weight dtype should be float32, got {layer.weight.dtype}"
    # After a forward pass, weight must still be float32
    _ = layer(make_x())
    assert layer.weight.dtype == torch.float32, \
        "Weight dtype changed after forward pass"


# ---------------------------------------------------------------------------
# BitNetBlock test
# ---------------------------------------------------------------------------

def test_bitnetblock_output_shape_and_grad():
    """Block output shape must be (B, T, D) and gradient must flow."""
    block = BitNetBlock(D_MODEL, N_HEADS)
    x = make_x(requires_grad=True)
    out = block(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Shape mismatch: {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input after block backward"
    assert x.grad.abs().sum().item() > 0, "Input gradient is all zeros"


# ---------------------------------------------------------------------------
# BitNetModel tests
# ---------------------------------------------------------------------------

def test_bitnetmodel_logits_shape():
    """Model logits must have shape (B, T, vocab_size)."""
    model = make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits = model(input_ids)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), \
        f"Logits shape mismatch: {logits.shape}"


def test_bitnetmodel_full_backward():
    """Full backward pass through the model must succeed without error."""
    model = make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()
    # Check at least one BitLinear weight received a gradient
    found_grad = False
    for m in model.modules():
        if isinstance(m, BitLinear) and m.weight.grad is not None:
            found_grad = True
            break
    assert found_grad, "No BitLinear layer received a gradient"


# ---------------------------------------------------------------------------
# BitNetAnalyzer tests
# ---------------------------------------------------------------------------

def test_analyzer_model_sparsity_in_range():
    """model_sparsity must be in [0, 1]."""
    model = make_model()
    analyzer = BitNetAnalyzer()
    sparsity = analyzer.model_sparsity(model)
    assert 0.0 <= sparsity <= 1.0, f"Sparsity out of range: {sparsity}"


def test_analyzer_effective_bits_in_range():
    """effective_bits must be in [0, log2(3)]."""
    model = make_model()
    analyzer = BitNetAnalyzer()
    eff = analyzer.effective_bits(model)
    assert 0.0 <= eff <= math.log2(3) + 1e-6, \
        f"effective_bits out of range: {eff}"


def test_analyzer_count_bitlinear_layers():
    """Each BitNetBlock has 6 BitLinear layers (4 attn + 2 FFN).
    Total = n_layers * 6.
    """
    model = make_model()
    analyzer = BitNetAnalyzer()
    count = analyzer.count_bitlinear_layers(model)
    expected = N_LAYERS * 6  # 4 attn projections + ffn_up + ffn_down
    assert count == expected, \
        f"Expected {expected} BitLinear layers, got {count}"
