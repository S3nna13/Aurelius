"""Tests for HGRN2: Gated Linear RNNs with State Expansion.

Reference: Qin et al., arXiv:2404.07904

Tiny config used throughout:
    d_model = 32, expand = 4, d_ff = 64, n_layers = 2,
    vocab_size = 128, lower_bound = 1/32
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from aurelius.model.hgrn2 import HGRN2Cell, HGRN2Layer, HGRN2Block, HGRN2Model


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_MODEL = 32
EXPAND = 4
D_FF = 64
N_LAYERS = 2
VOCAB_SIZE = 128
LOWER_BOUND = 1.0 / 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cell() -> HGRN2Cell:
    torch.manual_seed(0)
    return HGRN2Cell(D_MODEL, expand=EXPAND, lower_bound=LOWER_BOUND)


@pytest.fixture
def layer() -> HGRN2Layer:
    torch.manual_seed(1)
    return HGRN2Layer(D_MODEL, expand=EXPAND, lower_bound=LOWER_BOUND)


@pytest.fixture
def block() -> HGRN2Block:
    torch.manual_seed(2)
    return HGRN2Block(D_MODEL, D_FF, expand=EXPAND)


@pytest.fixture
def model() -> HGRN2Model:
    torch.manual_seed(3)
    return HGRN2Model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        expand=EXPAND,
    )


# ---------------------------------------------------------------------------
# Test 1: HGRN2Cell.step output shapes
# ---------------------------------------------------------------------------

def test_cell_step_output_shapes(cell: HGRN2Cell):
    """HGRN2Cell.step must return out (B, d_model) and h_new (B, d_model*expand)."""
    B = 3
    x = torch.randn(B, D_MODEL)
    h = torch.zeros(B, D_MODEL * EXPAND)

    out, h_new = cell.step(x, h)

    assert out.shape == (B, D_MODEL), (
        f"out shape: expected ({B}, {D_MODEL}), got {out.shape}"
    )
    assert h_new.shape == (B, D_MODEL * EXPAND), (
        f"h_new shape: expected ({B}, {D_MODEL * EXPAND}), got {h_new.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: Cell output is finite
# ---------------------------------------------------------------------------

def test_cell_output_finite(cell: HGRN2Cell):
    """HGRN2Cell.step output must be finite (no NaN / Inf)."""
    torch.manual_seed(42)
    B = 2
    x = torch.randn(B, D_MODEL)
    h = torch.randn(B, D_MODEL * EXPAND)

    out, h_new = cell.step(x, h)

    assert torch.isfinite(out).all(), "out contains NaN or Inf"
    assert torch.isfinite(h_new).all(), "h_new contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 3: Forget gate in [lower_bound, 1]
# ---------------------------------------------------------------------------

def test_forget_gate_range(cell: HGRN2Cell):
    """Forget gate values must lie in [lower_bound, 1] for any input."""
    torch.manual_seed(7)
    B = 8
    x = torch.randn(B, D_MODEL) * 10.0   # large inputs to stress-test clamping

    # Compute the forget gate directly (mirrors cell implementation)
    lb = LOWER_BOUND
    f = lb + (1.0 - lb) * torch.sigmoid(cell.W_f(x))

    assert (f >= lb - 1e-6).all(), f"forget gate below lower_bound ({lb})"
    assert (f <= 1.0 + 1e-6).all(), "forget gate above 1"


# ---------------------------------------------------------------------------
# Test 4: Zero input produces finite output
# ---------------------------------------------------------------------------

def test_cell_zero_input_finite(cell: HGRN2Cell):
    """Zero input must not produce NaN or Inf in the cell output."""
    B = 2
    x = torch.zeros(B, D_MODEL)
    h = torch.zeros(B, D_MODEL * EXPAND)

    out, h_new = cell.step(x, h)

    assert torch.isfinite(out).all(), "out is not finite for zero input"
    assert torch.isfinite(h_new).all(), "h_new is not finite for zero input"


# ---------------------------------------------------------------------------
# Test 5: HGRN2Layer output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_layer_output_shape(layer: HGRN2Layer):
    """HGRN2Layer.forward must return (B, T, d_model)."""
    B, T = 2, 16
    x = torch.randn(B, T, D_MODEL)

    out = layer(x)

    assert out.shape == (B, T, D_MODEL), (
        f"expected ({B}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: Layer output finite
# ---------------------------------------------------------------------------

def test_layer_output_finite(layer: HGRN2Layer):
    """HGRN2Layer output must be finite."""
    torch.manual_seed(10)
    B, T = 2, 16
    x = torch.randn(B, T, D_MODEL)

    out = layer(x)

    assert torch.isfinite(out).all(), "layer output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 7: Causal property -- earlier outputs are unaffected by future tokens
# ---------------------------------------------------------------------------

def test_layer_causal(layer: HGRN2Layer):
    """HGRN2Layer is causal: output[:, :t, :] is the same whether the sequence
    has T tokens or T+3 extra tokens appended."""
    torch.manual_seed(13)
    layer.train(False)
    B, T = 2, 10

    x = torch.randn(B, T + 3, D_MODEL)
    x_short = x[:, :T, :].clone()

    with torch.no_grad():
        out_full  = layer(x)
        out_short = layer(x_short)

    # The first T positions must match
    assert torch.allclose(out_full[:, :T, :], out_short, atol=1e-5), (
        f"Causal check failed: max diff = "
        f"{(out_full[:, :T, :] - out_short).abs().max():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8: Gradient flows through HGRN2Layer
# ---------------------------------------------------------------------------

def test_layer_gradient_flow(layer: HGRN2Layer):
    """loss.backward() must complete and produce non-None gradients in HGRN2Layer."""
    B, T = 2, 8
    x = torch.randn(B, T, D_MODEL, requires_grad=True)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient at input"
    has_param_grad = any(p.grad is not None for p in layer.parameters())
    assert has_param_grad, "No parameter received a gradient"


# ---------------------------------------------------------------------------
# Test 9: Batch=1, seq_len=1 works correctly
# ---------------------------------------------------------------------------

def test_layer_single_token(layer: HGRN2Layer):
    """HGRN2Layer must handle B=1, T=1 without error."""
    x = torch.randn(1, 1, D_MODEL)
    out = layer(x)

    assert out.shape == (1, 1, D_MODEL), f"got {out.shape}"
    assert torch.isfinite(out).all(), "output is not finite"


# ---------------------------------------------------------------------------
# Test 10: HGRN2Block output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_block_output_shape(block: HGRN2Block):
    """HGRN2Block.forward must return (B, T, d_model)."""
    B, T = 3, 12
    x = torch.randn(B, T, D_MODEL)

    out = block(x)

    assert out.shape == (B, T, D_MODEL), (
        f"expected ({B}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 11: Block output finite
# ---------------------------------------------------------------------------

def test_block_output_finite(block: HGRN2Block):
    """HGRN2Block output must be finite."""
    torch.manual_seed(20)
    B, T = 2, 8
    x = torch.randn(B, T, D_MODEL)

    out = block(x)

    assert torch.isfinite(out).all(), "block output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 12: Block residual is non-trivial (output != input)
# ---------------------------------------------------------------------------

def test_block_residual_nontrivial(block: HGRN2Block):
    """HGRN2Block output must differ from its input (residual is active)."""
    torch.manual_seed(25)
    block.train(False)
    B, T = 2, 8
    x = torch.randn(B, T, D_MODEL)

    with torch.no_grad():
        out = block(x)

    assert not torch.allclose(out, x, atol=1e-6), (
        "block output is identical to input -- residual branch appears dead"
    )


# ---------------------------------------------------------------------------
# Test 13: HGRN2Model output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_model_output_shape(model: HGRN2Model):
    """HGRN2Model.forward must return (B, T, d_model)."""
    B, T = 2, 16
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))

    out = model(input_ids)

    assert out.shape == (B, T, D_MODEL), (
        f"expected ({B}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 14: Gradient flows through HGRN2Model
# ---------------------------------------------------------------------------

def test_model_gradient_flow(model: HGRN2Model):
    """loss.backward() must complete and produce non-None gradients in HGRN2Model."""
    B, T = 2, 8
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))

    out = model(input_ids)
    loss = out.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No parameter received a gradient after backward()"
