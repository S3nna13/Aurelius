"""Tests for src/model/act.py — Adaptive Computation Time (Graves 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.act import ACTLayer, ACTTransformer, HaltingUnit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
BATCH = 2
SEQ = 8


def _simple_block(d_model: int) -> nn.Module:
    """A minimal linear block for testing."""
    return nn.Linear(d_model, d_model)


def _act_layer(**kwargs) -> ACTLayer:
    block = _simple_block(D_MODEL)
    return ACTLayer(block, D_MODEL, **kwargs)


# ---------------------------------------------------------------------------
# HaltingUnit tests
# ---------------------------------------------------------------------------


def test_halting_unit_output_range():
    """Halting probabilities must lie in (0, 1).

    sigmoid is mathematically bounded by (0, 1), but float32 saturates to
    exactly 0.0 or 1.0 for very large magnitude inputs.  We therefore use
    normal-scale inputs (no * 100 amplification) where saturation does not
    occur, and verify the open-interval property holds for typical inputs.
    """
    unit = HaltingUnit(D_MODEL)
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, D_MODEL)  # standard-scale inputs
    probs = unit(x)
    assert (probs > 0).all(), "Some halting probabilities were <= 0"
    assert (probs < 1).all(), "Some halting probabilities were >= 1"


def test_halting_unit_shape():
    """HaltingUnit maps (B, S, D) -> (B, S, 1)."""
    unit = HaltingUnit(D_MODEL)
    x = torch.randn(BATCH, 16, D_MODEL)
    out = unit(x)
    assert out.shape == (BATCH, 16, 1), f"Expected (2, 16, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# ACTLayer tests
# ---------------------------------------------------------------------------


def test_act_layer_output_shape():
    """ACTLayer preserves the spatial shape of the input."""
    layer = _act_layer()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    output, _ = layer(x)
    assert output.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {output.shape}"
    )


def test_act_layer_ponder_cost_scalar():
    """ponder_cost must be a 0-d (scalar) tensor."""
    layer = _act_layer()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, ponder_cost = layer(x)
    assert ponder_cost.ndim == 0, f"ponder_cost should be 0-d, got ndim={ponder_cost.ndim}"


def test_act_layer_ponder_cost_positive():
    """ponder_cost must be >= 0."""
    layer = _act_layer()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, ponder_cost = layer(x)
    assert ponder_cost.item() >= 0.0, f"ponder_cost={ponder_cost.item()} is negative"


def test_act_layer_weights_sum_to_one():
    """Accumulation weights across steps must sum to ~1.0 per token.

    We verify this indirectly: if every block step returns a constant
    all-ones tensor, the output should also be (approximately) all-ones,
    because sum(weights) == 1.
    """

    class ConstantBlock(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x)

    layer = ACTLayer(ConstantBlock(), D_MODEL, max_steps=8)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    output, _ = layer(x)
    # Each element should be very close to 1.0 since weights sum to 1.
    assert torch.allclose(output, torch.ones_like(output), atol=1e-5), (
        "Weights do not sum to 1: output deviates from constant block output"
    )


def test_act_layer_stops_before_max():
    """With a high threshold and mild input, mean ponder steps < max_steps."""
    max_steps = 8
    layer = _act_layer(max_steps=max_steps, threshold=0.99)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    mean_steps = layer.mean_ponder_steps(x)
    assert mean_steps < max_steps, (
        f"mean_ponder_steps={mean_steps:.2f} should be < max_steps={max_steps}"
    )


def test_act_layer_gradients_flow():
    """Backward pass must not error; gradients should reach block parameters."""
    layer = _act_layer()
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    output, ponder_cost = layer(x)
    loss = output.sum() + ponder_cost
    loss.backward()
    # x should have received gradients.
    assert x.grad is not None, "No gradient flowed back to input x"
    # At least some block parameter should have a gradient.
    block_params = list(layer.block.parameters())
    assert any(p.grad is not None for p in block_params), (
        "No gradient reached block parameters"
    )


# ---------------------------------------------------------------------------
# ACTTransformer tests
# ---------------------------------------------------------------------------


def test_act_transformer_output_shape():
    """ACTTransformer maps (B, S, D) -> (B, S, D) with a scalar ponder cost."""
    model = ACTTransformer(D_MODEL, n_layers=2, max_steps=4)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    output, ponder_cost = model(x)
    assert output.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {output.shape}"
    )
    assert ponder_cost.ndim == 0, "total_ponder_cost should be a scalar tensor"


def test_act_ponder_cost_in_loss():
    """ponder_cost integrates with a main loss; end-to-end backward must work."""
    model = ACTTransformer(D_MODEL, n_layers=2, max_steps=4)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    target = torch.randn(BATCH, SEQ, D_MODEL)

    output, ponder_cost = model(x)
    main_loss = nn.functional.mse_loss(output, target)
    total_loss = main_loss + ponder_cost

    # Should not raise.
    total_loss.backward()

    # Verify at least one model parameter received a gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No parameters received gradients after backward"
