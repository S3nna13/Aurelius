"""Tests for adaptive_compute.py — ACT halting, act_forward, ponder_loss, ACTTransformerLayer."""

import pytest
import torch
import torch.nn as nn

from src.model.adaptive_compute import (
    ACTConfig,
    ACTTransformerLayer,
    HaltingUnit,
    act_forward,
    ponder_loss,
)

# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------

B = 2
T = 4
D = 32
MAX_STEPS = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def act_cfg():
    return ACTConfig(threshold=0.99, max_steps=MAX_STEPS, epsilon=0.01, ponder_cost=0.01)


@pytest.fixture
def halting_unit():
    torch.manual_seed(0)
    return HaltingUnit(d_model=D)


@pytest.fixture
def mock_layers():
    """Two nn.Linear(D, D) modules acting as mock transformer sub-layers."""
    torch.manual_seed(1)
    return nn.ModuleList(
        [
            nn.Linear(D, D),
            nn.Linear(D, D),
        ]
    )


@pytest.fixture
def hidden():
    torch.manual_seed(42)
    return torch.randn(B, T, D)


@pytest.fixture
def act_layer(mock_layers, act_cfg):
    torch.manual_seed(2)
    return ACTTransformerLayer(layers=mock_layers, d_model=D, config=act_cfg)


# ---------------------------------------------------------------------------
# 1. ACTConfig defaults
# ---------------------------------------------------------------------------


def test_act_config_defaults():
    cfg = ACTConfig()
    assert cfg.threshold == 0.99
    assert cfg.max_steps == 10
    assert cfg.epsilon == 0.01
    assert cfg.ponder_cost == 0.01


# ---------------------------------------------------------------------------
# 2. HaltingUnit output shape (B, T, 1)
# ---------------------------------------------------------------------------


def test_halting_unit_output_shape(halting_unit, hidden):
    out = halting_unit(hidden)
    assert out.shape == (B, T, 1), f"Expected ({B}, {T}, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. HaltingUnit output in (0, 1) — sigmoid guarantee
# ---------------------------------------------------------------------------


def test_halting_unit_range(halting_unit, hidden):
    out = halting_unit(hidden)
    assert (out > 0).all(), "All halting probs should be > 0"
    assert (out < 1).all(), "All halting probs should be < 1"


# ---------------------------------------------------------------------------
# 4. HaltingUnit is differentiable
# ---------------------------------------------------------------------------


def test_halting_unit_differentiable(halting_unit, hidden):
    h = hidden.requires_grad_(True)
    out = halting_unit(h)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


# ---------------------------------------------------------------------------
# 5. act_forward output shape matches input
# ---------------------------------------------------------------------------


def test_act_forward_output_shape(hidden, mock_layers, halting_unit, act_cfg):
    output, _ = act_forward(hidden, mock_layers, halting_unit, act_cfg)
    assert output.shape == hidden.shape, (
        f"Output shape {output.shape} should match input {hidden.shape}"
    )


# ---------------------------------------------------------------------------
# 6. act_forward ponder_cost is scalar
# ---------------------------------------------------------------------------


def test_act_forward_ponder_cost_scalar(hidden, mock_layers, halting_unit, act_cfg):
    _, pc = act_forward(hidden, mock_layers, halting_unit, act_cfg)
    assert pc.shape == torch.Size([]), f"ponder_cost should be scalar, got shape {pc.shape}"


# ---------------------------------------------------------------------------
# 7. act_forward ponder_cost >= 1 (at least one step always taken)
# ---------------------------------------------------------------------------


def test_act_forward_ponder_cost_at_least_one(hidden, mock_layers, halting_unit, act_cfg):
    _, pc = act_forward(hidden, mock_layers, halting_unit, act_cfg)
    assert pc.item() >= 1.0 - 1e-6, (
        f"ponder_cost should be >= 1.0 (at least one step), got {pc.item()}"
    )


# ---------------------------------------------------------------------------
# 8. act_forward with threshold=0.0 → single step (always halts immediately)
# ---------------------------------------------------------------------------


def test_act_forward_threshold_zero_single_step(hidden, mock_layers, halting_unit):
    cfg_zero = ACTConfig(threshold=0.0, max_steps=MAX_STEPS, epsilon=0.01, ponder_cost=0.01)
    _, pc = act_forward(hidden, mock_layers, halting_unit, cfg_zero)
    # With threshold=0.0, every token should halt on the first step
    # ponder_cost should be very close to 1.0
    assert pc.item() <= 1.0 + 1e-5, (
        f"With threshold=0.0 expected ~1 step, got ponder_cost={pc.item()}"
    )


# ---------------------------------------------------------------------------
# 9. ponder_loss returns scalar
# ---------------------------------------------------------------------------


def test_ponder_loss_scalar():
    pc = torch.tensor(2.5)
    loss = ponder_loss(pc, target_ponder=1.0)
    assert loss.shape == torch.Size([]), f"ponder_loss should be scalar, got {loss.shape}"


# ---------------------------------------------------------------------------
# 10. ponder_loss target == current → near 0
# ---------------------------------------------------------------------------


def test_ponder_loss_zero_when_on_target():
    target = 2.0
    pc = torch.tensor(target)
    loss = ponder_loss(pc, target_ponder=target)
    assert loss.item() < 1e-10, f"ponder_loss should be ~0 when on target, got {loss.item()}"


# ---------------------------------------------------------------------------
# 11. ACTTransformerLayer output shape correct
# ---------------------------------------------------------------------------


def test_act_layer_output_shape(act_layer, hidden):
    output, _ = act_layer(hidden)
    assert output.shape == hidden.shape, (
        f"ACTTransformerLayer output {output.shape} should match input {hidden.shape}"
    )


# ---------------------------------------------------------------------------
# 12. ACTTransformerLayer ponder_cost is scalar
# ---------------------------------------------------------------------------


def test_act_layer_ponder_cost_scalar(act_layer, hidden):
    _, pc = act_layer(hidden)
    assert pc.shape == torch.Size([]), f"ponder_cost should be scalar, got shape {pc.shape}"


# ---------------------------------------------------------------------------
# 13. ACTTransformerLayer differentiable
# ---------------------------------------------------------------------------


def test_act_layer_differentiable(act_layer, hidden):
    h = hidden.requires_grad_(True)
    output, pc = act_layer(h)
    loss = output.sum() + pc
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


# ---------------------------------------------------------------------------
# 14. High threshold → more steps than low threshold (ponder_cost higher)
# ---------------------------------------------------------------------------


def test_act_high_threshold_more_steps_than_low(hidden, mock_layers, halting_unit):
    cfg_low = ACTConfig(threshold=0.01, max_steps=MAX_STEPS, epsilon=0.01, ponder_cost=0.01)
    cfg_high = ACTConfig(threshold=0.999, max_steps=MAX_STEPS, epsilon=0.01, ponder_cost=0.01)

    # Use the same halting unit (deterministic weights from fixture seed)
    _, pc_low = act_forward(hidden, mock_layers, halting_unit, cfg_low)
    _, pc_high = act_forward(hidden, mock_layers, halting_unit, cfg_high)

    assert pc_high.item() >= pc_low.item(), (
        f"High threshold should yield >= ponder steps than low threshold. "
        f"Got pc_high={pc_high.item():.4f}, pc_low={pc_low.item():.4f}"
    )
