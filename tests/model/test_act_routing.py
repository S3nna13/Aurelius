"""Tests for src/model/act_routing.py — Adaptive Computation Time module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.act_routing import (
    ACTConfig,
    ACTWrapper,
    HaltingUnit,
    compute_act_loss,
    compute_act_state,
    compute_ponder_cost,
)

# Common test dimensions
B, T, D = 2, 4, 32


# ---------------------------------------------------------------------------
# ACTConfig
# ---------------------------------------------------------------------------


def test_act_config_defaults():
    cfg = ACTConfig()
    assert cfg.max_steps == 8
    assert cfg.halt_threshold == 0.99
    assert cfg.ponder_cost == 0.01
    assert cfg.epsilon == 1e-2


# ---------------------------------------------------------------------------
# HaltingUnit
# ---------------------------------------------------------------------------


def test_halting_unit_output_shape():
    unit = HaltingUnit(D)
    x = torch.randn(B, T, D)
    h = unit(x)
    assert h.shape == (B, T), f"Expected ({B}, {T}), got {h.shape}"


def test_halting_unit_output_range():
    unit = HaltingUnit(D)
    x = torch.randn(B, T, D)
    h = unit(x)
    assert (h > 0).all(), "Halting probs should be > 0"
    assert (h < 1).all(), "Halting probs should be < 1"


# ---------------------------------------------------------------------------
# compute_act_state
# ---------------------------------------------------------------------------


def test_compute_act_state_already_halted_zeroed():
    """Tokens that were already halted should get h_t = 0."""
    h_t = torch.ones(B, T) * 0.5
    cumulative = torch.zeros(B, T)
    halted = torch.ones(B, T, dtype=torch.bool)

    adjusted, new_halted = compute_act_state(h_t, cumulative, halted)
    assert (adjusted == 0.0).all(), "Already-halted tokens must have adjusted_h_t=0"


def test_compute_act_state_threshold_exceeded_uses_remainder():
    """When cumulative + h_t >= 1, remainder (1 - cumulative) should be used."""
    h_t = torch.ones(B, T) * 0.9
    cumulative = torch.ones(B, T) * 0.5  # cumulative + h_t = 1.4 >= 1.0
    halted = torch.zeros(B, T, dtype=torch.bool)

    adjusted, new_halted = compute_act_state(h_t, cumulative, halted)
    expected_remainder = (1.0 - cumulative).clamp(min=0.0)
    assert torch.allclose(adjusted, expected_remainder), (
        "Should use remainder when threshold exceeded"
    )


def test_compute_act_state_new_halted_dtype():
    """new_halted should be a boolean tensor."""
    h_t = torch.ones(B, T) * 0.9
    cumulative = torch.ones(B, T) * 0.5
    halted = torch.zeros(B, T, dtype=torch.bool)

    _, new_halted = compute_act_state(h_t, cumulative, halted)
    assert new_halted.dtype == torch.bool, f"Expected bool, got {new_halted.dtype}"


def test_compute_act_state_no_halt_below_threshold():
    """Tokens that don't exceed the threshold should not be newly halted."""
    h_t = torch.ones(B, T) * 0.1
    cumulative = torch.zeros(B, T)
    halted = torch.zeros(B, T, dtype=torch.bool)

    adjusted, new_halted = compute_act_state(h_t, cumulative, halted)
    assert not new_halted.any(), "Should not halt when cumulative stays below 1.0"


# ---------------------------------------------------------------------------
# compute_ponder_cost
# ---------------------------------------------------------------------------


def test_compute_ponder_cost_scalar():
    n_steps = torch.randint(1, 9, (B, T))
    cost = compute_ponder_cost(n_steps)
    assert cost.shape == (), f"Expected scalar, got shape {cost.shape}"


def test_compute_ponder_cost_all_one_step():
    n_steps = torch.ones(B, T, dtype=torch.long)
    cost = compute_ponder_cost(n_steps)
    assert cost.item() == pytest.approx(1.0), f"Expected 1.0, got {cost.item()}"


# ---------------------------------------------------------------------------
# ACTWrapper
# ---------------------------------------------------------------------------


def _make_wrapper(max_steps: int = 4) -> ACTWrapper:
    block = nn.Linear(D, D)
    cfg = ACTConfig(max_steps=max_steps)
    return ACTWrapper(block, D, cfg)


def test_act_wrapper_output_shape():
    wrapper = _make_wrapper()
    x = torch.randn(B, T, D)
    out, _ = wrapper(x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


def test_act_wrapper_returns_correct_keys():
    wrapper = _make_wrapper()
    x = torch.randn(B, T, D)
    _, info = wrapper(x)
    assert "ponder_cost" in info, "info must contain 'ponder_cost'"
    assert "mean_steps" in info, "info must contain 'mean_steps'"
    assert "halted_at_step" in info, "info must contain 'halted_at_step'"


def test_act_wrapper_mean_steps_in_range():
    max_steps = 6
    wrapper = _make_wrapper(max_steps=max_steps)
    x = torch.randn(B, T, D)
    _, info = wrapper(x)
    mean_steps = info["mean_steps"]
    assert 1.0 <= mean_steps <= max_steps, (
        f"mean_steps {mean_steps} not in [1, {max_steps}]"
    )


def test_act_wrapper_halted_at_step_shape():
    max_steps = 4
    wrapper = _make_wrapper(max_steps=max_steps)
    x = torch.randn(B, T, D)
    _, info = wrapper(x)
    hstep = info["halted_at_step"]
    assert hstep.shape == (B, T), f"Expected ({B}, {T}), got {hstep.shape}"


# ---------------------------------------------------------------------------
# compute_act_loss
# ---------------------------------------------------------------------------


def test_compute_act_loss_combines_correctly():
    task_loss = torch.tensor(2.0)
    ponder_cost = torch.tensor(3.0)
    ponder_weight = 0.5

    total = compute_act_loss(task_loss, ponder_cost, ponder_weight)
    expected = 2.0 + 0.5 * 3.0  # 3.5
    assert total.item() == pytest.approx(expected), (
        f"Expected {expected}, got {total.item()}"
    )
