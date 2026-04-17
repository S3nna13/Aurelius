"""Tests for gradient_noise.py — Neelakantan et al. 2015."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import pytest

from aurelius.training.gradient_noise import (
    GradientNoiseCallback,
    GradientNoiseOptimizer,
    GradientNoiseSchedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_model() -> nn.Linear:
    """Return a small deterministic linear model."""
    torch.manual_seed(42)
    return nn.Linear(4, 4)


def _optimizer_with_noise(
    model: nn.Module,
    eta: float = 0.01,
    gamma: float = 0.55,
    seed: int = 0,
) -> GradientNoiseOptimizer:
    inner = torch.optim.SGD(model.parameters(), lr=0.01)
    schedule = GradientNoiseSchedule(eta=eta, gamma=gamma)
    return GradientNoiseOptimizer(inner, schedule, seed=seed)


def _compute_loss(model: nn.Module) -> torch.Tensor:
    x = torch.randn(2, 4)
    return model(x).sum()


# ---------------------------------------------------------------------------
# GradientNoiseSchedule tests
# ---------------------------------------------------------------------------

def test_variance_at_step_zero_equals_eta():
    """σ² at step 0 must equal eta exactly."""
    schedule = GradientNoiseSchedule(eta=0.05, gamma=0.55)
    assert schedule.variance(0) == pytest.approx(0.05)


def test_variance_decreases_with_step():
    """Variance must be strictly decreasing as step increases."""
    schedule = GradientNoiseSchedule(eta=0.01, gamma=0.55)
    variances = [schedule.variance(t) for t in range(10)]
    for i in range(len(variances) - 1):
        assert variances[i] > variances[i + 1], (
            f"variance did not decrease between step {i} and {i + 1}"
        )


def test_std_equals_sqrt_variance():
    """std(t) must equal sqrt(variance(t)) for several steps."""
    schedule = GradientNoiseSchedule(eta=0.02, gamma=0.55)
    for t in [0, 1, 5, 100]:
        assert schedule.std(t) == pytest.approx(math.sqrt(schedule.variance(t)))


def test_noise_tensor_shape():
    """noise_tensor must return a tensor with the requested shape."""
    schedule = GradientNoiseSchedule()
    shape = (3, 4, 5)
    noise = schedule.noise_tensor(shape, step=0)
    assert noise.shape == torch.Size(shape)


def test_noise_tensor_finite():
    """All elements of the noise tensor must be finite."""
    schedule = GradientNoiseSchedule()
    noise = schedule.noise_tensor((100,), step=0)
    assert torch.isfinite(noise).all()


def test_noise_tensor_std_approximates_schedule_std():
    """Empirical std of a large noise tensor should approximate schedule.std."""
    schedule = GradientNoiseSchedule(eta=0.04, gamma=0.55)
    step = 10
    expected_std = schedule.std(step)
    # Draw many samples to reduce variance of the estimate.
    noise = schedule.noise_tensor((100_000,), step=step)
    empirical_std = noise.std().item()
    assert empirical_std == pytest.approx(expected_std, rel=0.05)


# ---------------------------------------------------------------------------
# GradientNoiseOptimizer tests
# ---------------------------------------------------------------------------

def test_step_increments_step_count():
    """_step_count must increase by 1 after each step() call."""
    model = _simple_model()
    opt = _optimizer_with_noise(model)
    assert opt._step_count == 0

    for expected in range(1, 4):
        opt.zero_grad()
        _compute_loss(model).backward()
        opt.step()
        assert opt._step_count == expected


def test_step_modifies_gradients():
    """Gradients observed inside the optimizer must differ from clean grads."""
    torch.manual_seed(0)
    model = _simple_model()

    # Capture clean gradients before any noise is added.
    _compute_loss(model).backward()
    clean_grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    # Now exercise GradientNoiseOptimizer with a *different* random seed so
    # the noise is non-zero and the comparison is meaningful.
    model2 = _simple_model()
    opt = _optimizer_with_noise(model2, eta=1.0, seed=99)
    opt.zero_grad()
    _compute_loss(model2).backward()

    # Capture grads just before step() adds noise — we need to compare
    # the gradient state *after* noise injection.  We do this by peeking
    # at the grad after calling step() using a tiny subclass.
    grads_before_noise = {
        name: p.grad.clone()
        for name, p in model2.named_parameters()
        if p.grad is not None
    }
    opt.step()

    # The gradients used internally differ from the pre-noise values.
    # We confirm by checking that clean_grads != grads seen by the inner
    # optimizer (i.e. model2's params were updated from noisy grads).
    # A simpler proxy: re-run without noise and check param values differ.
    model3 = _simple_model()
    inner3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    inner3.zero_grad()
    _compute_loss(model3).backward()
    inner3.step()

    # With eta=1.0 the noise is very large — param updates will differ.
    for (n2, p2), (n3, p3) in zip(
        model2.named_parameters(), model3.named_parameters()
    ):
        # At least one parameter must differ between noisy and clean.
        if not torch.allclose(p2, p3, atol=1e-6):
            return  # Test passes as soon as one difference is found.

    pytest.fail("Parameters are identical after noisy vs. clean optimizer steps.")


def test_zero_grad_zeroes_gradients():
    """zero_grad() must clear all parameter gradients."""
    model = _simple_model()
    opt = _optimizer_with_noise(model)

    _compute_loss(model).backward()
    # Confirm gradients exist.
    assert any(p.grad is not None for p in model.parameters())

    opt.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0)


def test_state_dict_contains_expected_keys():
    """state_dict() must contain 'optimizer', 'step_count', 'eta', 'gamma'."""
    model = _simple_model()
    opt = _optimizer_with_noise(model, eta=0.02, gamma=0.6)
    state = opt.state_dict()
    for key in ("optimizer", "step_count", "eta", "gamma"):
        assert key in state, f"Missing key: {key}"


def test_load_state_dict_restores_step_count():
    """load_state_dict() must restore _step_count from the saved state."""
    model = _simple_model()
    opt = _optimizer_with_noise(model)

    for _ in range(5):
        opt.zero_grad()
        _compute_loss(model).backward()
        opt.step()

    state = opt.state_dict()
    assert state["step_count"] == 5

    # Create a fresh optimizer and restore.
    model2 = _simple_model()
    opt2 = _optimizer_with_noise(model2)
    opt2.load_state_dict(state)
    assert opt2._step_count == 5


def test_param_groups_accessible():
    """param_groups property must expose the inner optimizer's param_groups."""
    model = _simple_model()
    opt = _optimizer_with_noise(model)
    pg = opt.param_groups
    assert isinstance(pg, list)
    assert len(pg) > 0
    assert "params" in pg[0]


# ---------------------------------------------------------------------------
# GradientNoiseCallback tests
# ---------------------------------------------------------------------------

def test_callback_record_appends_to_history():
    """record() must append the noise std to noise_std_history."""
    schedule = GradientNoiseSchedule(eta=0.01, gamma=0.55)
    cb = GradientNoiseCallback()
    assert cb.noise_std_history == []

    for step in range(5):
        val = cb.record(schedule, step)
        assert val == pytest.approx(schedule.std(step))
        assert len(cb.noise_std_history) == step + 1
        assert cb.noise_std_history[-1] == pytest.approx(schedule.std(step))


def test_plot_schedule_returns_correct_length():
    """plot_schedule() must return a list with exactly n_steps elements."""
    schedule = GradientNoiseSchedule()
    cb = GradientNoiseCallback()
    n = 50
    result = cb.plot_schedule(n, schedule)
    assert isinstance(result, list)
    assert len(result) == n


def test_plot_schedule_values_are_non_increasing():
    """Noise std must be non-increasing across all steps."""
    schedule = GradientNoiseSchedule(eta=0.01, gamma=0.55)
    cb = GradientNoiseCallback()
    values = cb.plot_schedule(100, schedule)
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], (
            f"Noise std increased between step {i} ({values[i]}) "
            f"and step {i + 1} ({values[i + 1]})"
        )
