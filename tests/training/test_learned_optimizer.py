"""Tests for the learned optimizer (L2L) module."""
import math

import pytest
import torch
import torch.nn as nn

from src.training.learned_optimizer import (
    GradientPreprocessor,
    LSTMOptimizer,
    LearnedOptimizerWrapper,
    MetaTrainingLoop,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lstm_optimizer():
    return LSTMOptimizer(input_size=2, hidden_size=20, n_layers=2)


@pytest.fixture
def simple_model():
    """Small linear model for testing."""
    return nn.Linear(8, 4)


@pytest.fixture
def wrapper(lstm_optimizer, simple_model):
    return LearnedOptimizerWrapper(lstm_optimizer, simple_model)


# ---------------------------------------------------------------------------
# GradientPreprocessor tests
# ---------------------------------------------------------------------------


def test_gradient_preprocessor_shape():
    """preprocess(rand_grad) should return shape (numel, 2)."""
    grad = torch.randn(4, 8)  # 32 elements
    out = GradientPreprocessor.preprocess(grad)
    assert out.shape == (grad.numel(), 2), (
        f"Expected shape ({grad.numel()}, 2), got {out.shape}"
    )


def test_gradient_preprocessor_range():
    """log_mag column (col 0) must be in [-1, 1]."""
    grad = torch.randn(16, 16)
    out = GradientPreprocessor.preprocess(grad)
    log_mag = out[:, 0]
    assert log_mag.min().item() >= -1.0 - 1e-6, (
        f"log_mag below -1: min={log_mag.min().item()}"
    )
    assert log_mag.max().item() <= 1.0 + 1e-6, (
        f"log_mag above 1: max={log_mag.max().item()}"
    )


def test_gradient_preprocessor_sign():
    """sign column (col 1) must only contain values in {-1, 0, 1}."""
    grad = torch.randn(10, 10)
    out = GradientPreprocessor.preprocess(grad)
    sign_col = out[:, 1]
    valid_values = {-1.0, 0.0, 1.0}
    unique_vals = set(sign_col.unique().tolist())
    assert unique_vals.issubset(valid_values), (
        f"Sign column contains unexpected values: {unique_vals - valid_values}"
    )


# ---------------------------------------------------------------------------
# LSTMOptimizer tests
# ---------------------------------------------------------------------------


def test_lstm_optimizer_output_shape(lstm_optimizer):
    """forward should return updates with shape (numel,)."""
    numel = 32
    grad_features = torch.randn(numel, 1, 2)
    updates, state = lstm_optimizer(grad_features, state=None)
    assert updates.shape == (numel,), (
        f"Expected updates shape ({numel},), got {updates.shape}"
    )


def test_lstm_optimizer_state_maintained(lstm_optimizer):
    """LSTM state should change between steps."""
    numel = 16
    grad_features = torch.randn(numel, 1, 2)

    _, state1 = lstm_optimizer(grad_features, state=None)
    _, state2 = lstm_optimizer(grad_features, state=state1)

    # Hidden state (h) should differ between first and second step
    h1 = state1[0]  # (n_layers, numel, hidden_size)
    h2 = state2[0]
    assert not torch.allclose(h1, h2), "LSTM hidden state did not change between steps"


# ---------------------------------------------------------------------------
# LearnedOptimizerWrapper tests
# ---------------------------------------------------------------------------


def test_learned_optimizer_step(wrapper, simple_model):
    """step() should return a dict with the correct keys."""
    x = torch.randn(4, 8)
    loss = simple_model(x).sum()
    result = wrapper.step(loss)
    assert "n_params_updated" in result, "Missing key 'n_params_updated'"
    assert "mean_update_norm" in result, "Missing key 'mean_update_norm'"


def test_learned_optimizer_updates_params(lstm_optimizer, simple_model):
    """Parameters should change after a step."""
    wrapper = LearnedOptimizerWrapper(lstm_optimizer, simple_model)
    before = {name: p.data.clone() for name, p in simple_model.named_parameters()}

    x = torch.randn(4, 8)
    loss = simple_model(x).sum()
    wrapper.step(loss)

    changed = any(
        not torch.allclose(p.data, before[name])
        for name, p in simple_model.named_parameters()
    )
    assert changed, "No parameters were updated after LearnedOptimizerWrapper.step()"


def test_learned_optimizer_reset_states(wrapper, simple_model):
    """After reset_states(), internal state dict should be empty."""
    # Run a step to populate states
    x = torch.randn(4, 8)
    loss = simple_model(x).sum()
    wrapper.step(loss)

    assert len(wrapper._states) > 0, "States should be populated after a step"

    wrapper.reset_states()
    assert len(wrapper._states) == 0, (
        f"States not cleared after reset_states(); still has {len(wrapper._states)} entries"
    )


# ---------------------------------------------------------------------------
# MetaTrainingLoop tests
# ---------------------------------------------------------------------------


def test_meta_training_step(lstm_optimizer):
    """meta_step should return a float."""
    meta_loop = MetaTrainingLoop(lstm_optimizer, meta_lr=1e-3)
    task_model = nn.Linear(4, 2)

    def task_loss_fn():
        x = torch.randn(8, 4)
        return task_model(x).pow(2).sum()

    result = meta_loop.meta_step(task_model, task_loss_fn, n_unroll=3)
    assert isinstance(result, float), f"meta_step should return float, got {type(result)}"


def test_lstm_optimizer_gradients(lstm_optimizer):
    """meta_step should produce gradients on lstm_optimizer parameters."""
    meta_loop = MetaTrainingLoop(lstm_optimizer, meta_lr=1e-3)
    task_model = nn.Linear(4, 2)

    def task_loss_fn():
        x = torch.randn(8, 4)
        return task_model(x).pow(2).sum()

    # Run a meta step — meta_optimizer.step() will consume gradients,
    # so we check that at least one param had a non-None grad by verifying
    # the lstm parameters were updated (grad was computed and step taken).
    params_before = [p.data.clone() for p in lstm_optimizer.parameters()]
    meta_loop.meta_step(task_model, task_loss_fn, n_unroll=3)
    params_after = [p.data for p in lstm_optimizer.parameters()]

    # At least one parameter in the lstm should have been updated by the meta step
    any_changed = any(
        not torch.allclose(b, a) for b, a in zip(params_before, params_after)
    )
    assert any_changed, "No LSTM optimizer parameters changed after meta_step — gradients may not have been computed"
