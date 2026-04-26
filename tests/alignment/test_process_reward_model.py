"""Tests for src/alignment/process_reward_model.py (~16 tests)."""

import math

import pytest
import torch
import torch.nn as nn
from aurelius.alignment.process_reward_model import (
    PRMHead,
    PRMLoss,
    PRMTrainer,
    ProcessRewardModel,
    StepReward,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, T, D = 2, 10, 32  # batch, sequence length, model dim


@pytest.fixture()
def prm_head():
    torch.manual_seed(0)
    return PRMHead(d_model=D)


@pytest.fixture()
def hidden_states():
    torch.manual_seed(1)
    return torch.randn(B, T, D)


@pytest.fixture()
def dummy_backbone():
    """A trivial backbone that is never actually called by ProcessRewardModel."""
    return nn.Identity()


@pytest.fixture()
def prm_model(dummy_backbone):
    torch.manual_seed(2)
    return ProcessRewardModel(backbone=dummy_backbone, d_model=D)


@pytest.fixture()
def step_mask():
    """Step boundaries at positions 2 and 7 for both batch items."""
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 2] = True
    mask[:, 7] = True
    return mask  # 2 steps per item


@pytest.fixture()
def prm_trainer(prm_model):
    opt = torch.optim.SGD(prm_model.prm_head.parameters(), lr=1e-2)
    return PRMTrainer(model=prm_model, optimizer=opt)


# ---------------------------------------------------------------------------
# PRMHead tests (1-3)
# ---------------------------------------------------------------------------


def test_prm_head_output_shape(prm_head, hidden_states):
    """PRMHead.forward should return shape (B, T)."""
    out = prm_head(hidden_states)
    assert out.shape == (B, T)


def test_prm_head_output_finite(prm_head, hidden_states):
    """PRMHead outputs must be finite."""
    out = prm_head(hidden_states)
    assert torch.isfinite(out).all()


def test_prm_head_gradient_flows(prm_head, hidden_states):
    """Gradient should flow back through PRMHead."""
    hs = hidden_states.requires_grad_(True)
    out = prm_head(hs)
    out.sum().backward()
    assert hs.grad is not None
    assert torch.isfinite(hs.grad).all()


# ---------------------------------------------------------------------------
# ProcessRewardModel.forward tests (4-5)
# ---------------------------------------------------------------------------


def test_prm_model_forward_shape(prm_model, hidden_states):
    """ProcessRewardModel.forward should return (B, T)."""
    out = prm_model(hidden_states)
    assert out.shape == (B, T)


def test_prm_model_output_finite(prm_model, hidden_states):
    """ProcessRewardModel.forward outputs must be finite."""
    out = prm_model(hidden_states)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# score_steps tests (6-8)
# ---------------------------------------------------------------------------


def test_score_steps_output_shape(prm_model, hidden_states, step_mask):
    """score_steps should return (B, n_steps) where n_steps = max steps in batch."""
    out = prm_model.score_steps(hidden_states, step_mask)
    n_steps = int(step_mask.sum(dim=1).max().item())
    assert out.shape == (B, n_steps)


def test_score_steps_values_at_boundaries(prm_model, hidden_states, step_mask):
    """Rewards at step boundaries in score_steps should match forward() at those positions."""
    all_rewards = prm_model(hidden_states)  # (B, T)
    scored = prm_model.score_steps(hidden_states, step_mask)  # (B, 2)

    # Item 0, step 0 -> position 2; step 1 -> position 7
    assert torch.isclose(scored[0, 0], all_rewards[0, 2])
    assert torch.isclose(scored[0, 1], all_rewards[0, 7])
    # Item 1 similarly
    assert torch.isclose(scored[1, 0], all_rewards[1, 2])
    assert torch.isclose(scored[1, 1], all_rewards[1, 7])


def test_score_steps_variable_steps_padding(prm_model, hidden_states):
    """Batch items with fewer steps should be padded with 0."""
    # Item 0: 3 steps; item 1: 1 step
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 1] = True
    mask[0, 5] = True
    mask[0, 8] = True
    mask[1, 3] = True

    out = prm_model.score_steps(hidden_states, mask)  # (B, 3)
    assert out.shape == (B, 3)
    # Item 1 should have 0-padding at positions 1 and 2
    assert out[1, 1].item() == 0.0
    assert out[1, 2].item() == 0.0


# ---------------------------------------------------------------------------
# PRMLoss tests (9-12)
# ---------------------------------------------------------------------------


def test_prm_loss_scalar_output():
    """PRMLoss should return a scalar tensor."""
    loss_fn = PRMLoss()
    preds = torch.randn(B, T)
    labels = torch.randint(0, 2, (B, T)).float()
    mask = torch.ones(B, T, dtype=torch.bool)
    loss = loss_fn(preds, labels, mask)
    assert loss.ndim == 0


def test_prm_loss_finite():
    """PRMLoss output should be finite."""
    loss_fn = PRMLoss()
    preds = torch.randn(B, T)
    labels = torch.randint(0, 2, (B, T)).float()
    mask = torch.ones(B, T, dtype=torch.bool)
    loss = loss_fn(preds, labels, mask)
    assert math.isfinite(loss.item())


def test_prm_loss_all_false_mask():
    """PRMLoss should return 0 when mask is all-False."""
    loss_fn = PRMLoss()
    preds = torch.randn(B, T)
    labels = torch.randint(0, 2, (B, T)).float()
    mask = torch.zeros(B, T, dtype=torch.bool)
    loss = loss_fn(preds, labels, mask)
    assert loss.item() == 0.0


def test_prm_loss_gradient_flows():
    """Gradient should flow back through PRMLoss."""
    loss_fn = PRMLoss()
    preds = torch.randn(B, T, requires_grad=True)
    labels = torch.randint(0, 2, (B, T)).float()
    mask = torch.ones(B, T, dtype=torch.bool)
    loss = loss_fn(preds, labels, mask)
    loss.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


# ---------------------------------------------------------------------------
# PRMTrainer tests (13-16)
# ---------------------------------------------------------------------------


def test_trainer_train_step_keys(prm_trainer, hidden_states, step_mask):
    """train_step must return dict with 'loss', 'accuracy', 'n_steps'."""
    labels = torch.randint(0, 2, (B, T)).float()
    result = prm_trainer.train_step(hidden_states, labels, step_mask)
    assert "loss" in result
    assert "accuracy" in result
    assert "n_steps" in result


def test_trainer_accuracy_in_range(prm_trainer, hidden_states, step_mask):
    """Accuracy reported by train_step must be in [0, 1]."""
    labels = torch.randint(0, 2, (B, T)).float()
    result = prm_trainer.train_step(hidden_states, labels, step_mask)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluate_chain_length(prm_trainer, hidden_states, step_mask):
    """evaluate_chain should return one StepReward per step boundary in item 0."""
    chain = prm_trainer.evaluate_chain(hidden_states, step_mask)
    n_expected = int(step_mask[0].sum().item())
    assert isinstance(chain, list)
    assert len(chain) == n_expected
    assert all(isinstance(s, StepReward) for s in chain)


def test_evaluate_chain_rewards_match_model(prm_trainer, hidden_states, step_mask):
    """StepReward values should match the model's forward pass at step positions."""
    chain = prm_trainer.evaluate_chain(hidden_states, step_mask)

    with torch.no_grad():
        all_rewards = prm_trainer.model(hidden_states)  # (B, T)

    boundary_positions = step_mask[0].nonzero(as_tuple=False).squeeze(1).tolist()
    for sr, pos in zip(chain, boundary_positions):
        expected = all_rewards[0, pos].item()
        assert math.isclose(sr.reward, expected, rel_tol=1e-5, abs_tol=1e-7), (
            f"step {sr.step_idx}: got {sr.reward}, expected {expected}"
        )
