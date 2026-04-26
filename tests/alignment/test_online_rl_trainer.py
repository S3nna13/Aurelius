"""Tests for src/alignment/online_rl_trainer.py (12+ tests)."""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.online_rl_trainer import (
    GAEComputation,
    OnlineRLConfig,
    OnlineRLTrainer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 2  # batch size
T = 8  # time steps for GAE sequences


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config():
    return OnlineRLConfig()


@pytest.fixture()
def trainer(config):
    return OnlineRLTrainer(config=config)


def _make_batch(B: int = B, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    logprobs = torch.randn(B)
    old_logprobs = torch.randn(B)
    advantages = torch.randn(B)
    returns = torch.randn(B)
    values = torch.randn(B)
    entropy = torch.tensor(0.5)
    return {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "returns": returns,
        "values": values,
        "entropy": entropy,
    }


# ---------------------------------------------------------------------------
# OnlineRLConfig tests (1-2)
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Default config should have expected hyper-parameters."""
    cfg = OnlineRLConfig()
    assert cfg.lr == pytest.approx(1e-5)
    assert cfg.kl_coef == pytest.approx(0.1)
    assert cfg.clip_eps == pytest.approx(0.2)
    assert cfg.value_coef == pytest.approx(0.5)
    assert cfg.entropy_coef == pytest.approx(0.01)


def test_config_custom():
    """Custom config values should be stored correctly."""
    cfg = OnlineRLConfig(lr=3e-4, gamma=0.95, gae_lambda=0.9)
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.gamma == pytest.approx(0.95)
    assert cfg.gae_lambda == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# GAEComputation tests (3-7)
# ---------------------------------------------------------------------------


def test_gae_lengths():
    """GAE output lists should have same length as input."""
    cfg = OnlineRLConfig()
    rewards = [1.0] * T
    values = [0.5] * T
    dones = [False] * T
    adv, ret = GAEComputation.compute_gae(rewards, values, dones, cfg)
    assert len(adv) == T
    assert len(ret) == T


def test_gae_finite():
    """GAE advantages and returns should be finite."""
    cfg = OnlineRLConfig()
    rewards = [0.1 * i for i in range(T)]
    values = [0.2 * i for i in range(T)]
    dones = [False] * T
    adv, ret = GAEComputation.compute_gae(rewards, values, dones, cfg)
    assert all(math.isfinite(a) for a in adv)
    assert all(math.isfinite(r) for r in ret)


def test_gae_done_zeroes_bootstrap():
    """Done=True at last step: advantage should not bootstrap from next value."""
    cfg = OnlineRLConfig(gamma=0.99, gae_lambda=1.0)
    rewards = [1.0]
    values = [0.5]
    dones = [True]
    adv, ret = GAEComputation.compute_gae(rewards, values, dones, cfg)
    # delta = r + gamma * 0 * mask - V = 1.0 - 0.5 = 0.5
    assert adv[0] == pytest.approx(0.5, abs=1e-6)


def test_gae_returns_equal_advantage_plus_value():
    """returns[t] should equal advantages[t] + values[t]."""
    cfg = OnlineRLConfig()
    rewards = [float(i) for i in range(T)]
    values = [0.3 * i for i in range(T)]
    dones = [False] * (T - 1) + [True]
    adv, ret = GAEComputation.compute_gae(rewards, values, dones, cfg)
    for t in range(T):
        assert ret[t] == pytest.approx(adv[t] + values[t], abs=1e-5)


def test_gae_single_step():
    """Single-step episode: advantage = r - V (no discounting needed)."""
    cfg = OnlineRLConfig(gamma=0.99, gae_lambda=1.0)
    rewards = [2.0]
    values = [1.0]
    dones = [True]
    adv, ret = GAEComputation.compute_gae(rewards, values, dones, cfg)
    assert adv[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_policy_loss tests (8-9)
# ---------------------------------------------------------------------------


def test_policy_loss_scalar(trainer):
    """compute_policy_loss should return a scalar tensor."""
    logp = torch.randn(B)
    old_logp = torch.randn(B)
    adv = torch.randn(B)
    loss = trainer.compute_policy_loss(logp, old_logp, adv, trainer.config.clip_eps)
    assert loss.ndim == 0


def test_policy_loss_gradient_flows(trainer):
    """Gradient should flow through the policy loss."""
    logp = torch.randn(B, requires_grad=True)
    old_logp = torch.randn(B)
    adv = torch.randn(B)
    loss = trainer.compute_policy_loss(logp, old_logp, adv, trainer.config.clip_eps)
    loss.backward()
    assert logp.grad is not None
    assert torch.isfinite(logp.grad).all()


# ---------------------------------------------------------------------------
# compute_value_loss tests (10)
# ---------------------------------------------------------------------------


def test_value_loss_scalar_finite(trainer):
    """compute_value_loss should return a finite scalar."""
    values = torch.randn(B, requires_grad=True)
    returns = torch.randn(B)
    loss = trainer.compute_value_loss(values, returns)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())
    loss.backward()
    assert values.grad is not None


# ---------------------------------------------------------------------------
# train_step tests (11-14)
# ---------------------------------------------------------------------------


def test_train_step_returns_expected_keys(trainer):
    """train_step must return dict with policy_loss, value_loss, entropy, total_loss, kl."""
    batch = _make_batch()
    result = trainer.train_step(batch)
    for key in ("policy_loss", "value_loss", "entropy", "total_loss", "kl"):
        assert key in result, f"Missing key: {key}"


def test_train_step_values_finite(trainer):
    """All values returned by train_step should be finite tensors."""
    batch = _make_batch()
    result = trainer.train_step(batch)
    for key, val in result.items():
        assert math.isfinite(val.item()), f"{key} is not finite: {val}"


def test_train_step_total_loss_gradient(trainer):
    """total_loss from train_step should support backward."""
    batch = _make_batch(seed=7)
    # Make logprobs require grad so we can test backward
    batch["logprobs"] = batch["logprobs"].detach().requires_grad_(True)
    batch["values"] = batch["values"].detach().requires_grad_(True)
    result = trainer.train_step(batch)
    result["total_loss"].backward()
    assert batch["logprobs"].grad is not None


def test_train_step_kl_approx(trainer):
    """KL should be approximately 0 when logprobs == old_logprobs."""
    torch.manual_seed(5)
    lp = torch.randn(B)
    batch = {
        "logprobs": lp,
        "old_logprobs": lp.clone(),  # same => KL ~ 0
        "advantages": torch.randn(B),
        "returns": torch.randn(B),
        "values": torch.randn(B),
        "entropy": torch.tensor(0.3),
    }
    result = trainer.train_step(batch)
    assert result["kl"].item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Registry test (15)
# ---------------------------------------------------------------------------


def test_registry_entry():
    """ALIGNMENT_REGISTRY['online_rl'] should be OnlineRLTrainer."""
    from src.alignment import ALIGNMENT_REGISTRY  # noqa: F401

    assert "online_rl" in ALIGNMENT_REGISTRY
    assert ALIGNMENT_REGISTRY["online_rl"] is OnlineRLTrainer
