"""Tests for async_rl_trainer.py — 12 tests."""

from __future__ import annotations

import pytest
import torch
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.async_rl_trainer import (
    AsyncRLConfig,
    AsyncRLTrainer,
    DoubleSidedIS,
    Trajectory,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

RL_CFG = AsyncRLConfig(
    eps_low=0.2,
    eps_high=0.2,
    max_staleness=2,
    group_size=2,
)


def make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(MODEL_CFG)


def make_trainer() -> AsyncRLTrainer:
    model = make_model()
    return AsyncRLTrainer(model, RL_CFG)


def make_trajectory(
    seq_len: int = 8,
    rollout_version: int = 0,
) -> Trajectory:
    token_ids = torch.randint(0, MODEL_CFG.vocab_size, (seq_len,))
    log_probs_rollout = torch.randn(seq_len - 1)
    rewards = torch.randn(seq_len - 1)
    return Trajectory(
        token_ids=token_ids,
        log_probs_rollout=log_probs_rollout,
        rewards=rewards,
        rollout_version=rollout_version,
    )


# ---------------------------------------------------------------------------
# Test 1: AsyncRLConfig dataclass instantiates
# ---------------------------------------------------------------------------


def test_async_rl_config_instantiates():
    cfg = AsyncRLConfig()
    assert cfg.eps_low == 0.2
    assert cfg.eps_high == 0.2
    assert cfg.max_staleness == 4
    assert cfg.min_trajectory_len == 1
    assert cfg.group_size == 4
    assert cfg.gamma == 1.0


# ---------------------------------------------------------------------------
# Test 2: Trajectory dataclass instantiates
# ---------------------------------------------------------------------------


def test_trajectory_instantiates():
    traj = make_trajectory()
    assert isinstance(traj.token_ids, torch.Tensor)
    assert isinstance(traj.log_probs_rollout, torch.Tensor)
    assert isinstance(traj.rewards, torch.Tensor)
    assert isinstance(traj.rollout_version, int)
    assert traj.current_version == 0


# ---------------------------------------------------------------------------
# Test 3: AsyncRLTrainer instantiates
# ---------------------------------------------------------------------------


def test_async_rl_trainer_instantiates():
    trainer = make_trainer()
    assert trainer._version == 0
    assert isinstance(trainer.config, AsyncRLConfig)


# ---------------------------------------------------------------------------
# Test 4: DoubleSidedIS.importance_ratio returns correct shape
# ---------------------------------------------------------------------------


def test_importance_ratio_shape():
    is_fn = DoubleSidedIS(eps_low=0.2, eps_high=0.2)
    T = 10
    log_pi_theta = torch.randn(T)
    log_pi_rollout = torch.randn(T)
    ratio = is_fn.importance_ratio(log_pi_theta, log_pi_rollout)
    assert ratio.shape == (T,), f"Expected shape ({T},), got {ratio.shape}"


# ---------------------------------------------------------------------------
# Test 5: IS ratio is 1.0 when log probs are identical
# ---------------------------------------------------------------------------


def test_importance_ratio_identical_log_probs():
    is_fn = DoubleSidedIS(eps_low=0.2, eps_high=0.2)
    T = 8
    log_probs = torch.randn(T)
    ratio = is_fn.importance_ratio(log_probs, log_probs)
    assert torch.allclose(ratio, torch.ones(T), atol=1e-6), f"Expected all 1.0, got {ratio}"


# ---------------------------------------------------------------------------
# Test 6: DoubleSidedIS.clip_ratio clamps values outside the range
# ---------------------------------------------------------------------------


def test_clip_ratio_clamps_correctly():
    is_fn = DoubleSidedIS(eps_low=0.2, eps_high=0.2)
    # Values: below lower bound, inside range, above upper bound
    ratio = torch.tensor([0.5, 1.0, 1.5])
    clipped = is_fn.clip_ratio(ratio)
    lo = 1.0 - 0.2
    hi = 1.0 + 0.2
    assert clipped[0].item() == pytest.approx(lo, abs=1e-6), "Below lo not clamped"
    assert clipped[1].item() == pytest.approx(1.0, abs=1e-6), "In-range value altered"
    assert clipped[2].item() == pytest.approx(hi, abs=1e-6), "Above hi not clamped"


# ---------------------------------------------------------------------------
# Test 7: _is_stale returns True when staleness exceeded
# ---------------------------------------------------------------------------


def test_is_stale_returns_true_when_exceeded():
    trainer = make_trainer()
    # Set current version so that staleness = max_staleness + 1
    trainer._version = 0
    traj = make_trajectory(rollout_version=0)
    # Manually set current_version to exceed max_staleness
    traj.current_version = RL_CFG.max_staleness + 1
    assert trainer._is_stale(traj) is True


# ---------------------------------------------------------------------------
# Test 8: _filter_trajectories removes stale trajectories
# ---------------------------------------------------------------------------


def test_filter_removes_stale_trajectories():
    trainer = make_trainer()
    trainer._version = 10  # high current version

    fresh_traj = make_trajectory(rollout_version=9)  # staleness = 1 (ok)
    stale_traj = make_trajectory(rollout_version=0)  # staleness = 10 > max_staleness=2

    filtered = trainer._filter_trajectories([fresh_traj, stale_traj])
    assert len(filtered) == 1
    assert filtered[0].rollout_version == 9


# ---------------------------------------------------------------------------
# Test 9: _group_reward_baseline returns zero-mean advantages per group
# ---------------------------------------------------------------------------


def test_group_reward_baseline_zero_mean():
    trainer = make_trainer()
    # Build 2 trajectories (1 group of size 2 per RL_CFG.group_size=2)
    t1 = make_trajectory()
    t2 = make_trajectory()
    # Fix rewards to known values
    t1.rewards = torch.tensor([1.0, 3.0])  # mean = 2.0
    t2.rewards = torch.tensor([5.0, 7.0])  # mean = 6.0

    advantages = trainer._group_reward_baseline([t1, t2])
    adv_sum = advantages[0] + advantages[1]
    assert torch.isclose(adv_sum, torch.tensor(0.0), atol=1e-5), (
        f"Group advantages do not sum to zero: {adv_sum}"
    )


# ---------------------------------------------------------------------------
# Test 10: compute_log_probs returns Tensor of shape (T-1,)
# ---------------------------------------------------------------------------


def test_compute_log_probs_shape():
    trainer = make_trainer()
    T = 8
    token_ids = torch.randint(0, MODEL_CFG.vocab_size, (T,))
    with torch.no_grad():
        log_probs = trainer.compute_log_probs(token_ids)
    assert log_probs.shape == (T - 1,), f"Expected shape ({T - 1},), got {log_probs.shape}"


# ---------------------------------------------------------------------------
# Test 11: train_step returns dict with 'loss' key
# ---------------------------------------------------------------------------


def test_train_step_returns_loss_key():
    trainer = make_trainer()
    optimizer = optim.SGD(trainer.model.parameters(), lr=1e-3)
    trajectories = [make_trajectory(rollout_version=0) for _ in range(4)]
    result = trainer.train_step(trajectories, optimizer)
    assert "loss" in result, f"'loss' key missing from result: {result.keys()}"
    assert "n_trajectories" in result
    assert "mean_advantage" in result


# ---------------------------------------------------------------------------
# Test 12: Loss is finite
# ---------------------------------------------------------------------------


def test_train_step_loss_is_finite():
    torch.manual_seed(42)
    trainer = make_trainer()
    optimizer = optim.SGD(trainer.model.parameters(), lr=1e-3)
    trajectories = [make_trajectory(rollout_version=0) for _ in range(4)]
    result = trainer.train_step(trajectories, optimizer)
    assert torch.isfinite(torch.tensor(result["loss"])), f"Loss is not finite: {result['loss']}"
