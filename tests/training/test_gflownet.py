"""Tests for GFlowNet training module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.gflownet import (
    GFlowNetConfig,
    GFlowNetTrainer,
    Trajectory,
    compute_backward_prob,
    compute_trajectory_balance_loss,
    sample_trajectory,
)

# ---------------------------------------------------------------------------
# Tiny config helpers
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

GFN_CFG = GFlowNetConfig(
    n_trajectories=2,
    max_seq_len=4,
    temperature=1.0,
    lambda_tb=1.0,
    epsilon=0.0,
)

PROMPT_IDS = [1, 2, 3]


def make_model() -> AureliusTransformer:
    return AureliusTransformer(TINY_CFG)


def make_trainer(model=None, config=None):
    if model is None:
        model = make_model()
    if config is None:
        config = GFN_CFG
    optimizer = AdamW(model.parameters(), lr=1e-4)

    def reward_fn(seq):
        return 1.0

    return GFlowNetTrainer(model, optimizer, reward_fn, config)


# ---------------------------------------------------------------------------
# 1. GFlowNetConfig defaults
# ---------------------------------------------------------------------------


def test_gflownet_config_defaults():
    cfg = GFlowNetConfig()
    assert cfg.n_trajectories == 8
    assert cfg.max_seq_len == 16
    assert cfg.temperature == 1.0
    assert cfg.lambda_tb == 1.0
    assert cfg.epsilon == 0.05


# ---------------------------------------------------------------------------
# 2. Trajectory fields
# ---------------------------------------------------------------------------


def test_trajectory_fields():
    traj = Trajectory(states=[[1, 2], [1, 2, 3]], actions=[3])
    assert traj.states == [[1, 2], [1, 2, 3]]
    assert traj.actions == [3]
    assert traj.log_pf == 0.0
    assert traj.log_pb == 0.0
    assert traj.reward == 0.0


# ---------------------------------------------------------------------------
# 3. compute_trajectory_balance_loss returns scalar
# ---------------------------------------------------------------------------


def test_tb_loss_returns_scalar():
    log_z = torch.tensor(0.0)
    log_pf = torch.tensor([-1.0, -2.0])
    log_pb = torch.tensor([-1.5, -2.5])
    log_reward = torch.tensor([0.5, 0.8])
    loss = compute_trajectory_balance_loss(log_z, log_pf, log_pb, log_reward)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


# ---------------------------------------------------------------------------
# 4. compute_trajectory_balance_loss zero when perfectly balanced
# ---------------------------------------------------------------------------


def test_tb_loss_zero_when_balanced():
    # log_z + log_pf == log_pb + log_reward => loss == 0
    log_pf = torch.tensor([1.0, 2.0])
    log_pb = torch.tensor([0.5, 1.0])
    log_reward = torch.tensor([0.5, 1.0])
    torch.tensor(0.0)
    # Adjust log_z so balance holds: log_z = log_pb + log_reward - log_pf
    log_z_balanced = (log_pb + log_reward - log_pf).mean()
    loss = compute_trajectory_balance_loss(log_z_balanced, log_pf, log_pb, log_reward)
    assert loss.item() < 1e-10


# ---------------------------------------------------------------------------
# 5. compute_trajectory_balance_loss positive when unbalanced
# ---------------------------------------------------------------------------


def test_tb_loss_positive_when_unbalanced():
    log_z = torch.tensor(5.0)  # deliberately wrong
    log_pf = torch.tensor([-1.0, -2.0])
    log_pb = torch.tensor([-1.5, -2.5])
    log_reward = torch.tensor([0.5, 0.8])
    loss = compute_trajectory_balance_loss(log_z, log_pf, log_pb, log_reward)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 6. sample_trajectory returns Trajectory
# ---------------------------------------------------------------------------


def test_sample_trajectory_returns_trajectory():
    model = make_model()
    traj = sample_trajectory(model, PROMPT_IDS, GFN_CFG, reward_fn=lambda s: 1.0)
    assert isinstance(traj, Trajectory)


# ---------------------------------------------------------------------------
# 7. sample_trajectory states list non-empty
# ---------------------------------------------------------------------------


def test_sample_trajectory_states_nonempty():
    model = make_model()
    traj = sample_trajectory(model, PROMPT_IDS, GFN_CFG, reward_fn=lambda s: 1.0)
    assert len(traj.states) > 0
    assert traj.states[0] == PROMPT_IDS


# ---------------------------------------------------------------------------
# 8. sample_trajectory actions list length = states - 1
# ---------------------------------------------------------------------------


def test_sample_trajectory_actions_length():
    model = make_model()
    traj = sample_trajectory(model, PROMPT_IDS, GFN_CFG, reward_fn=lambda s: 1.0)
    assert len(traj.actions) == len(traj.states) - 1


# ---------------------------------------------------------------------------
# 9. compute_backward_prob returns negative float
# ---------------------------------------------------------------------------


def test_compute_backward_prob_negative():
    traj = Trajectory(
        states=[[1, 2], [1, 2, 3], [1, 2, 3, 4]],
        actions=[3, 4],
    )
    log_pb = compute_backward_prob(traj)
    assert isinstance(log_pb, float)
    assert log_pb < 0.0


# ---------------------------------------------------------------------------
# 10. GFlowNetTrainer.log_z is nn.Parameter
# ---------------------------------------------------------------------------


def test_trainer_log_z_is_parameter():
    trainer = make_trainer()
    assert isinstance(trainer.log_z, nn.Parameter)


# ---------------------------------------------------------------------------
# 11. GFlowNetTrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_train_step_returns_required_keys():
    trainer = make_trainer()
    result = trainer.train_step(PROMPT_IDS)
    assert "loss" in result
    assert "mean_reward" in result
    assert "log_z" in result
    assert "n_trajectories" in result


# ---------------------------------------------------------------------------
# 12. GFlowNetTrainer.train_step loss is finite
# ---------------------------------------------------------------------------


def test_train_step_loss_is_finite():
    trainer = make_trainer()
    result = trainer.train_step(PROMPT_IDS)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# 13. GFlowNetTrainer.generate_diverse returns list
# ---------------------------------------------------------------------------


def test_generate_diverse_returns_list():
    trainer = make_trainer()
    result = trainer.generate_diverse(PROMPT_IDS, n=3)
    assert isinstance(result, list)
    for seq in result:
        assert isinstance(seq, list)


# ---------------------------------------------------------------------------
# 14. GFlowNetTrainer.generate_diverse length = n
# ---------------------------------------------------------------------------


def test_generate_diverse_length():
    trainer = make_trainer()
    n = 4
    result = trainer.generate_diverse(PROMPT_IDS, n=n)
    assert len(result) == n
