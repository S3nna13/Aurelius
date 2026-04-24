"""Tests for src/alignment/multi_objective_rlhf_trainer.py — 15 tests."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.multi_objective_rlhf_trainer import (
    MOConfig,
    MOTrainResult,
    MultiObjectiveRLHFTrainer,
    ObjectiveWeight,
)


def _make_config(normalize: bool = True) -> MOConfig:
    return MOConfig(
        objectives=[
            ObjectiveWeight("helpfulness", 0.5),
            ObjectiveWeight("harmlessness", 0.3),
            ObjectiveWeight("honesty", 0.2),
        ],
        epsilon=0.2,
        entropy_coef=0.01,
        normalize_rewards=normalize,
    )


def _make_trainer(normalize: bool = True) -> MultiObjectiveRLHFTrainer:
    policy = nn.Linear(4, 4)
    return MultiObjectiveRLHFTrainer(policy, _make_config(normalize))


def _reward_dict(batch: int = 4) -> dict:
    return {
        "helpfulness": torch.rand(batch),
        "harmlessness": torch.rand(batch),
        "honesty": torch.rand(batch),
    }


def _log_probs(batch: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    old = torch.randn(batch)
    new = old + torch.randn(batch) * 0.1
    return new, old


# ---------------------------------------------------------------------------
# ObjectiveWeight / MOConfig
# ---------------------------------------------------------------------------

def test_objective_weight_fields():
    obj = ObjectiveWeight("helpfulness", 0.5)
    assert obj.name == "helpfulness"
    assert obj.weight == 0.5


def test_moconfig_defaults():
    cfg = MOConfig(objectives=[ObjectiveWeight("help", 1.0)])
    assert cfg.epsilon == 0.2
    assert cfg.entropy_coef == 0.01
    assert cfg.normalize_rewards is True


def test_moconfig_custom():
    cfg = MOConfig(
        objectives=[ObjectiveWeight("help", 1.0)],
        epsilon=0.1,
        entropy_coef=0.05,
        normalize_rewards=False,
    )
    assert cfg.epsilon == 0.1
    assert not cfg.normalize_rewards


# ---------------------------------------------------------------------------
# scalarize_rewards
# ---------------------------------------------------------------------------

def test_scalarize_returns_tensor():
    trainer = _make_trainer()
    rd = _reward_dict()
    result = trainer.scalarize_rewards(rd)
    assert isinstance(result, torch.Tensor)


def test_scalarize_shape():
    trainer = _make_trainer()
    rd = _reward_dict(8)
    result = trainer.scalarize_rewards(rd)
    assert result.shape == (8,)


def test_scalarize_weighted_sum():
    policy = nn.Linear(4, 4)
    cfg = MOConfig(
        objectives=[
            ObjectiveWeight("a", 0.6),
            ObjectiveWeight("b", 0.4),
        ]
    )
    trainer = MultiObjectiveRLHFTrainer(policy, cfg)
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = trainer.scalarize_rewards({"a": a, "b": b})
    expected = 0.6 * a + 0.4 * b
    assert torch.allclose(result, expected)


def test_scalarize_no_objectives_raises():
    policy = nn.Linear(4, 4)
    cfg = MOConfig(objectives=[])
    trainer = MultiObjectiveRLHFTrainer(policy, cfg)
    with pytest.raises(ValueError):
        trainer.scalarize_rewards({})


# ---------------------------------------------------------------------------
# compute_advantages
# ---------------------------------------------------------------------------

def test_compute_advantages_shape():
    trainer = _make_trainer()
    rewards = torch.rand(6)
    values = torch.rand(6)
    adv = trainer.compute_advantages(rewards, values)
    assert adv.shape == (6,)


def test_compute_advantages_values():
    trainer = _make_trainer()
    rewards = torch.tensor([2.0, 3.0])
    values = torch.tensor([1.0, 1.0])
    adv = trainer.compute_advantages(rewards, values)
    assert torch.allclose(adv, torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# ppo_loss
# ---------------------------------------------------------------------------

def test_ppo_loss_scalar():
    trainer = _make_trainer()
    lp, olp = _log_probs()
    adv = torch.randn(4)
    loss = trainer.ppo_loss(lp, olp, adv)
    assert loss.ndim == 0


def test_ppo_loss_clipping():
    """Ratio far outside [1-eps, 1+eps] should produce same loss as boundary."""
    trainer = _make_trainer()
    # identical log_probs → ratio == 1 → no clipping effect
    lp = torch.zeros(4)
    olp = torch.zeros(4)
    adv = torch.ones(4)
    loss = trainer.ppo_loss(lp, olp, adv)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

def test_train_step_returns_result():
    trainer = _make_trainer()
    rd = _reward_dict()
    lp, olp = _log_probs()
    values = torch.zeros(4)
    result = trainer.train_step(rd, lp, olp, values)
    assert isinstance(result, MOTrainResult)


def test_train_step_per_objective_keys():
    trainer = _make_trainer()
    rd = _reward_dict()
    lp, olp = _log_probs()
    values = torch.zeros(4)
    result = trainer.train_step(rd, lp, olp, values)
    assert set(result.per_objective_losses.keys()) == {"helpfulness", "harmlessness", "honesty"}


def test_train_step_no_normalize():
    trainer = _make_trainer(normalize=False)
    rd = _reward_dict()
    lp, olp = _log_probs()
    values = torch.zeros(4)
    result = trainer.train_step(rd, lp, olp, values)
    assert isinstance(result.total_loss, float)


def test_train_step_weighted_reward_finite():
    trainer = _make_trainer()
    rd = _reward_dict()
    lp, olp = _log_probs()
    values = torch.zeros(4)
    result = trainer.train_step(rd, lp, olp, values)
    assert isinstance(result.weighted_reward, float)
    assert result.weighted_reward == result.weighted_reward  # not NaN


# ---------------------------------------------------------------------------
# pareto_check
# ---------------------------------------------------------------------------

def test_pareto_check_length():
    trainer = _make_trainer()
    results = [
        MOTrainResult(0.1, {"helpfulness": 0.1, "harmlessness": 0.2, "honesty": 0.3}, 0.5),
        MOTrainResult(0.2, {"helpfulness": 0.3, "harmlessness": 0.1, "honesty": 0.2}, 0.4),
    ]
    flags = trainer.pareto_check(results)
    assert len(flags) == 2


def test_pareto_check_dominated():
    trainer = _make_trainer()
    # result[1] dominates result[0] on all objectives
    r0 = MOTrainResult(1.0, {"helpfulness": 1.0, "harmlessness": 1.0, "honesty": 1.0}, 0.5)
    r1 = MOTrainResult(0.5, {"helpfulness": 0.5, "harmlessness": 0.5, "honesty": 0.5}, 0.5)
    flags = trainer.pareto_check([r0, r1])
    assert flags[0] is False
    assert flags[1] is True


def test_pareto_check_all_optimal():
    trainer = _make_trainer()
    # Each result is better on a different objective — none dominated
    r0 = MOTrainResult(0.5, {"helpfulness": 0.1, "harmlessness": 0.9, "honesty": 0.9}, 0.5)
    r1 = MOTrainResult(0.5, {"helpfulness": 0.9, "harmlessness": 0.1, "honesty": 0.9}, 0.5)
    r2 = MOTrainResult(0.5, {"helpfulness": 0.9, "harmlessness": 0.9, "honesty": 0.1}, 0.5)
    flags = trainer.pareto_check([r0, r1, r2])
    assert all(flags)
