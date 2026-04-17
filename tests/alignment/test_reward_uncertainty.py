"""Tests for src/alignment/reward_uncertainty.py"""

import pytest
import torch
import torch.nn as nn

from aurelius.alignment.reward_uncertainty import (
    MCDropoutReward,
    DeepEnsembleReward,
    UncertaintyFilter,
    RewardUncertaintyTrainer,
)

D_MODEL = 32
B = 8


def test_mc_dropout_forward_shape():
    model = MCDropoutReward(D_MODEL)
    x = torch.randn(B, D_MODEL)
    out = model(x)
    assert out.shape == (B,)


def test_mc_dropout_forward_finite():
    model = MCDropoutReward(D_MODEL)
    x = torch.randn(B, D_MODEL)
    out = model(x)
    assert torch.isfinite(out).all()


def test_mc_dropout_uncertainty_shape():
    model = MCDropoutReward(D_MODEL, dropout_p=0.3)
    x = torch.randn(B, D_MODEL)
    mean, std = model.predict_with_uncertainty(x, n_samples=10)
    assert mean.shape == (B,)
    assert std.shape == (B,)


def test_mc_dropout_std_nonneg():
    model = MCDropoutReward(D_MODEL, dropout_p=0.3)
    x = torch.randn(B, D_MODEL)
    _, std = model.predict_with_uncertainty(x, n_samples=10)
    assert (std >= 0).all()


def test_mc_dropout_restores_eval_mode():
    model = MCDropoutReward(D_MODEL)
    model.eval()
    model.predict_with_uncertainty(torch.randn(2, D_MODEL), n_samples=5)
    assert not model.training


def test_mc_dropout_gradient_flows():
    model = MCDropoutReward(D_MODEL)
    x = torch.randn(B, D_MODEL, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


def test_ensemble_forward_shape():
    models = [MCDropoutReward(D_MODEL) for _ in range(3)]
    ensemble = DeepEnsembleReward(models)
    x = torch.randn(B, D_MODEL)
    out = ensemble.forward(x)
    assert out.shape == (B,)


def test_ensemble_uncertainty_shape():
    models = [MCDropoutReward(D_MODEL) for _ in range(3)]
    ensemble = DeepEnsembleReward(models)
    x = torch.randn(B, D_MODEL)
    mean, std = ensemble.predict_with_uncertainty(x)
    assert mean.shape == (B,)
    assert std.shape == (B,)


def test_ensemble_single_model_std_zero():
    models = [MCDropoutReward(D_MODEL)]
    ensemble = DeepEnsembleReward(models)
    x = torch.randn(B, D_MODEL)
    _, std = ensemble.predict_with_uncertainty(x)
    assert (std == 0).all()


def test_ensemble_empty_raises():
    with pytest.raises(ValueError):
        DeepEnsembleReward([])


def test_ensemble_update_member():
    models = [MCDropoutReward(D_MODEL) for _ in range(3)]
    ensemble = DeepEnsembleReward(models)
    new_model = MCDropoutReward(D_MODEL)
    ensemble.update_member(1, new_model)
    assert ensemble.models[1] is new_model


def test_ensemble_update_member_out_of_range():
    models = [MCDropoutReward(D_MODEL) for _ in range(2)]
    ensemble = DeepEnsembleReward(models)
    with pytest.raises(IndexError):
        ensemble.update_member(5, MCDropoutReward(D_MODEL))


def test_filter_keeps_low_uncertainty():
    flt = UncertaintyFilter(threshold=0.5)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    uncertainties = torch.tensor([0.1, 0.6, 0.3, 0.9])
    kept_rewards, mask = flt.filter(rewards, uncertainties)
    assert mask[0] and mask[2]
    assert not mask[1] and not mask[3]
    assert kept_rewards.shape == (2,)


def test_filter_calibrate_threshold():
    flt = UncertaintyFilter(threshold=1.0)
    uncertainties = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    t = flt.calibrate_threshold(uncertainties, percentile=90)
    assert isinstance(t, float)
    assert 0.0 <= t <= 1.0


def test_filter_calibrate_empty():
    flt = UncertaintyFilter(threshold=0.5)
    t = flt.calibrate_threshold(torch.tensor([]), percentile=90)
    assert t == 0.0


def test_trainer_step_keys():
    model = MCDropoutReward(D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = RewardUncertaintyTrainer(model, opt)
    x_w = torch.randn(B, D_MODEL)
    x_l = torch.randn(B, D_MODEL)
    stats = trainer.train_step(x_w, x_l)
    assert "loss" in stats
    assert "reward_margin" in stats
    assert "mean_reward_w" in stats
    assert "mean_reward_l" in stats


def test_trainer_step_loss_finite():
    model = MCDropoutReward(D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = RewardUncertaintyTrainer(model, opt)
    stats = trainer.train_step(torch.randn(B, D_MODEL), torch.randn(B, D_MODEL))
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_trainer_step_updates_weights():
    torch.manual_seed(0)
    model = MCDropoutReward(D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = RewardUncertaintyTrainer(model, opt)
    W_before = model.fc1.weight.data.clone()
    trainer.train_step(torch.randn(B, D_MODEL), torch.randn(B, D_MODEL))
    assert not torch.allclose(model.fc1.weight.data, W_before)
