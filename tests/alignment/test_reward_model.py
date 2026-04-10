"""Tests for the reward model with Bradley-Terry preference learning."""
from __future__ import annotations

import copy
import math

import pytest
import torch

from src.alignment.reward_model import (
    RewardModelConfig,
    RewardHead,
    RewardModel,
    bradley_terry_loss,
    RewardModelTrainer,
    EnsembleRewardModel,
    build_reward_fn,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )


@pytest.fixture
def backbone(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def rm_cfg():
    return RewardModelConfig(d_model=64, dropout=0.0, margin=0.0, label_smoothing=0.0)


@pytest.fixture
def reward_model(backbone, rm_cfg):
    torch.manual_seed(1)
    return RewardModel(backbone, rm_cfg)


@pytest.fixture
def ids_pair():
    torch.manual_seed(42)
    chosen = torch.randint(0, 256, (4, 16))
    rejected = torch.randint(0, 256, (4, 16))
    return chosen, rejected


# ---------------------------------------------------------------------------
# 1. RewardModelConfig defaults
# ---------------------------------------------------------------------------

def test_reward_model_config_defaults():
    cfg = RewardModelConfig()
    assert cfg.d_model == 64
    assert cfg.dropout == 0.1
    assert cfg.margin == 0.0
    assert cfg.label_smoothing == 0.0


# ---------------------------------------------------------------------------
# 2. RewardHead output shape is (B,)
# ---------------------------------------------------------------------------

def test_reward_head_output_shape():
    head = RewardHead(d_model=64, dropout=0.0)
    hidden = torch.randn(3, 10, 64)
    out = head(hidden)
    assert out.shape == (3,), f"Expected (3,), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. RewardHead output is scalar per sequence, not per token
# ---------------------------------------------------------------------------

def test_reward_head_scalar_per_sequence():
    head = RewardHead(d_model=64, dropout=0.0)
    B, T = 5, 20
    hidden = torch.randn(B, T, 64)
    out = head(hidden)
    # Must be 1-D with length == batch size, not (B, T) or (B, T, 1)
    assert out.ndim == 1
    assert out.shape[0] == B


# ---------------------------------------------------------------------------
# 4. RewardModel output shape is (B,)
# ---------------------------------------------------------------------------

def test_reward_model_output_shape(reward_model):
    ids = torch.randint(0, 256, (3, 16))
    out = reward_model(ids)
    assert out.shape == (3,), f"Expected (3,), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. RewardModel is differentiable
# ---------------------------------------------------------------------------

def test_reward_model_differentiable(reward_model):
    ids = torch.randint(0, 256, (2, 8))
    out = reward_model(ids)
    loss = out.sum()
    loss.backward()
    # At least the reward head should have gradients
    assert reward_model.reward_head.proj.weight.grad is not None


# ---------------------------------------------------------------------------
# 6. bradley_terry_loss returns (Tensor, dict)
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_returns_tensor_and_dict():
    chosen = torch.tensor([1.0, 2.0])
    rejected = torch.tensor([0.0, 0.5])
    result = bradley_terry_loss(chosen, rejected)
    assert isinstance(result, tuple) and len(result) == 2
    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 7. bradley_terry_loss loss is scalar and finite
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_scalar_and_finite():
    chosen = torch.tensor([1.0, 2.0, 0.5])
    rejected = torch.tensor([0.0, 0.5, 1.0])
    loss, _ = bradley_terry_loss(chosen, rejected)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 8. bradley_terry_loss dict has all required keys
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_dict_keys():
    chosen = torch.tensor([1.0])
    rejected = torch.tensor([0.0])
    _, metrics = bradley_terry_loss(chosen, rejected)
    required = {"accuracy", "mean_gap", "mean_chosen_reward", "mean_rejected_reward"}
    assert required.issubset(set(metrics.keys())), f"Missing keys: {required - set(metrics.keys())}"


# ---------------------------------------------------------------------------
# 9. bradley_terry_loss accuracy=1.0 when all chosen > rejected
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_accuracy_all_chosen_wins():
    chosen = torch.tensor([2.0, 3.0, 5.0])
    rejected = torch.tensor([1.0, 1.0, 1.0])
    _, metrics = bradley_terry_loss(chosen, rejected)
    assert metrics["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 10. bradley_terry_loss accuracy=0.0 when all chosen < rejected
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_accuracy_all_rejected_wins():
    chosen = torch.tensor([0.0, 0.5, 0.1])
    rejected = torch.tensor([1.0, 2.0, 3.0])
    _, metrics = bradley_terry_loss(chosen, rejected)
    assert metrics["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 11. bradley_terry_loss with margin > 0 increases loss
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_margin_increases_loss():
    chosen = torch.tensor([1.0, 1.5, 2.0])
    rejected = torch.tensor([0.5, 0.5, 0.5])
    loss_no_margin, _ = bradley_terry_loss(chosen, rejected, margin=0.0)
    loss_with_margin, _ = bradley_terry_loss(chosen, rejected, margin=1.0)
    assert loss_with_margin > loss_no_margin


# ---------------------------------------------------------------------------
# 12. RewardModelTrainer.train_step returns dict with correct keys
# ---------------------------------------------------------------------------

def test_trainer_train_step_returns_correct_keys(reward_model, ids_pair, rm_cfg):
    chosen, rejected = ids_pair
    optimizer = torch.optim.SGD(reward_model.parameters(), lr=1e-3)
    trainer = RewardModelTrainer(reward_model, rm_cfg, optimizer)
    metrics = trainer.train_step(chosen, rejected)
    required = {"loss", "accuracy", "mean_gap", "mean_chosen_reward", "mean_rejected_reward"}
    assert required.issubset(set(metrics.keys()))


# ---------------------------------------------------------------------------
# 13. RewardModelTrainer.evaluate returns dict without updating params
# ---------------------------------------------------------------------------

def test_trainer_evaluate_no_grad_update(reward_model, ids_pair, rm_cfg):
    chosen, rejected = ids_pair
    optimizer = torch.optim.SGD(reward_model.parameters(), lr=1e-3)
    trainer = RewardModelTrainer(reward_model, rm_cfg, optimizer)

    # Snapshot weights before evaluate
    weight_before = reward_model.reward_head.proj.weight.clone().detach()

    metrics = trainer.evaluate(chosen, rejected)

    weight_after = reward_model.reward_head.proj.weight.clone().detach()
    assert torch.equal(weight_before, weight_after), "evaluate() must not update parameters"

    required = {"loss", "accuracy", "mean_gap", "mean_chosen_reward", "mean_rejected_reward"}
    assert required.issubset(set(metrics.keys()))


# ---------------------------------------------------------------------------
# 14. EnsembleRewardModel output shapes are (B,) for both mean and std
# ---------------------------------------------------------------------------

def test_ensemble_reward_model_output_shapes(small_cfg):
    torch.manual_seed(7)
    models = [RewardModel(AureliusTransformer(small_cfg)) for _ in range(3)]
    ensemble = EnsembleRewardModel(models)
    ids = torch.randint(0, 256, (4, 8))
    mean_r, std_r = ensemble(ids)
    assert mean_r.shape == (4,), f"mean shape: {mean_r.shape}"
    assert std_r.shape == (4,), f"std shape: {std_r.shape}"


# ---------------------------------------------------------------------------
# 15. EnsembleRewardModel.uncertainty std=0 when all models identical
# ---------------------------------------------------------------------------

def test_ensemble_uncertainty_zero_for_identical_models(backbone):
    rm = RewardModel(backbone, RewardModelConfig(d_model=64, dropout=0.0))
    # Use same model instance (deep-copied) for all ensemble members
    models = [copy.deepcopy(rm) for _ in range(4)]
    ensemble = EnsembleRewardModel(models)

    ids = torch.randint(0, 256, (3, 8))
    with torch.no_grad():
        uncertainty = ensemble.uncertainty(ids)

    assert uncertainty.shape == (3,)
    assert torch.allclose(uncertainty, torch.zeros(3), atol=1e-5), (
        f"Expected ~0 uncertainty for identical models, got {uncertainty}"
    )


# ---------------------------------------------------------------------------
# Legacy tests (kept for regression)
# ---------------------------------------------------------------------------

def test_build_reward_fn(backbone):
    from unittest.mock import MagicMock
    rm = RewardModel(backbone)
    tok = MagicMock()
    tok.encode = lambda text: [1, 2, 3, 4, 5]
    fn = build_reward_fn(rm, tok)
    result = fn("hello", "world")
    assert isinstance(result, float)
    assert math.isfinite(result)


def test_reward_model_score_returns_float(backbone):
    rm = RewardModel(backbone)
    ids = torch.randint(0, 256, (1, 8))
    s = rm.score(ids)
    assert isinstance(s, float)
    assert math.isfinite(s)
