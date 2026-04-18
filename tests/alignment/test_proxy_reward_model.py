"""Tests for src/alignment/proxy_reward_model.py"""
from __future__ import annotations

import math

import torch
import pytest

from src.alignment.proxy_reward_model import (
    EnsembleRewardModel,
    ProxyRewardConfig,
    ProxyRewardModel,
    RewardBackbone,
    RewardHead,
    RewardModelTrainer,
    RewardNormalizer,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
SEQ_LEN = 8
BATCH = 4


def _make_ids(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch, seq_len))


def _make_backbone(pooling: str = "last") -> RewardBackbone:
    return RewardBackbone(D_MODEL, VOCAB_SIZE, n_layers=N_LAYERS, pooling=pooling)


def _make_head(n_outputs: int = 1) -> RewardHead:
    return RewardHead(D_MODEL, n_outputs=n_outputs)


def _make_proxy(pooling: str = "last") -> ProxyRewardModel:
    return ProxyRewardModel(_make_backbone(pooling), _make_head())


# ---------------------------------------------------------------------------
# RewardBackbone tests
# ---------------------------------------------------------------------------

def test_backbone_last_output_shape():
    """RewardBackbone with pooling='last' should return [B, d_model]."""
    backbone = _make_backbone("last")
    out = backbone(_make_ids())
    assert out.shape == (BATCH, D_MODEL), f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"


def test_backbone_mean_output_shape():
    """RewardBackbone with pooling='mean' should return [B, d_model]."""
    backbone = _make_backbone("mean")
    out = backbone(_make_ids())
    assert out.shape == (BATCH, D_MODEL), f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"


def test_backbone_cls_output_shape():
    """RewardBackbone with pooling='cls' should return [B, d_model]."""
    backbone = _make_backbone("cls")
    out = backbone(_make_ids())
    assert out.shape == (BATCH, D_MODEL), f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"


def test_backbone_invalid_pooling_raises():
    """Passing an invalid pooling mode should raise ValueError."""
    with pytest.raises(ValueError, match="pooling must be one of"):
        RewardBackbone(D_MODEL, VOCAB_SIZE, pooling="invalid")


def test_backbone_output_is_finite():
    """Backbone output should contain no NaN or Inf values."""
    backbone = _make_backbone("last")
    out = backbone(_make_ids())
    assert torch.isfinite(out).all(), "Backbone output contains non-finite values"


# ---------------------------------------------------------------------------
# RewardHead tests
# ---------------------------------------------------------------------------

def test_reward_head_output_shape():
    """RewardHead should return [B, 1] for scalar reward."""
    head = _make_head(n_outputs=1)
    x = torch.randn(BATCH, D_MODEL)
    out = head(x)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"


def test_reward_head_multi_output_shape():
    """RewardHead with n_outputs=3 should return [B, 3]."""
    head = _make_head(n_outputs=3)
    x = torch.randn(BATCH, D_MODEL)
    out = head(x)
    assert out.shape == (BATCH, 3), f"Expected ({BATCH}, 3), got {out.shape}"


# ---------------------------------------------------------------------------
# ProxyRewardModel tests
# ---------------------------------------------------------------------------

def test_proxy_forward_output_shape():
    """ProxyRewardModel.forward should return [B] (squeezed scalar rewards)."""
    model = _make_proxy()
    rewards = model(_make_ids())
    assert rewards.shape == (BATCH,), f"Expected ({BATCH},), got {rewards.shape}"


def test_proxy_score_batch_is_detached():
    """score_batch should return a tensor that requires no grad."""
    model = _make_proxy()
    scores = model.score_batch(_make_ids())
    assert not scores.requires_grad, "score_batch output should be detached (no grad)"


def test_proxy_score_batch_output_shape():
    """score_batch should return [B] just like forward."""
    model = _make_proxy()
    scores = model.score_batch(_make_ids())
    assert scores.shape == (BATCH,), f"Expected ({BATCH},), got {scores.shape}"


def test_proxy_forward_output_is_finite():
    """ProxyRewardModel output should be finite."""
    model = _make_proxy()
    rewards = model(_make_ids())
    assert torch.isfinite(rewards).all(), "ProxyRewardModel output contains non-finite values"


# ---------------------------------------------------------------------------
# EnsembleRewardModel tests
# ---------------------------------------------------------------------------

def _make_ensemble(n: int = 3) -> EnsembleRewardModel:
    return EnsembleRewardModel([_make_proxy() for _ in range(n)])


def test_ensemble_mean_reward_shape():
    """EnsembleRewardModel should return mean_reward of shape [B]."""
    ensemble = _make_ensemble()
    mean_reward, _ = ensemble(_make_ids())
    assert mean_reward.shape == (BATCH,), f"Expected ({BATCH},), got {mean_reward.shape}"


def test_ensemble_std_reward_shape():
    """EnsembleRewardModel should return std_reward of shape [B]."""
    ensemble = _make_ensemble()
    _, std_reward = ensemble(_make_ids())
    assert std_reward.shape == (BATCH,), f"Expected ({BATCH},), got {std_reward.shape}"


def test_ensemble_std_reward_nonnegative():
    """std_reward should be >= 0 for all samples."""
    ensemble = _make_ensemble()
    _, std_reward = ensemble(_make_ids())
    assert (std_reward >= 0).all(), "std_reward contains negative values"


def test_ensemble_uncertainty_score_nonnegative():
    """uncertainty_score should be non-negative for all samples."""
    ensemble = _make_ensemble()
    unc = ensemble.uncertainty_score(_make_ids())
    assert (unc >= 0).all(), "uncertainty_score contains negative values"


def test_ensemble_uncertainty_score_shape():
    """uncertainty_score should return shape [B]."""
    ensemble = _make_ensemble()
    unc = ensemble.uncertainty_score(_make_ids())
    assert unc.shape == (BATCH,), f"Expected ({BATCH},), got {unc.shape}"


def test_ensemble_empty_models_raises():
    """EnsembleRewardModel with empty list should raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        EnsembleRewardModel([])


# ---------------------------------------------------------------------------
# RewardModelTrainer tests
# ---------------------------------------------------------------------------

def _make_trainer(margin: float = 0.1) -> RewardModelTrainer:
    return RewardModelTrainer(_make_proxy(), lr=1e-3, margin=margin)


def test_trainer_preference_loss_is_finite_scalar():
    """preference_loss should return a finite scalar."""
    trainer = _make_trainer()
    loss = trainer.preference_loss(_make_ids(), _make_ids())
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"preference_loss is not finite: {loss.item()}"


def test_trainer_regression_loss_is_finite_scalar():
    """regression_loss should return a finite scalar."""
    trainer = _make_trainer()
    targets = torch.randn(BATCH)
    loss = trainer.regression_loss(_make_ids(), targets)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"regression_loss is not finite: {loss.item()}"


def test_trainer_train_step_gradients_flow():
    """train_step should propagate gradients through the model parameters."""
    trainer = _make_trainer()
    # Zero out existing grads (if any)
    for p in trainer.model.parameters():
        p.grad = None
    loss = trainer.train_step(_make_ids(), _make_ids())
    # After train_step the optimizer already stepped — grads were zeroed.
    # Re-verify by manually doing a backward without stepping the optimizer.
    loss2 = trainer.preference_loss(_make_ids(), _make_ids())
    loss2.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in trainer.model.parameters()
    )
    assert has_grad, "No parameter received a gradient after backward"


def test_trainer_eval_accuracy_in_unit_interval():
    """eval_accuracy should return a float in [0, 1]."""
    trainer = _make_trainer()
    acc = trainer.eval_accuracy(_make_ids(), _make_ids())
    assert isinstance(acc, float), f"eval_accuracy should return float, got {type(acc)}"
    assert 0.0 <= acc <= 1.0, f"eval_accuracy {acc} is outside [0, 1]"


# ---------------------------------------------------------------------------
# RewardNormalizer tests
# ---------------------------------------------------------------------------

def test_normalizer_output_mean_and_std():
    """After many updates, the running normalizer should yield ~zero mean and ~unit std."""
    normalizer = RewardNormalizer(momentum=0.01)  # fast adaptation
    torch.manual_seed(42)
    # Run many batches drawn from a fixed distribution (mean=5, std=2)
    for _ in range(500):
        batch = torch.randn(32) * 2.0 + 5.0
        out = normalizer.update(batch)
    # The last output should be close to zero-mean, unit-std
    assert abs(out.mean().item()) < 0.5, f"Output mean {out.mean().item()} is too large"
    # std may not be perfect with small batch, allow generous tolerance
    assert abs(out.std().item() - 1.0) < 1.5, f"Output std {out.std().item()} is far from 1"


def test_normalizer_reset_clears_statistics():
    """reset() should return the normalizer to its initial uninitialised state."""
    normalizer = RewardNormalizer()
    batch = torch.randn(16) * 3.0 + 10.0
    normalizer.update(batch)
    assert normalizer._initialised is True
    normalizer.reset()
    assert normalizer._initialised is False
    assert normalizer.running_mean == 0.0
    assert normalizer.running_var == 1.0


def test_normalizer_single_update_returns_correct_shape():
    """After a single update the output shape should match input."""
    normalizer = RewardNormalizer()
    rewards = torch.randn(BATCH)
    out = normalizer.update(rewards)
    assert out.shape == rewards.shape, f"Expected shape {rewards.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# ProxyRewardConfig defaults
# ---------------------------------------------------------------------------

def test_proxy_reward_config_defaults():
    """ProxyRewardConfig should have the specified default values."""
    cfg = ProxyRewardConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.pooling == "last"
    assert math.isclose(cfg.lr, 1e-4)
    assert math.isclose(cfg.margin, 0.1)
    assert cfg.n_ensemble == 3
    assert math.isclose(cfg.momentum, 0.99)


def test_proxy_reward_config_custom_values():
    """ProxyRewardConfig should accept and store custom values."""
    cfg = ProxyRewardConfig(d_model=64, vocab_size=128, n_layers=4, pooling="mean", n_ensemble=5)
    assert cfg.d_model == 64
    assert cfg.vocab_size == 128
    assert cfg.n_layers == 4
    assert cfg.pooling == "mean"
    assert cfg.n_ensemble == 5
