"""Tests for reward_ensemble — uncertainty-aware ensemble reward model."""
import math

import pytest
import torch

from src.alignment.reward_ensemble import (
    EnsembleConfig,
    RewardHead,
    RewardEnsemble,
    MCDropoutReward,
    conservative_reward,
    RewardEnsembleTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Constants and shared fixtures
# ---------------------------------------------------------------------------

N_MODELS = 2
D_MODEL = 64
B = 2
T = 8
VOCAB_SIZE = 256

torch.manual_seed(42)


@pytest.fixture
def tiny_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=D_MODEL,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture
def tiny_backbone(tiny_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(tiny_cfg)


@pytest.fixture
def ensemble_cfg():
    return EnsembleConfig(n_models=N_MODELS, aggregation="mean")


@pytest.fixture
def ensemble_model(tiny_backbone, ensemble_cfg):
    torch.manual_seed(42)
    return RewardEnsemble(tiny_backbone, ensemble_cfg)


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ---------------------------------------------------------------------------
# 1. test_ensemble_config_defaults
# ---------------------------------------------------------------------------

def test_ensemble_config_defaults():
    """EnsembleConfig must expose correct default field values."""
    cfg = EnsembleConfig()
    assert cfg.n_models == 5
    assert cfg.aggregation == "mean"
    assert cfg.ucb_beta == 1.0
    assert cfg.dropout_rate == 0.1
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 2. test_reward_head_output_shape
# ---------------------------------------------------------------------------

def test_reward_head_output_shape():
    """RewardHead.forward must return shape (B,) for a (B, T, D) input."""
    torch.manual_seed(42)
    head = RewardHead(D_MODEL)
    hidden = torch.randn(B, T, D_MODEL)
    out = head(hidden)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_reward_head_output_is_finite
# ---------------------------------------------------------------------------

def test_reward_head_output_is_finite():
    """RewardHead output must contain only finite values."""
    torch.manual_seed(42)
    head = RewardHead(D_MODEL)
    hidden = torch.randn(B, T, D_MODEL)
    out = head(hidden)
    assert torch.isfinite(out).all(), f"Non-finite values in output: {out}"


# ---------------------------------------------------------------------------
# 4. test_reward_ensemble_forward_returns_tuple
# ---------------------------------------------------------------------------

def test_reward_ensemble_forward_returns_tuple(ensemble_model, input_ids):
    """RewardEnsemble.forward must return a 2-tuple."""
    ensemble_model.eval()
    with torch.no_grad():
        result = ensemble_model(input_ids)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"Expected 2-tuple, got {type(result)} of length {len(result)}"
    )


# ---------------------------------------------------------------------------
# 5. test_reward_ensemble_mean_reward_shape
# ---------------------------------------------------------------------------

def test_reward_ensemble_mean_reward_shape(ensemble_model, input_ids):
    """mean_reward from RewardEnsemble.forward must have shape (B,)."""
    ensemble_model.eval()
    with torch.no_grad():
        mean_reward, _ = ensemble_model(input_ids)
    assert mean_reward.shape == (B,), f"Expected ({B},), got {mean_reward.shape}"


# ---------------------------------------------------------------------------
# 6. test_reward_ensemble_std_reward_nonneg
# ---------------------------------------------------------------------------

def test_reward_ensemble_std_reward_nonneg(ensemble_model, input_ids):
    """std_reward from RewardEnsemble.forward must be >= 0."""
    ensemble_model.eval()
    with torch.no_grad():
        _, std_reward = ensemble_model(input_ids)
    assert (std_reward >= 0).all(), f"Negative std: {std_reward}"
    assert std_reward.shape == (B,)


# ---------------------------------------------------------------------------
# 7. test_aggregate_mean
# ---------------------------------------------------------------------------

def test_aggregate_mean(ensemble_model):
    """aggregate with 'mean' must return the arithmetic mean across models."""
    cfg = EnsembleConfig(n_models=3, aggregation="mean")
    ensemble_model.config = cfg
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
    result = ensemble_model.aggregate(rewards)
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(result, expected, atol=1e-5)
    assert result.shape == (B,)


# ---------------------------------------------------------------------------
# 8. test_aggregate_min_lte_mean
# ---------------------------------------------------------------------------

def test_aggregate_min_lte_mean(ensemble_model):
    """aggregate 'min' result must be <= 'mean' result element-wise."""
    rewards = torch.randn(N_MODELS, B)

    cfg_mean = EnsembleConfig(n_models=N_MODELS, aggregation="mean")
    ensemble_model.config = cfg_mean
    mean_agg = ensemble_model.aggregate(rewards)

    cfg_min = EnsembleConfig(n_models=N_MODELS, aggregation="min")
    ensemble_model.config = cfg_min
    min_agg = ensemble_model.aggregate(rewards)

    assert (min_agg <= mean_agg + 1e-6).all(), (
        f"min aggregation exceeds mean: min={min_agg}, mean={mean_agg}"
    )


# ---------------------------------------------------------------------------
# 9. test_uncertainty_keys
# ---------------------------------------------------------------------------

def test_uncertainty_keys(ensemble_model, input_ids):
    """RewardEnsemble.uncertainty must return dict with required keys."""
    ensemble_model.eval()
    with torch.no_grad():
        result = ensemble_model.uncertainty(input_ids)
    required = {"mean", "std", "epistemic", "lower_bound"}
    assert required.issubset(result.keys()), (
        f"Missing keys: {required - result.keys()}"
    )
    for k in required:
        assert result[k].shape == (B,), f"Key '{k}' has wrong shape: {result[k].shape}"


# ---------------------------------------------------------------------------
# 10. test_mc_dropout_predict_shapes
# ---------------------------------------------------------------------------

def test_mc_dropout_predict_shapes():
    """MCDropoutReward.predict must return (mean, std) with shape (B,)."""
    torch.manual_seed(42)
    head = RewardHead(D_MODEL, dropout=0.2)
    mc = MCDropoutReward(head, n_forward=5)
    hidden = torch.randn(B, T, D_MODEL)
    mean, std = mc.predict(hidden)
    assert mean.shape == (B,), f"mean shape {mean.shape}"
    assert std.shape == (B,), f"std shape {std.shape}"


# ---------------------------------------------------------------------------
# 11. test_conservative_reward
# ---------------------------------------------------------------------------

def test_conservative_reward():
    """conservative_reward must equal mean - beta*std."""
    mean = torch.tensor([2.0, 3.0, 1.0])
    std = torch.tensor([0.5, 1.0, 0.2])
    beta = 1.5
    result = conservative_reward(mean, std, beta)
    expected = mean - beta * std
    assert torch.allclose(result, expected, atol=1e-6), (
        f"conservative_reward mismatch: {result} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 12. test_train_step_returns_required_keys
# ---------------------------------------------------------------------------

def test_train_step_returns_required_keys(ensemble_model, input_ids):
    """RewardEnsembleTrainer.train_step must return dict with required keys."""
    cfg = EnsembleConfig(n_models=N_MODELS)
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=1e-3)
    trainer = RewardEnsembleTrainer(ensemble_model, optimizer, cfg)

    rejected_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    result = trainer.train_step(input_ids, rejected_ids)

    required = {"loss", "mean_margin", "agreement"}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


# ---------------------------------------------------------------------------
# 13. test_train_step_loss_is_finite
# ---------------------------------------------------------------------------

def test_train_step_loss_is_finite(ensemble_model, input_ids):
    """RewardEnsembleTrainer.train_step loss must be finite."""
    cfg = EnsembleConfig(n_models=N_MODELS)
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=1e-3)
    trainer = RewardEnsembleTrainer(ensemble_model, optimizer, cfg)

    rejected_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    result = trainer.train_step(input_ids, rejected_ids)

    assert math.isfinite(result["loss"]), f"Non-finite loss: {result['loss']}"


# ---------------------------------------------------------------------------
# 14. test_evaluate_returns_required_keys
# ---------------------------------------------------------------------------

def test_evaluate_returns_required_keys(ensemble_model, input_ids):
    """RewardEnsembleTrainer.evaluate must return dict with required keys."""
    cfg = EnsembleConfig(n_models=N_MODELS)
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=1e-3)
    trainer = RewardEnsembleTrainer(ensemble_model, optimizer, cfg)

    rejected_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    result = trainer.evaluate(input_ids, rejected_ids)

    required = {"loss", "mean_margin", "agreement"}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


# ---------------------------------------------------------------------------
# 15. test_aggregate_ucb_ge_mean
# ---------------------------------------------------------------------------

def test_aggregate_ucb_ge_mean(ensemble_model):
    """aggregate 'ucb' result must be >= 'mean' result element-wise."""
    rewards = torch.randn(N_MODELS, B)

    cfg_mean = EnsembleConfig(n_models=N_MODELS, aggregation="mean")
    ensemble_model.config = cfg_mean
    mean_agg = ensemble_model.aggregate(rewards)

    cfg_ucb = EnsembleConfig(n_models=N_MODELS, aggregation="ucb", ucb_beta=1.0)
    ensemble_model.config = cfg_ucb
    ucb_agg = ensemble_model.aggregate(rewards)

    assert (ucb_agg >= mean_agg - 1e-6).all(), (
        f"ucb aggregation less than mean: ucb={ucb_agg}, mean={mean_agg}"
    )
