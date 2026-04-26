"""Tests for the outcome-supervised reward model (ORM)."""

from __future__ import annotations

import pytest
import torch

from src.alignment.outcome_reward import (
    ORMConfig,
    OutcomeRewardModel,
    RewardHead,
    calibrate_rewards,
    ensemble_uncertainty,
    reward_loss,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
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


@pytest.fixture(scope="module")
def tiny_backbone():
    torch.manual_seed(42)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture(scope="module")
def orm_config():
    return ORMConfig(
        d_model=64,
        n_layers_head=2,
        dropout=0.0,
        reward_scale=1.0,
        use_mean_pooling=True,
        n_ensemble=1,
    )


@pytest.fixture(scope="module")
def orm(tiny_backbone, orm_config):
    torch.manual_seed(0)
    return OutcomeRewardModel(tiny_backbone, orm_config)


# ---------------------------------------------------------------------------
# Test 1: ORMConfig defaults
# ---------------------------------------------------------------------------


def test_ormconfig_defaults():
    cfg = ORMConfig()
    assert cfg.d_model == 512
    assert cfg.n_layers_head == 2
    assert cfg.dropout == 0.1
    assert cfg.reward_scale == 1.0
    assert cfg.use_mean_pooling is True
    assert cfg.n_ensemble == 1


# ---------------------------------------------------------------------------
# Test 2: RewardHead output shape (B, 1)
# ---------------------------------------------------------------------------


def test_reward_head_output_shape():
    head = RewardHead(d_model=64, n_layers=2, dropout=0.0)
    x = torch.randn(4, 64)
    out = head(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: RewardHead with different n_layers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 3, 5])
def test_reward_head_different_n_layers(n_layers):
    head = RewardHead(d_model=32, n_layers=n_layers, dropout=0.0)
    x = torch.randn(2, 32)
    out = head(x)
    assert out.shape == (2, 1), f"n_layers={n_layers}: expected (2, 1), got {out.shape}"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 4: OutcomeRewardModel forward output shape (B, 1)
# ---------------------------------------------------------------------------


def test_orm_forward_output_shape(orm):
    ids = torch.randint(0, 256, (3, 16))
    out = orm(ids)
    assert out.shape == (3, 1), f"Expected (3, 1), got {out.shape}"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 5: OutcomeRewardModel ensemble n=3 still gives (B, 1) mean
# ---------------------------------------------------------------------------


def test_orm_ensemble_n3(tiny_backbone):
    cfg = ORMConfig(d_model=64, n_layers_head=2, dropout=0.0, n_ensemble=3)
    model = OutcomeRewardModel(tiny_backbone, cfg)
    ids = torch.randint(0, 256, (4, 8))
    out = model(ids)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 6: Backbone is frozen (no grad flows to backbone params after backward)
# ---------------------------------------------------------------------------


def test_orm_backbone_frozen(tiny_backbone):
    """Backbone parameters must not accumulate gradients when ORM is trained."""
    cfg = ORMConfig(d_model=64, n_layers_head=2, dropout=0.0)
    model = OutcomeRewardModel(tiny_backbone, cfg)

    ids = torch.randint(0, 256, (2, 8))
    out = model(ids)  # (2, 1)
    loss = out.mean()
    loss.backward()

    # All backbone parameters should have grad=None (frozen via torch.no_grad())
    for name, param in model.backbone.named_parameters():
        assert param.grad is None, (
            f"Backbone param '{name}' has a gradient — backbone should be frozen."
        )


# ---------------------------------------------------------------------------
# Test 7: reward_loss returns a scalar
# ---------------------------------------------------------------------------


def test_reward_loss_is_scalar():
    predicted = torch.tensor([1.0, 0.5, 0.8, 0.2])
    chosen_mask = torch.tensor([True, False, True, False])
    loss = reward_loss(predicted, chosen_mask)
    assert loss.ndim == 0, "reward_loss must return a scalar"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 8: reward_loss correct direction (chosen > rejected → lower loss)
# ---------------------------------------------------------------------------


def test_reward_loss_direction():
    """When chosen rewards are clearly higher, loss should be lower."""
    # Case A: chosen >> rejected
    pred_good = torch.tensor([2.0, 0.0, 2.0, 0.0])
    mask = torch.tensor([True, False, True, False])
    loss_good = reward_loss(pred_good, mask)

    # Case B: chosen << rejected (inverted)
    pred_bad = torch.tensor([0.0, 2.0, 0.0, 2.0])
    loss_bad = reward_loss(pred_bad, mask)

    assert loss_good.item() < loss_bad.item(), (
        f"Expected good-ordering loss ({loss_good.item():.4f}) < "
        f"bad-ordering loss ({loss_bad.item():.4f})"
    )


# ---------------------------------------------------------------------------
# Test 9: reward_loss with odd batch drops last sample gracefully
# ---------------------------------------------------------------------------


def test_reward_loss_odd_batch():
    """Odd batch size should not raise; last sample is dropped."""
    predicted = torch.tensor([1.0, 0.2, 0.9])  # 3 samples
    chosen_mask = torch.tensor([True, False, True])
    # After dropping the last sample: [1.0, 0.2] with mask [True, False]
    loss = reward_loss(predicted, chosen_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 10: calibrate_rewards mean≈0, std≈1
# ---------------------------------------------------------------------------


def test_calibrate_rewards_normalized():
    torch.manual_seed(7)
    rewards = torch.randn(1000)
    cal = calibrate_rewards(rewards)
    assert abs(cal.mean().item()) < 1e-4, f"Mean not near 0: {cal.mean().item()}"
    assert abs(cal.std(unbiased=False).item() - 1.0) < 1e-4, (
        f"Std not near 1: {cal.std(unbiased=False).item()}"
    )


# ---------------------------------------------------------------------------
# Test 11: calibrate_rewards constant input returns zeros
# ---------------------------------------------------------------------------


def test_calibrate_rewards_constant():
    rewards = torch.full((10,), 3.14)
    cal = calibrate_rewards(rewards)
    assert torch.all(cal == 0.0), f"Constant input should return zeros, got {cal}"


# ---------------------------------------------------------------------------
# Test 12: ensemble_uncertainty correct keys and shapes
# ---------------------------------------------------------------------------


def test_ensemble_uncertainty_keys_and_shapes():
    B, n_ensemble = 5, 4
    rewards = torch.randn(B, n_ensemble)
    result = ensemble_uncertainty(rewards)

    assert set(result.keys()) == {"mean", "std", "disagreement"}, (
        f"Unexpected keys: {result.keys()}"
    )
    assert result["mean"].shape == (B,), f"mean shape: {result['mean'].shape}"
    assert result["std"].shape == (B,), f"std shape: {result['std'].shape}"
    assert result["disagreement"].ndim == 0, "disagreement should be scalar"


# ---------------------------------------------------------------------------
# Test 13: ensemble_uncertainty disagreement = 0 when all heads agree
# ---------------------------------------------------------------------------


def test_ensemble_uncertainty_zero_disagreement():
    """When all heads produce identical rewards, disagreement should be 0."""
    B, n_ensemble = 4, 3
    # All heads produce the same value per sample
    rewards = torch.ones(B, n_ensemble) * torch.arange(B).float().unsqueeze(1)
    result = ensemble_uncertainty(rewards)
    assert result["disagreement"].item() < 1e-6, (
        f"Expected disagreement ≈ 0, got {result['disagreement'].item()}"
    )
