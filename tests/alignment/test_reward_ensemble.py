"""Tests for reward_ensemble — uncertainty-weighted ensemble reward model."""
import torch
import pytest

from src.alignment.reward_ensemble import (
    EnsembleConfig,
    RewardHead,
    EnsembleRewardModel,
    aggregate_rewards,
    detect_ood_samples,
    EnsembleTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Constants and shared fixtures
# ---------------------------------------------------------------------------

N_MEMBERS = 3
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
    return EnsembleConfig(n_members=N_MEMBERS, aggregation="mean")


@pytest.fixture
def ensemble_model(tiny_backbone, ensemble_cfg):
    torch.manual_seed(42)
    return EnsembleRewardModel(tiny_backbone, ensemble_cfg, D_MODEL)


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ---------------------------------------------------------------------------
# 1. test_ensemble_config_defaults
# ---------------------------------------------------------------------------

def test_ensemble_config_defaults():
    """EnsembleConfig must expose correct default field values."""
    cfg = EnsembleConfig()
    assert cfg.n_members == 4
    assert cfg.aggregation == "mean"
    assert cfg.uncertainty_threshold == 0.5


# ---------------------------------------------------------------------------
# 2. test_reward_head_forward_shape
# ---------------------------------------------------------------------------

def test_reward_head_forward_shape():
    """RewardHead.forward must return shape (B,) for a (B, T, D) input."""
    torch.manual_seed(42)
    head = RewardHead(D_MODEL)
    hidden = torch.randn(B, T, D_MODEL)
    out = head(hidden)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. test_reward_head_output_scalar
# ---------------------------------------------------------------------------

def test_reward_head_output_scalar():
    """RewardHead with a single sample must return shape (1,)."""
    torch.manual_seed(42)
    head = RewardHead(D_MODEL)
    hidden = torch.randn(1, T, D_MODEL)
    out = head(hidden)
    assert out.shape == (1,), f"Expected (1,), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. test_ensemble_reward_model_forward_shapes
# ---------------------------------------------------------------------------

def test_ensemble_reward_model_forward_shapes(ensemble_model, input_ids):
    """EnsembleRewardModel.forward must return (rewards, uncertainty), both (B,)."""
    ensemble_model.eval()
    with torch.no_grad():
        rewards, uncertainty = ensemble_model(input_ids)
    assert rewards.shape == (B,), f"rewards shape: {rewards.shape}"
    assert uncertainty.shape == (B,), f"uncertainty shape: {uncertainty.shape}"
    assert torch.isfinite(rewards).all()
    assert torch.isfinite(uncertainty).all()


# ---------------------------------------------------------------------------
# 5. test_ensemble_reward_model_get_all_rewards_shape
# ---------------------------------------------------------------------------

def test_ensemble_reward_model_get_all_rewards_shape(ensemble_model, input_ids):
    """get_all_rewards must return shape (B, n_members)."""
    ensemble_model.eval()
    with torch.no_grad():
        all_rewards = ensemble_model.get_all_rewards(input_ids)
    assert all_rewards.shape == (B, N_MEMBERS), (
        f"Expected ({B}, {N_MEMBERS}), got {all_rewards.shape}"
    )


# ---------------------------------------------------------------------------
# 6. test_aggregate_rewards_mean
# ---------------------------------------------------------------------------

def test_aggregate_rewards_mean():
    """'mean' aggregation must equal arithmetic mean across members."""
    cfg = EnsembleConfig(n_members=3, aggregation="mean")
    rewards = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])  # (2, 3)
    agg, std = aggregate_rewards(rewards, cfg)

    expected_mean = torch.tensor([3.0, 4.0])
    assert torch.allclose(agg, expected_mean, atol=1e-5), (
        f"Mean mismatch: {agg} vs {expected_mean}"
    )
    assert agg.shape == (B,)
    assert std.shape == (B,)


# ---------------------------------------------------------------------------
# 7. test_aggregate_rewards_min
# ---------------------------------------------------------------------------

def test_aggregate_rewards_min():
    """'min' aggregation must return the minimum across members."""
    cfg = EnsembleConfig(n_members=3, aggregation="min")
    rewards = torch.tensor([[1.0, 3.0, 5.0], [6.0, 4.0, 2.0]])  # (2, 3)
    agg, std = aggregate_rewards(rewards, cfg)

    expected_min = torch.tensor([1.0, 2.0])
    assert torch.allclose(agg, expected_min, atol=1e-5), (
        f"Min mismatch: {agg} vs {expected_min}"
    )
    assert agg.shape == (B,)
    assert std.shape == (B,)


# ---------------------------------------------------------------------------
# 8. test_aggregate_rewards_uncertainty_weighted
# ---------------------------------------------------------------------------

def test_aggregate_rewards_uncertainty_weighted():
    """'uncertainty_weighted' aggregation must return valid finite rewards."""
    cfg = EnsembleConfig(n_members=3, aggregation="uncertainty_weighted")
    torch.manual_seed(42)
    rewards = torch.randn(B, N_MEMBERS)
    agg, std = aggregate_rewards(rewards, cfg)

    assert agg.shape == (B,), f"Expected ({B},), got {agg.shape}"
    assert std.shape == (B,)
    assert torch.isfinite(agg).all(), "uncertainty_weighted produced non-finite rewards"


# ---------------------------------------------------------------------------
# 9. test_detect_ood_samples_mask
# ---------------------------------------------------------------------------

def test_detect_ood_samples_mask():
    """detect_ood_samples must return correct boolean mask (True where > threshold)."""
    uncertainty = torch.tensor([0.1, 0.6, 0.4, 0.9])
    threshold = 0.5
    mask = detect_ood_samples(uncertainty, threshold)

    expected = torch.tensor([False, True, False, True])
    assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"
    assert mask.shape == (4,)
    assert torch.equal(mask, expected), f"Mask mismatch: {mask} vs {expected}"


# ---------------------------------------------------------------------------
# 10. test_ensemble_trainer_train_step_keys
# ---------------------------------------------------------------------------

def test_ensemble_trainer_train_step_keys(ensemble_model, input_ids):
    """train_step must return a dict with 'loss', 'mean_reward_gap', 'mean_uncertainty'."""
    cfg = EnsembleConfig(n_members=N_MEMBERS)
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=1e-3)
    trainer = EnsembleTrainer(ensemble_model, optimizer, cfg)

    rejected_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    result = trainer.train_step(input_ids, rejected_ids)

    assert "loss" in result, "Missing key 'loss'"
    assert "mean_reward_gap" in result, "Missing key 'mean_reward_gap'"
    assert "mean_uncertainty" in result, "Missing key 'mean_uncertainty'"


# ---------------------------------------------------------------------------
# 11. test_ensemble_trainer_loss_positive
# ---------------------------------------------------------------------------

def test_ensemble_trainer_loss_positive(ensemble_model, input_ids):
    """Bradley-Terry loss must be positive (BCE-based)."""
    cfg = EnsembleConfig(n_members=N_MEMBERS)
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=1e-3)
    trainer = EnsembleTrainer(ensemble_model, optimizer, cfg)

    rejected_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    result = trainer.train_step(input_ids, rejected_ids)

    assert result["loss"] > 0, f"Expected loss > 0, got {result['loss']}"
    import math
    assert math.isfinite(result["loss"]), "Loss is not finite"


# ---------------------------------------------------------------------------
# 12. test_ensemble_members_diversity
# ---------------------------------------------------------------------------

def test_ensemble_members_diversity(ensemble_model, input_ids):
    """Different reward heads must produce different reward values (not all identical)."""
    ensemble_model.eval()
    with torch.no_grad():
        all_rewards = ensemble_model.get_all_rewards(input_ids)  # (B, n_members)

    # Check across members for at least one sample
    for b in range(B):
        member_rewards = all_rewards[b]  # (n_members,)
        # All values being exactly equal would indicate no diversity
        if not torch.allclose(
            member_rewards, member_rewards[0].expand_as(member_rewards), atol=1e-6
        ):
            return  # found diversity — test passes

    pytest.fail(
        f"All {N_MEMBERS} reward heads produced identical rewards for every sample. "
        f"Rewards: {all_rewards}"
    )
