"""Tests for token-level dense reward signals (src/alignment/token_reward.py)."""

import math

import pytest
import torch
import torch.optim as optim

from src.alignment.token_reward import (
    TokenRewardConfig,
    TokenRewardModel,
    TokenRewardTrainer,
    compute_gae,
    compute_returns,
    shape_rewards,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
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
def backbone(small_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture(scope="module")
def reward_model(backbone):
    return TokenRewardModel(backbone, d_model=64)


@pytest.fixture(scope="module")
def policy(small_cfg):
    torch.manual_seed(7)
    return AureliusTransformer(small_cfg)


@pytest.fixture(scope="module")
def trainer(policy, reward_model):
    cfg = TokenRewardConfig()
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    return TokenRewardTrainer(policy, reward_model, optimizer, cfg)


# ---------------------------------------------------------------------------
# Test 1: TokenRewardConfig defaults
# ---------------------------------------------------------------------------


def test_token_reward_config_defaults():
    cfg = TokenRewardConfig()
    assert cfg.reward_type == "dense"
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.normalize is True
    assert cfg.clip_reward == 10.0


# ---------------------------------------------------------------------------
# Test 2: TokenRewardModel output shape (B, T)
# ---------------------------------------------------------------------------


def test_token_reward_model_output_shape(reward_model):
    B, T = 2, 8
    input_ids = torch.randint(0, 256, (B, T))
    out = reward_model(input_ids)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: TokenRewardModel output is finite
# ---------------------------------------------------------------------------


def test_token_reward_model_output_finite(reward_model):
    input_ids = torch.randint(0, 256, (2, 8))
    out = reward_model(input_ids)
    assert torch.isfinite(out).all(), "TokenRewardModel output contains non-finite values"


# ---------------------------------------------------------------------------
# Test 4: compute_returns shape (B, T)
# ---------------------------------------------------------------------------


def test_compute_returns_shape():
    B, T = 3, 10
    rewards = torch.randn(B, T)
    returns = compute_returns(rewards, gamma=0.99)
    assert returns.shape == (B, T), f"Expected ({B}, {T}), got {returns.shape}"


# ---------------------------------------------------------------------------
# Test 5: compute_returns last position equals last reward
# ---------------------------------------------------------------------------


def test_compute_returns_last_position():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    returns = compute_returns(rewards, gamma=0.99)
    # G_{T-1} = r_{T-1}
    assert torch.isclose(returns[0, -1], rewards[0, -1]), (
        f"Last return {returns[0, -1]} != last reward {rewards[0, -1]}"
    )


# ---------------------------------------------------------------------------
# Test 6: compute_returns first position > last (positive rewards)
# ---------------------------------------------------------------------------


def test_compute_returns_first_greater_than_last():
    rewards = torch.ones(1, 5)  # all positive rewards
    returns = compute_returns(rewards, gamma=0.99)
    # G_0 accumulates all future rewards, G_{T-1} = r_{T-1} only
    assert returns[0, 0].item() > returns[0, -1].item(), (
        f"First return {returns[0, 0]} should be > last return {returns[0, -1]}"
    )


# ---------------------------------------------------------------------------
# Test 7: compute_gae shape (B, T)
# ---------------------------------------------------------------------------


def test_compute_gae_shape():
    B, T = 2, 12
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    adv = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    assert adv.shape == (B, T), f"Expected ({B}, {T}), got {adv.shape}"


# ---------------------------------------------------------------------------
# Test 8: compute_gae with lam=1 and zero values matches discounted returns
# ---------------------------------------------------------------------------


def test_compute_gae_lam1_zero_values_equals_returns():
    torch.manual_seed(0)
    B, T = 2, 6
    rewards = torch.rand(B, T)
    values = torch.zeros(B, T)

    adv = compute_gae(rewards, values, gamma=0.99, lam=1.0)
    ret = compute_returns(rewards, gamma=0.99)

    # With lam=1 and V=0: A_t = r_t + gamma*V_{t+1} - V_t + gamma*1*A_{t+1}
    # = r_t + gamma*A_{t+1}, which recursively equals the discounted return
    assert torch.allclose(adv, ret, atol=1e-5), (
        f"GAE with lam=1, values=0 should match discounted returns.\n"
        f"Max diff: {(adv - ret).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 9: shape_rewards clips to [-clip, clip]
# ---------------------------------------------------------------------------


def test_shape_rewards_clips():
    cfg = TokenRewardConfig(clip_reward=5.0, normalize=False)
    rewards = torch.tensor([[100.0, -200.0, 3.0, -3.0]])
    shaped = shape_rewards(rewards, cfg)
    assert shaped.max().item() <= 5.0 + 1e-6
    assert shaped.min().item() >= -5.0 - 1e-6


# ---------------------------------------------------------------------------
# Test 10: shape_rewards normalize: mean ≈ 0 per sequence
# ---------------------------------------------------------------------------


def test_shape_rewards_normalize_mean():
    cfg = TokenRewardConfig(clip_reward=100.0, normalize=True)
    torch.manual_seed(1)
    rewards = torch.randn(3, 16)
    shaped = shape_rewards(rewards, cfg)
    seq_means = shaped.mean(dim=1)
    assert torch.allclose(seq_means, torch.zeros_like(seq_means), atol=1e-5), (
        f"Expected per-sequence mean ≈ 0, got {seq_means}"
    )


# ---------------------------------------------------------------------------
# Test 11: shape_rewards normalize: std ≈ 1 per sequence
# ---------------------------------------------------------------------------


def test_shape_rewards_normalize_std():
    cfg = TokenRewardConfig(clip_reward=100.0, normalize=True)
    torch.manual_seed(2)
    rewards = torch.randn(3, 16)
    shaped = shape_rewards(rewards, cfg)
    seq_stds = shaped.std(dim=1)
    assert torch.allclose(seq_stds, torch.ones_like(seq_stds), atol=1e-4), (
        f"Expected per-sequence std ≈ 1, got {seq_stds}"
    )


# ---------------------------------------------------------------------------
# Test 12: train_step returns required keys
# ---------------------------------------------------------------------------


def test_train_step_returns_required_keys(trainer):
    input_ids = torch.randint(0, 256, (2, 8))
    result = trainer.train_step(input_ids)
    required_keys = {"loss", "mean_reward", "mean_advantage", "mean_return"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"


# ---------------------------------------------------------------------------
# Test 13: train_step loss is finite
# ---------------------------------------------------------------------------


def test_train_step_loss_is_finite(trainer):
    input_ids = torch.randint(0, 256, (2, 8))
    result = trainer.train_step(input_ids)
    assert math.isfinite(result["loss"]), f"Loss is not finite: {result['loss']}"


# ---------------------------------------------------------------------------
# Test 14: train_step mean_reward is float
# ---------------------------------------------------------------------------


def test_train_step_mean_reward_is_float(trainer):
    input_ids = torch.randint(0, 256, (2, 8))
    result = trainer.train_step(input_ids)
    assert isinstance(result["mean_reward"], float), (
        f"Expected float, got {type(result['mean_reward'])}"
    )


# ---------------------------------------------------------------------------
# Test 15: compute_gae when values=0: advantage equals discounted future rewards
# ---------------------------------------------------------------------------


def test_compute_gae_zero_values_equals_returns():
    torch.manual_seed(99)
    B, T = 2, 8
    rewards = torch.rand(B, T) * 2.0  # positive rewards
    values = torch.zeros(B, T)

    adv = compute_gae(rewards, values, gamma=0.99, lam=1.0)
    ret = compute_returns(rewards, gamma=0.99)

    # When V=0 and lam=1: A_t should equal G_t (discounted return from t)
    assert torch.allclose(adv, ret, atol=1e-5), (
        f"GAE with values=0 should equal discounted returns.\nadv[0]: {adv[0]}\nret[0]: {ret[0]}"
    )
