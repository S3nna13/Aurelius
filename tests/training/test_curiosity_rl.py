"""
Tests for src/training/curiosity_rl.py

Uses: d_obs=16, d_encoding=8, n_actions=8, B=4, T=6, vocab_size=16
"""

import torch
import pytest

from src.training.curiosity_rl import (
    RandomNetworkDistillation,
    InverseDynamicsModel,
    ForwardDynamicsModel,
    ICMModule,
    RewardShaper,
    TokenLevelCuriosity,
    CuriosityConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_OBS = 16
D_ENCODING = 8
N_ACTIONS = 8
B = 4
T = 6
VOCAB_SIZE = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rnd():
    return RandomNetworkDistillation(d_obs=D_OBS, d_encoding=D_ENCODING)


@pytest.fixture
def obs():
    return torch.randn(B, D_OBS)


@pytest.fixture
def actions():
    return torch.randint(0, N_ACTIONS, (B,))


@pytest.fixture
def icm():
    return ICMModule(d_obs=D_OBS, n_actions=N_ACTIONS)


@pytest.fixture
def reward_shaper():
    return RewardShaper(intrinsic_weight=0.1, extrinsic_weight=1.0)


@pytest.fixture
def tlc():
    return TokenLevelCuriosity(vocab_size=VOCAB_SIZE, d_model=32)


# ---------------------------------------------------------------------------
# RND tests
# ---------------------------------------------------------------------------

def test_rnd_intrinsic_reward_shape(rnd, obs):
    """RND intrinsic_reward returns tensor of shape [B]."""
    reward = rnd.intrinsic_reward(obs)
    assert reward.shape == (B,), f"Expected shape ({B},), got {reward.shape}"


def test_rnd_intrinsic_reward_non_negative(rnd, obs):
    """RND intrinsic rewards are non-negative (squared error)."""
    reward = rnd.intrinsic_reward(obs)
    assert (reward >= 0).all(), "All RND intrinsic rewards must be non-negative"


def test_rnd_intrinsic_reward_decreases_after_update(rnd):
    """
    After many updates on the same batch the predictor should converge,
    reducing the intrinsic reward for familiar observations.
    """
    obs_familiar = torch.randn(B, D_OBS)
    reward_before = rnd.intrinsic_reward(obs_familiar).mean().item()

    for _ in range(200):
        rnd.update_predictor(obs_familiar)

    reward_after = rnd.intrinsic_reward(obs_familiar).mean().item()
    assert reward_after < reward_before, (
        f"Expected reward to decrease after training; before={reward_before:.4f}, "
        f"after={reward_after:.4f}"
    )


def test_rnd_update_predictor_finite_loss(rnd, obs):
    """RND update_predictor returns a finite scalar loss."""
    loss = rnd.update_predictor(obs)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# InverseDynamicsModel tests
# ---------------------------------------------------------------------------

def test_inverse_dynamics_output_shape():
    """InverseDynamicsModel forward produces [B, n_actions]."""
    model = InverseDynamicsModel(d_obs=D_OBS, n_actions=N_ACTIONS)
    o = torch.randn(B, D_OBS)
    no = torch.randn(B, D_OBS)
    logits = model(o, no)
    assert logits.shape == (B, N_ACTIONS), (
        f"Expected ({B}, {N_ACTIONS}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# ForwardDynamicsModel tests
# ---------------------------------------------------------------------------

def test_forward_dynamics_output_shape():
    """ForwardDynamicsModel forward produces [B, d_obs]."""
    model = ForwardDynamicsModel(d_obs=D_OBS, n_actions=N_ACTIONS)
    o = torch.randn(B, D_OBS)
    a = torch.randint(0, N_ACTIONS, (B,))
    pred = model(o, a)
    assert pred.shape == (B, D_OBS), (
        f"Expected ({B}, {D_OBS}), got {pred.shape}"
    )


# ---------------------------------------------------------------------------
# ICMModule tests
# ---------------------------------------------------------------------------

def test_icm_intrinsic_reward_shape(icm, obs, actions):
    """ICMModule.intrinsic_reward returns [B]."""
    next_obs = torch.randn(B, D_OBS)
    reward = icm.intrinsic_reward(obs, next_obs, actions)
    assert reward.shape == (B,), f"Expected ({B},), got {reward.shape}"


def test_icm_intrinsic_reward_non_negative(icm, obs, actions):
    """ICMModule intrinsic rewards are non-negative."""
    next_obs = torch.randn(B, D_OBS)
    reward = icm.intrinsic_reward(obs, next_obs, actions)
    assert (reward >= 0).all(), "All ICM intrinsic rewards must be non-negative"


def test_icm_total_loss_finite_scalar(icm, obs, actions):
    """ICMModule.total_loss returns a finite scalar."""
    next_obs = torch.randn(B, D_OBS)
    policy_loss = torch.tensor(1.0, requires_grad=True)
    loss = icm.total_loss(obs, next_obs, actions, policy_loss)
    assert loss.ndim == 0, "total_loss should be a scalar"
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"


def test_icm_total_loss_backward(icm, obs, actions):
    """Gradients flow through ICMModule.total_loss."""
    obs_req = obs.detach().requires_grad_(True)
    next_obs = torch.randn(B, D_OBS)
    policy_loss = torch.tensor(1.0, requires_grad=True)
    loss = icm.total_loss(obs_req, next_obs, actions, policy_loss)
    loss.backward()
    # At least the policy_loss grad must be populated
    assert policy_loss.grad is not None, "policy_loss should have gradient"
    # ICM parameters should have gradients
    param_grads = [
        p.grad for p in icm.parameters() if p.grad is not None
    ]
    assert len(param_grads) > 0, "ICM parameters should have gradients after backward"


# ---------------------------------------------------------------------------
# RewardShaper tests
# ---------------------------------------------------------------------------

def test_reward_shaper_shape(reward_shaper):
    """RewardShaper.shape returns [B]."""
    ext = torch.randn(B)
    intr = torch.randn(B).abs()
    shaped = reward_shaper.shape(ext, intr)
    assert shaped.shape == (B,), f"Expected ({B},), got {shaped.shape}"


def test_reward_shaper_zero_intrinsic_weight():
    """With intrinsic_weight=0, shaped reward equals extrinsic reward."""
    shaper = RewardShaper(intrinsic_weight=0.0, extrinsic_weight=1.0)
    ext = torch.tensor([1.0, 2.0, 3.0, 4.0])
    intr = torch.tensor([100.0, 200.0, 300.0, 400.0])
    shaped = shaper.shape(ext, intr)
    assert torch.allclose(shaped, ext), (
        "With intrinsic_weight=0, shaped rewards should equal extrinsic rewards"
    )


def test_reward_shaper_normalize_mean_near_zero():
    """After normalization, mean of rewards should be near 0."""
    shaper = RewardShaper()
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    normed = shaper.normalize_rewards(rewards)
    assert abs(normed.mean().item()) < 1.0, (
        f"Normalized mean should be close to 0, got {normed.mean().item():.4f}"
    )


def test_reward_shaper_normalize_std_near_one():
    """After normalization, std of rewards should be near 1."""
    shaper = RewardShaper()
    # Use a fresh shaper with a sufficiently spread distribution
    rewards = torch.arange(1.0, 17.0)   # 16 values, good spread
    normed = shaper.normalize_rewards(rewards)
    std_val = normed.std(unbiased=False).item()
    assert 0.1 < std_val < 10.0, (
        f"Normalized std should be in a reasonable range near 1, got {std_val:.4f}"
    )


# ---------------------------------------------------------------------------
# TokenLevelCuriosity tests
# ---------------------------------------------------------------------------

def test_token_surprise_shape(tlc):
    """TokenLevelCuriosity.token_surprise returns [B, T]."""
    logits = torch.randn(B, T, VOCAB_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    surprise = tlc.token_surprise(logits, input_ids)
    assert surprise.shape == (B, T), f"Expected ({B}, {T}), got {surprise.shape}"


def test_sequence_novelty_range(tlc):
    """TokenLevelCuriosity.sequence_novelty values are in [0, 1]."""
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    history = set()
    novelty = tlc.sequence_novelty(input_ids, history)
    assert novelty.shape == (B,), f"Expected ({B},), got {novelty.shape}"
    assert (novelty >= 0).all() and (novelty <= 1).all(), (
        "Novelty scores must be in [0, 1]"
    )


def test_sequence_novelty_decreases_after_update(tlc):
    """After updating history with the same sequences, novelty should decrease."""
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    history: set = set()

    novelty_before = tlc.sequence_novelty(input_ids, history).mean().item()
    # Add all trigrams from this batch to history
    tlc.update_history(input_ids)
    # Pass the internal history to sequence_novelty
    novelty_after = tlc.sequence_novelty(input_ids, tlc._ngram_history).mean().item()
    assert novelty_after < novelty_before or novelty_before == 0.0, (
        "Novelty should decrease (or stay 0) after seen sequences are added to history"
    )


def test_token_surprise_positive(tlc):
    """Token surprise (negative log prob) should be non-negative."""
    logits = torch.randn(B, T, VOCAB_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    surprise = tlc.token_surprise(logits, input_ids)
    assert (surprise >= 0).all(), "Token surprise (neg log-prob) should be non-negative"


# ---------------------------------------------------------------------------
# CuriosityConfig tests
# ---------------------------------------------------------------------------

def test_curiosity_config_defaults():
    """CuriosityConfig has expected default values."""
    cfg = CuriosityConfig()
    assert cfg.d_obs == 32
    assert cfg.d_encoding == 16
    assert cfg.n_actions == 16
    assert cfg.beta == pytest.approx(0.2)
    assert cfg.eta == pytest.approx(0.01)
    assert cfg.intrinsic_weight == pytest.approx(0.1)
    assert cfg.vocab_size == 64
    assert cfg.d_model == 32
