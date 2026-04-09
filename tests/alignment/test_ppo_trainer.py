"""Tests for src/alignment/ppo_trainer.py."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.ppo_trainer import (
    PPOConfig,
    PPOTrainer,
    Rollout,
    compute_entropy_bonus,
    compute_gae_advantages,
    ppo_policy_loss,
    ppo_value_loss,
)

# ---------------------------------------------------------------------------
# Shared tiny config
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

PROMPT_LEN = 4
MAX_NEW_TOKENS = 4


def _make_model():
    return AureliusTransformer(TINY_CFG)


def _make_value_fn():
    """Simple value function: nn.Linear(vocab_size, 1) on mean-pooled logits."""
    return nn.Linear(TINY_CFG.vocab_size, 1)


def _make_prompt(B=1):
    return torch.randint(0, TINY_CFG.vocab_size, (B, PROMPT_LEN))


def _make_trainer():
    policy = _make_model()
    ref_policy = _make_model()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    value_fn = _make_value_fn()
    reward_fn = lambda tokens: 1.0
    policy_opt = optim.SGD(policy.parameters(), lr=1e-3)
    value_opt = optim.SGD(value_fn.parameters(), lr=1e-3)
    cfg = PPOConfig(n_epochs=1, minibatch_size=4)
    return PPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        value_fn=value_fn,
        reward_fn=reward_fn,
        policy_optimizer=policy_opt,
        value_optimizer=value_opt,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# 1. PPOConfig defaults
# ---------------------------------------------------------------------------

def test_ppo_config_defaults():
    cfg = PPOConfig()
    assert cfg.clip_epsilon == 0.2
    assert cfg.value_clip == 0.2
    assert cfg.entropy_coeff == 0.01
    assert cfg.value_coeff == 0.5
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.n_epochs == 4
    assert cfg.minibatch_size == 4
    assert cfg.max_grad_norm == 0.5
    assert cfg.normalize_advantages is True
    assert cfg.kl_target == 0.01


# ---------------------------------------------------------------------------
# 2. compute_gae_advantages output shapes (T,) for both
# ---------------------------------------------------------------------------

def test_gae_advantages_shapes():
    T = 10
    cfg = PPOConfig()
    rewards = torch.ones(T)
    values = torch.zeros(T)
    dones = torch.zeros(T)
    adv, ret = compute_gae_advantages(rewards, values, dones, cfg)
    assert adv.shape == (T,), f"Expected ({T},), got {adv.shape}"
    assert ret.shape == (T,), f"Expected ({T},), got {ret.shape}"


# ---------------------------------------------------------------------------
# 3. compute_gae_advantages with zero rewards gives negative advantages (value baseline)
# ---------------------------------------------------------------------------

def test_gae_zero_rewards_negative_advantages():
    T = 5
    cfg = PPOConfig()
    rewards = torch.zeros(T)
    # Positive value estimates with no rewards -> advantages should be <= 0
    values = torch.ones(T) * 2.0
    dones = torch.zeros(T)
    adv, ret = compute_gae_advantages(rewards, values, dones, cfg)
    # With zero rewards and positive values, delta_t = 0 + gamma*V(t+1) - V(t)
    # At last step: delta = 0 + 0 - 2.0 = -2.0, which is negative
    # Earlier steps: delta = gamma * 2.0 - 2.0 = (0.99 - 1) * 2 = -0.02, also negative
    assert adv[-1].item() < 0, "Expected negative advantage at last timestep with zero rewards and positive values"


# ---------------------------------------------------------------------------
# 4. compute_gae_advantages terminal state (done=1) cuts off future
# ---------------------------------------------------------------------------

def test_gae_terminal_done_cuts_future():
    cfg = PPOConfig(gamma=0.99, gae_lambda=1.0)
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    # Done at t=1 means future after t=1 is cut off
    dones = torch.tensor([0.0, 1.0, 0.0])

    adv, ret = compute_gae_advantages(rewards, values, dones, cfg)
    # ret[2] = 1.0 (last step, no future, done[2]=0 but no t+1)
    assert ret[2].item() == pytest.approx(1.0)
    # ret[1] = reward[1] + gamma * values[2] * (1 - done[1]) = 1 + 0 = 1.0 (done cuts future)
    assert ret[1].item() == pytest.approx(1.0)
    # ret[0] = reward[0] + gamma * values[1] * (1 - done[0]) = 1 + gamma * 0 = 1
    # but with lam=1.0, gae carries forward, so ret[0] = adv[0] + values[0]
    # adv[0] = delta[0] + gamma * lam * (1 - done[0]) * adv[1]
    #        = (1 + gamma * 0 - 0) + gamma * 1.0 * 1.0 * adv[1]
    #        = 1 + 0.99 * adv[1]
    # adv[1] = delta[1] + gamma * lam * (1-done[1]) * adv[2] = 1 + 0 = 1.0
    # adv[0] = 1 + 0.99 * 1.0 = 1.99
    assert ret[0].item() == pytest.approx(1.99, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. ppo_policy_loss returns scalar
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_scalar():
    T = 8
    log_probs = torch.randn(T)
    old_log_probs = torch.randn(T)
    advantages = torch.randn(T)
    loss = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 6. ppo_policy_loss clipping: large ratio gets clipped
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_clipping():
    T = 4
    # Force large positive ratio (log_probs >> old_log_probs) with positive advantages
    # so clipping is active
    log_probs = torch.ones(T) * 5.0
    old_log_probs = torch.zeros(T)
    advantages = torch.ones(T)  # positive advantages -> unclipped would use large ratio

    loss_clipped = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)

    # Compute unclipped manually: r = exp(5), loss_unclipped = -mean(r * A)
    r = (log_probs - old_log_probs).exp()
    loss_unclipped_manual = -(r * advantages).mean()

    # Clipped loss should differ (be less negative / closer to 0)
    assert not torch.isclose(loss_clipped, loss_unclipped_manual, atol=1e-3), (
        "Expected clipped loss to differ from unclipped loss for large ratio"
    )
    # Clipped loss magnitude should be smaller (less negative reward for policy)
    assert loss_clipped.item() > loss_unclipped_manual.item()


# ---------------------------------------------------------------------------
# 7. ppo_value_loss returns scalar
# ---------------------------------------------------------------------------

def test_ppo_value_loss_scalar():
    T = 8
    values = torch.randn(T)
    old_values = torch.randn(T)
    returns = torch.randn(T)
    loss = ppo_value_loss(values, old_values, returns, value_clip=0.2)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 8. compute_entropy_bonus returns scalar
# ---------------------------------------------------------------------------

def test_compute_entropy_bonus_scalar():
    log_probs = torch.randn(16)
    entropy = compute_entropy_bonus(log_probs)
    assert entropy.shape == torch.Size([]), f"Expected scalar, got shape {entropy.shape}"


# ---------------------------------------------------------------------------
# 9. Rollout dataclass stores all fields
# ---------------------------------------------------------------------------

def test_rollout_dataclass_fields():
    B, T = 2, 8
    rollout = Rollout(
        input_ids=torch.zeros(B, T, dtype=torch.long),
        log_probs=torch.zeros(B, T),
        values=torch.zeros(B, T),
        rewards=torch.zeros(B),
        advantages=torch.zeros(B, T),
        returns=torch.zeros(B, T),
    )
    assert rollout.input_ids.shape == (B, T)
    assert rollout.log_probs.shape == (B, T)
    assert rollout.values.shape == (B, T)
    assert rollout.rewards.shape == (B,)
    assert rollout.advantages.shape == (B, T)
    assert rollout.returns.shape == (B, T)


# ---------------------------------------------------------------------------
# 10. PPOTrainer collect_rollout returns Rollout with correct shapes
# ---------------------------------------------------------------------------

def test_collect_rollout_shapes():
    trainer = _make_trainer()
    B = 2
    prompt = _make_prompt(B=B)
    rollout = trainer.collect_rollout(prompt, max_new_tokens=MAX_NEW_TOKENS)
    T = MAX_NEW_TOKENS
    assert isinstance(rollout, Rollout)
    assert rollout.input_ids.shape == (B, T), f"input_ids shape: {rollout.input_ids.shape}"
    assert rollout.log_probs.shape == (B, T), f"log_probs shape: {rollout.log_probs.shape}"
    assert rollout.values.shape == (B, T), f"values shape: {rollout.values.shape}"
    assert rollout.rewards.shape == (B,), f"rewards shape: {rollout.rewards.shape}"
    assert rollout.advantages.shape == (B, T), f"advantages shape: {rollout.advantages.shape}"
    assert rollout.returns.shape == (B, T), f"returns shape: {rollout.returns.shape}"


# ---------------------------------------------------------------------------
# 11. PPOTrainer ppo_step returns correct keys
# ---------------------------------------------------------------------------

def test_ppo_step_returns_correct_keys():
    trainer = _make_trainer()
    B = 2
    prompt = _make_prompt(B=B)
    rollout = trainer.collect_rollout(prompt, max_new_tokens=MAX_NEW_TOKENS)
    metrics = trainer.ppo_step(rollout)
    assert "policy_loss" in metrics, f"Missing policy_loss in {metrics.keys()}"
    assert "value_loss" in metrics, f"Missing value_loss in {metrics.keys()}"
    assert "entropy" in metrics, f"Missing entropy in {metrics.keys()}"
    assert "kl" in metrics, f"Missing kl in {metrics.keys()}"


# ---------------------------------------------------------------------------
# 12. PPOTrainer train_step returns finite metrics
# ---------------------------------------------------------------------------

def test_train_step_finite_metrics():
    trainer = _make_trainer()
    prompt = _make_prompt(B=2)
    metrics = trainer.train_step(prompt)
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} is not float: {type(val)}"
        assert not (val != val), f"{key} is NaN"  # NaN check without math.isnan
        assert abs(val) < 1e9, f"{key}={val} is unexpectedly large (possibly Inf)"


# ---------------------------------------------------------------------------
# 13. ppo_policy_loss same old/new log_probs -> ratio=1, loss depends only on advantages
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_ratio_one():
    T = 8
    log_probs = torch.randn(T)
    # Same log_probs -> ratio = exp(0) = 1.0 everywhere
    # loss = -mean(1 * A) = -mean(A) since ratio is 1 and clamp(1, 0.8, 1.2) = 1
    advantages = torch.randn(T)
    loss = ppo_policy_loss(log_probs, log_probs, advantages, clip_epsilon=0.2)
    expected = -advantages.mean()
    assert torch.isclose(loss, expected, atol=1e-5), (
        f"Expected {expected.item():.6f}, got {loss.item():.6f}"
    )
