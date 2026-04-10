"""Tests for src/alignment/ppo_trainer.py — PPO RLHF trainer."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.ppo_trainer import (
    PPOConfig,
    ValueHead,
    PPOTrainer,
    compute_gae,
    ppo_policy_loss,
    ppo_value_loss,
    entropy_bonus,
)

# ---------------------------------------------------------------------------
# Shared tiny model config (fast for tests)
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


def _make_model() -> AureliusTransformer:
    return AureliusTransformer(TINY_CFG)


def _make_prompt(B: int = 1) -> torch.Tensor:
    return torch.randint(0, TINY_CFG.vocab_size, (B, PROMPT_LEN))


def _make_trainer(B: int = 1, n_epochs: int = 1) -> PPOTrainer:
    policy = _make_model()
    ref_model = _make_model()
    reward_fn = lambda tokens: 1.0  # constant reward
    cfg = PPOConfig(n_epochs=n_epochs, n_rollout_steps=4)
    optimizer = optim.SGD(policy.parameters(), lr=1e-3)
    return PPOTrainer(
        policy=policy,
        ref_model=ref_model,
        reward_fn=reward_fn,
        config=cfg,
        optimizer=optimizer,
    )


# ---------------------------------------------------------------------------
# 1. PPOConfig defaults
# ---------------------------------------------------------------------------

def test_ppoconfig_defaults():
    cfg = PPOConfig()
    assert cfg.clip_ratio == 0.2
    assert cfg.vf_coeff == 0.5
    assert cfg.entropy_coeff == 0.01
    assert cfg.n_epochs == 4
    assert cfg.n_rollout_steps == 8
    assert cfg.gamma == 1.0
    assert cfg.gae_lambda == 0.95
    assert cfg.temperature == 1.0
    assert cfg.max_grad_norm == 1.0


# ---------------------------------------------------------------------------
# 2. ValueHead output shape (B, T)
# ---------------------------------------------------------------------------

def test_value_head_output_shape():
    B, T, d_model = 3, 7, 64
    vh = ValueHead(d_model)
    hidden = torch.randn(B, T, d_model)
    out = vh(hidden)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. ValueHead is differentiable
# ---------------------------------------------------------------------------

def test_value_head_differentiable():
    B, T, d_model = 2, 5, 64
    vh = ValueHead(d_model)
    hidden = torch.randn(B, T, d_model, requires_grad=True)
    out = vh(hidden)
    loss = out.sum()
    loss.backward()
    assert hidden.grad is not None, "Expected gradient to flow through ValueHead"
    assert vh.linear.weight.grad is not None, "Expected weight gradient in ValueHead"


# ---------------------------------------------------------------------------
# 4. compute_gae output shapes are (T,) for both outputs
# ---------------------------------------------------------------------------

def test_compute_gae_output_shapes():
    T = 10
    rewards = torch.ones(T)
    values = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, gamma=1.0, gae_lambda=0.95)
    assert adv.shape == (T,), f"advantages shape should be ({T},), got {adv.shape}"
    assert ret.shape == (T,), f"returns shape should be ({T},), got {ret.shape}"


# ---------------------------------------------------------------------------
# 5. compute_gae advantages and returns shapes match rewards
# ---------------------------------------------------------------------------

def test_compute_gae_shapes_match_rewards():
    T = 15
    rewards = torch.randn(T)
    values = torch.randn(T)
    adv, ret = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape


# ---------------------------------------------------------------------------
# 6. compute_gae with all-zero rewards and values: advantages ~0
# ---------------------------------------------------------------------------

def test_compute_gae_zero_rewards_zero_values():
    T = 8
    rewards = torch.zeros(T)
    values = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, gamma=1.0, gae_lambda=0.95)
    assert torch.allclose(adv, torch.zeros(T), atol=1e-6), (
        f"Expected advantages ~0 for zero rewards/values, got {adv}"
    )
    assert torch.allclose(ret, torch.zeros(T), atol=1e-6), (
        f"Expected returns ~0 for zero rewards/values, got {ret}"
    )


# ---------------------------------------------------------------------------
# 7. ppo_policy_loss returns (Tensor, dict)
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_return_types():
    T = 8
    log_probs = torch.randn(2, T)
    old_log_probs = torch.randn(2, T)
    advantages = torch.randn(2, T)
    result = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_ratio=0.2)
    assert isinstance(result, tuple), "Expected tuple return"
    assert len(result) == 2, "Expected (Tensor, dict)"
    loss, metrics = result
    assert isinstance(loss, torch.Tensor), "First element should be Tensor"
    assert isinstance(metrics, dict), "Second element should be dict"


# ---------------------------------------------------------------------------
# 8. ppo_policy_loss dict has correct keys
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_dict_keys():
    log_probs = torch.randn(2, 8)
    old_log_probs = torch.randn(2, 8)
    advantages = torch.randn(2, 8)
    _, metrics = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_ratio=0.2)
    assert "policy_loss" in metrics, f"Missing 'policy_loss' in {metrics.keys()}"
    assert "clip_fraction" in metrics, f"Missing 'clip_fraction' in {metrics.keys()}"
    assert "mean_ratio" in metrics, f"Missing 'mean_ratio' in {metrics.keys()}"


# ---------------------------------------------------------------------------
# 9. ppo_policy_loss clip_fraction in [0, 1]
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_clip_fraction_range():
    log_probs = torch.randn(4, 16)
    old_log_probs = torch.randn(4, 16)
    advantages = torch.randn(4, 16)
    _, metrics = ppo_policy_loss(log_probs, old_log_probs, advantages, clip_ratio=0.2)
    cf = metrics["clip_fraction"]
    assert 0.0 <= cf <= 1.0, f"clip_fraction={cf} out of [0, 1]"


# ---------------------------------------------------------------------------
# 10. ppo_policy_loss with ratio=1 (same policy): no clipping
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_ratio_one_no_clipping():
    T = 8
    log_probs = torch.randn(2, T)
    # Same log_probs -> ratio = 1 everywhere -> no clipping
    advantages = torch.randn(2, T)
    loss, metrics = ppo_policy_loss(log_probs, log_probs, advantages, clip_ratio=0.2)
    assert metrics["clip_fraction"] == pytest.approx(0.0), (
        f"Expected no clipping when ratio=1, got clip_fraction={metrics['clip_fraction']}"
    )
    # loss should equal -mean(advantages)
    expected = -advantages.mean()
    assert torch.isclose(loss, expected, atol=1e-5), (
        f"Expected loss={expected.item():.6f}, got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 11. ppo_value_loss returns scalar >= 0
# ---------------------------------------------------------------------------

def test_ppo_value_loss_scalar_nonneg():
    values = torch.randn(4, 8)
    returns = torch.randn(4, 8)
    loss = ppo_value_loss(values, returns)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, f"MSE loss should be >= 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 12. entropy_bonus returns scalar >= 0
# ---------------------------------------------------------------------------

def test_entropy_bonus_scalar_nonneg():
    # For sampled log probs that are negative (log probs are <= 0),
    # entropy_bonus = -mean(log_probs) should be >= 0
    # Use log_softmax output (always <= 0) to get valid log probs
    logits = torch.randn(4, 16)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    sampled_lp = log_probs[:, :8].reshape(-1)  # sample some

    ent = entropy_bonus(sampled_lp)
    assert ent.shape == torch.Size([]), f"Expected scalar, got shape {ent.shape}"
    assert ent.item() >= 0.0, f"entropy_bonus should be >= 0 for valid log probs, got {ent.item()}"


# ---------------------------------------------------------------------------
# 13. PPOTrainer.collect_rollout returns dict with correct keys
# ---------------------------------------------------------------------------

def test_collect_rollout_keys():
    trainer = _make_trainer()
    prompt = _make_prompt(B=2)
    rollout = trainer.collect_rollout(prompt)
    required_keys = {"tokens", "log_probs", "values", "rewards"}
    assert required_keys <= set(rollout.keys()), (
        f"Missing keys: {required_keys - set(rollout.keys())}"
    )


# ---------------------------------------------------------------------------
# 13b. collect_rollout tensor shapes
# ---------------------------------------------------------------------------

def test_collect_rollout_shapes():
    trainer = _make_trainer()
    B = 2
    T = trainer.config.n_rollout_steps
    prompt = _make_prompt(B=B)
    rollout = trainer.collect_rollout(prompt)
    assert rollout["tokens"].shape == (B, T), f"tokens shape: {rollout['tokens'].shape}"
    assert rollout["log_probs"].shape == (B, T), f"log_probs shape: {rollout['log_probs'].shape}"
    assert rollout["values"].shape == (B, T), f"values shape: {rollout['values'].shape}"
    assert rollout["rewards"].shape == (B,), f"rewards shape: {rollout['rewards'].shape}"


# ---------------------------------------------------------------------------
# 14. PPOTrainer.train_step returns dict with correct keys
# ---------------------------------------------------------------------------

def test_train_step_returns_correct_keys():
    trainer = _make_trainer()
    prompt = _make_prompt(B=1)
    metrics = trainer.train_step(prompt)
    required_keys = {"policy_loss", "value_loss", "entropy", "total_loss"}
    assert required_keys <= set(metrics.keys()), (
        f"Missing keys: {required_keys - set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# 15. PPOTrainer.train_step losses are finite
# ---------------------------------------------------------------------------

def test_train_step_losses_finite():
    trainer = _make_trainer()
    prompt = _make_prompt(B=2)
    metrics = trainer.train_step(prompt)
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"
        assert math.isfinite(val), f"{key}={val} is not finite (NaN or Inf)"


# ---------------------------------------------------------------------------
# Bonus: ref_model is frozen
# ---------------------------------------------------------------------------

def test_ref_model_frozen():
    trainer = _make_trainer()
    for p in trainer.ref_model.parameters():
        assert not p.requires_grad, "ref_model parameters should be frozen"


# ---------------------------------------------------------------------------
# Bonus: ValueHead params added to optimizer
# ---------------------------------------------------------------------------

def test_value_head_params_in_optimizer():
    trainer = _make_trainer()
    # Collect all params registered in optimizer
    all_opt_params = set()
    for pg in trainer.optimizer.param_groups:
        for p in pg["params"]:
            all_opt_params.add(id(p))
    # At least one value head param should be in the optimizer
    vh_params = list(trainer.value_head.parameters())
    assert any(id(p) in all_opt_params for p in vh_params), (
        "ValueHead parameters should be added to the optimizer"
    )
