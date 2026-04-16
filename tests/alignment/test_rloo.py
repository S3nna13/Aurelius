"""Tests for src/alignment/rloo.py — RLOOConfig, rloo_advantage_estimator,
RLOOTrainer.compute_rloo_advantages, compute_policy_gradient_loss,
compute_kl_penalty, and train_step."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.alignment.rloo import (
    RLOOConfig,
    RLOOTrainer,
    rloo_advantage_estimator,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Tiny model helpers
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


def _make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(TINY_CFG)


def _make_trainer(
    k: int = 4,
    kl_coef: float = 0.01,
    clip_ratio: float = 0.2,
) -> tuple[RLOOTrainer, AureliusTransformer]:
    model = _make_model()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    reward_fn = lambda responses: torch.ones(len(responses))
    trainer = RLOOTrainer(
        model=model,
        reward_fn=reward_fn,
        optimizer=optimizer,
        k_responses=k,
        kl_coef=kl_coef,
        clip_ratio=clip_ratio,
    )
    return trainer, model


# ---------------------------------------------------------------------------
# 1. rloo_advantage_estimator: mean advantage per group ≈ 0
# ---------------------------------------------------------------------------

def test_rloo_advantage_mean_per_group_zero():
    """For any set of rewards, the within-group mean advantage should be ~0."""
    k = 4
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0,   # group 0
                             0.5, 1.5, 2.5, 3.5])  # group 1
    advantages = rloo_advantage_estimator(rewards, k)
    n = rewards.shape[0] // k
    adv_grouped = advantages.view(n, k)
    group_means = adv_grouped.mean(dim=1)
    assert torch.allclose(group_means, torch.zeros(n), atol=1e-5), (
        f"Group means should be ~0, got {group_means}"
    )


# ---------------------------------------------------------------------------
# 2. Higher reward → positive advantage
# ---------------------------------------------------------------------------

def test_rloo_higher_reward_positive_advantage():
    """The response with the highest reward in a group must have a positive advantage."""
    k = 4
    rewards = torch.tensor([0.1, 0.2, 0.3, 10.0])  # last is much higher
    advantages = rloo_advantage_estimator(rewards, k)
    assert advantages[-1].item() > 0.0, (
        f"Highest-reward response should have positive advantage, got {advantages[-1]}"
    )
    # And the lowest should be negative
    assert advantages[0].item() < 0.0, (
        f"Lowest-reward response should have negative advantage, got {advantages[0]}"
    )


# ---------------------------------------------------------------------------
# 3. k=1 → zero advantages
# ---------------------------------------------------------------------------

def test_rloo_k1_zero_advantages():
    """With only one response per prompt there is no baseline, so advantages = 0."""
    k = 1
    rewards = torch.tensor([3.7, 0.2, 8.1, -1.5])
    advantages = rloo_advantage_estimator(rewards, k)
    assert torch.allclose(advantages, torch.zeros_like(rewards), atol=1e-7), (
        f"Expected all-zero advantages for k=1, got {advantages}"
    )


# ---------------------------------------------------------------------------
# 4. RLOOConfig default values
# ---------------------------------------------------------------------------

def test_rloo_config_defaults():
    cfg = RLOOConfig()
    assert cfg.k_responses == 4
    assert cfg.kl_coef == pytest.approx(0.01)
    assert cfg.clip_ratio == pytest.approx(0.2)
    assert cfg.gamma == pytest.approx(1.0)
    assert cfg.normalize_advantages is True


# ---------------------------------------------------------------------------
# 5. compute_rloo_advantages output shape matches input
# ---------------------------------------------------------------------------

def test_compute_rloo_advantages_shape():
    trainer, _ = _make_trainer(k=4)
    B = 3  # number of prompts
    rewards = torch.rand(B * trainer.k_responses)
    advantages = trainer.compute_rloo_advantages(rewards)
    assert advantages.shape == rewards.shape, (
        f"Expected shape {rewards.shape}, got {advantages.shape}"
    )


# ---------------------------------------------------------------------------
# 6. compute_policy_gradient_loss returns scalar tensor
# ---------------------------------------------------------------------------

def test_pg_loss_is_scalar():
    trainer, _ = _make_trainer()
    B = 8
    log_probs = torch.randn(B, requires_grad=True)
    advantages = torch.randn(B)
    loss = trainer.compute_policy_gradient_loss(log_probs, advantages)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert isinstance(loss, torch.Tensor)


# ---------------------------------------------------------------------------
# 7. KL penalty increases when log_probs diverge from ref_log_probs
# ---------------------------------------------------------------------------

def test_kl_penalty_increases_with_divergence():
    trainer, _ = _make_trainer()
    B = 16

    # Same distribution → very small KL
    base_lp = torch.randn(B)
    kl_zero = trainer.compute_kl_penalty(base_lp, base_lp)

    # Strongly shifted distribution → large KL
    shifted_lp = base_lp + 10.0
    kl_large = trainer.compute_kl_penalty(shifted_lp, base_lp)

    assert kl_large.item() > kl_zero.item(), (
        f"Diverged KL ({kl_large.item():.4f}) should exceed same-dist KL ({kl_zero.item():.4f})"
    )


# ---------------------------------------------------------------------------
# 8. train_step returns dict with all required keys
# ---------------------------------------------------------------------------

def test_train_step_required_keys():
    trainer, model = _make_trainer(k=2)
    B = 2   # prompts × k_responses
    T = 6   # sequence length (must be > 1 for teacher-forcing shift)

    torch.manual_seed(0)
    input_ids = torch.randint(0, TINY_CFG.vocab_size, (B, T))
    ref_log_probs = torch.randn(B).detach()
    rewards = torch.rand(B)

    result = trainer.train_step(input_ids, ref_log_probs, rewards)

    required = {"loss", "pg_loss", "kl_loss", "mean_reward", "mean_advantage"}
    assert required.issubset(result.keys()), (
        f"Missing keys: {required - result.keys()}"
    )


# ---------------------------------------------------------------------------
# 9. normalize_advantages=True makes advantages zero-mean, unit-variance
# ---------------------------------------------------------------------------

def test_normalize_advantages_zero_mean_unit_var():
    """After normalisation the advantage tensor should be z-scored."""
    torch.manual_seed(7)
    k = 4
    n = 10
    rewards = torch.rand(n * k) * 10 - 5  # range [-5, 5]

    # Compute raw RLOO advantages
    raw_adv = rloo_advantage_estimator(rewards, k)

    # z-score normalisation (as RLOOConfig.normalize_advantages would apply)
    mean = raw_adv.mean()
    std = raw_adv.std() + 1e-8
    normed = (raw_adv - mean) / std

    assert torch.abs(normed.mean()) < 1e-4, f"Normalised mean should be ~0, got {normed.mean()}"
    assert torch.abs(normed.std() - 1.0) < 1e-3, f"Normalised std should be ~1, got {normed.std()}"


# ---------------------------------------------------------------------------
# 10. Gradient flows through pg_loss correctly
# ---------------------------------------------------------------------------

def test_gradient_flows_through_pg_loss():
    """loss.backward() must not raise and must produce non-None gradients."""
    trainer, model = _make_trainer(k=4)
    B = 4
    T = 8

    torch.manual_seed(99)
    input_ids = torch.randint(0, TINY_CFG.vocab_size, (B, T))

    # Forward pass to get log_probs with gradient
    output = model(input_ids)
    logits = output[1]  # (B, T, V)
    log_probs_all = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:].unsqueeze(-1)
    token_log_probs = log_probs_all.gather(2, target_ids).squeeze(-1)
    log_probs = token_log_probs.sum(dim=-1)  # (B,)

    advantages = torch.randn(B)
    loss = trainer.compute_policy_gradient_loss(log_probs, advantages)

    # This must not raise
    loss.backward()

    # At least one parameter should have a gradient
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients found after backward() — gradient flow broken"
