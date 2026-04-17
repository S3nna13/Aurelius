"""Tests for aurelius.alignment.online_dpo (Online DPO, Guo et al. 2024).

Covers all 13+ required test cases:
  1.  OnlineDPOConfig defaults
  2.  CompletionSampler.log_probs_of shape (B,)
  3.  CompletionSampler.log_probs_of values are finite
  4.  CompletionSampler.log_probs_of: higher logit → higher log prob
  5.  OnlinePairBuilder.build_pairs chosen/rejected shape (B,)
  6.  OnlinePairBuilder.build_pairs chosen reward >= rejected reward
  7.  OnlineDPOLoss returns scalar loss and correct dict keys
  8.  OnlineDPOLoss accuracy == 1.0 when pi_chosen >> pi_rejected
  9.  OnlineDPOLoss gradients flow through loss
  10. OnlineDPOTrainer.freeze_ref freezes all ref params
  11. OnlineDPOTrainer.online_step returns correct keys
  12. OnlineDPOTrainer.online_step loss is finite
  13. OnlineDPOTrainer.online_step updates policy parameters
  14. CompletionSampler.sample output shape (B * n_samples, T)
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from aurelius.alignment.online_dpo_guo2024 import (
    OnlineDPOConfig,
    CompletionSampler,
    OnlinePairBuilder,
    OnlineDPOLoss,
    OnlineDPOTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_linear(in_features: int = 8, out_features: int = 16) -> nn.Module:
    """Minimal nn.Module used as a stand-in policy / ref model."""
    return nn.Linear(in_features, out_features)


def make_trainer(
    beta: float = 0.1,
    n_completions: int = 4,
) -> tuple[OnlineDPOTrainer, nn.Module, nn.Module]:
    """Return (trainer, policy, ref) with fresh tiny models."""
    policy = make_tiny_linear()
    ref = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    config = OnlineDPOConfig(beta=beta, n_completions=n_completions)
    loss_fn = OnlineDPOLoss(beta=beta)
    trainer = OnlineDPOTrainer(policy, ref, optimizer, config, loss_fn)
    return trainer, policy, ref


# ---------------------------------------------------------------------------
# Test 1: OnlineDPOConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = OnlineDPOConfig()
    assert cfg.beta == 0.1
    assert cfg.n_completions == 4
    assert cfg.temperature == 1.0
    assert cfg.top_k_pairs == 1


# ---------------------------------------------------------------------------
# Test 2: CompletionSampler.log_probs_of returns shape (B,)
# ---------------------------------------------------------------------------

def test_log_probs_of_shape():
    B, T, V = 3, 5, 32
    sampler = CompletionSampler(vocab_size=V, temperature=1.0)
    logits = torch.randn(B, T, V)
    token_ids = torch.randint(0, V, (B, T))
    out = sampler.log_probs_of(logits, token_ids)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: CompletionSampler.log_probs_of values are finite
# ---------------------------------------------------------------------------

def test_log_probs_of_finite():
    B, T, V = 4, 6, 64
    sampler = CompletionSampler(vocab_size=V)
    logits = torch.randn(B, T, V)
    token_ids = torch.randint(0, V, (B, T))
    out = sampler.log_probs_of(logits, token_ids)
    assert torch.isfinite(out).all(), f"log_probs_of contains non-finite values: {out}"


# ---------------------------------------------------------------------------
# Test 4: CompletionSampler.log_probs_of — higher logit → higher log prob
# ---------------------------------------------------------------------------

def test_log_probs_of_higher_logit_higher_prob():
    """Sequence where token 0 is overwhelmingly preferred should have higher log-prob."""
    B, T, V = 2, 4, 16
    sampler = CompletionSampler(vocab_size=V)

    # Logits: token 0 is strongly preferred
    logits = torch.zeros(B, T, V)
    logits[:, :, 0] = 100.0  # massively boost token 0

    # High: tokens are all 0 (the preferred token)
    tokens_high = torch.zeros(B, T, dtype=torch.long)
    # Low: tokens are all 1 (a non-preferred token)
    tokens_low = torch.ones(B, T, dtype=torch.long)

    lp_high = sampler.log_probs_of(logits, tokens_high)
    lp_low = sampler.log_probs_of(logits, tokens_low)

    assert (lp_high > lp_low).all(), (
        f"High-logit tokens should have higher log prob: {lp_high} vs {lp_low}"
    )


# ---------------------------------------------------------------------------
# Test 5: OnlinePairBuilder.build_pairs returns correct shapes
# ---------------------------------------------------------------------------

def test_build_pairs_shapes():
    B, K = 5, 4
    config = OnlineDPOConfig(n_completions=K)
    builder = OnlinePairBuilder(config)
    rewards = torch.randn(B * K)
    chosen, rejected = builder.build_pairs(rewards)
    assert chosen.shape == (B,), f"Expected ({B},), got {chosen.shape}"
    assert rejected.shape == (B,), f"Expected ({B},), got {rejected.shape}"


# ---------------------------------------------------------------------------
# Test 6: chosen index has higher (or equal) reward than rejected index
# ---------------------------------------------------------------------------

def test_build_pairs_chosen_ge_rejected():
    torch.manual_seed(42)
    B, K = 6, 4
    config = OnlineDPOConfig(n_completions=K)
    builder = OnlinePairBuilder(config)
    rewards = torch.randn(B * K)
    chosen, rejected = builder.build_pairs(rewards)

    chosen_rewards = rewards[chosen]    # (B,)
    rejected_rewards = rewards[rejected]  # (B,)

    assert (chosen_rewards >= rejected_rewards).all(), (
        "Chosen completion must have reward >= rejected for every prompt.\n"
        f"chosen:   {chosen_rewards}\nrejected: {rejected_rewards}"
    )


# ---------------------------------------------------------------------------
# Test 7: OnlineDPOLoss returns scalar loss and correct dict keys
# ---------------------------------------------------------------------------

def test_dpo_loss_output_types_and_keys():
    B = 8
    loss_fn = OnlineDPOLoss(beta=0.1)
    pi_c = torch.randn(B, requires_grad=True)
    pi_r = torch.randn(B, requires_grad=True)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    loss, metrics = loss_fn(pi_c, pi_r, ref_c, ref_r)

    assert isinstance(loss, torch.Tensor), "loss must be a Tensor"
    assert loss.ndim == 0, f"loss must be scalar, got ndim={loss.ndim}"
    assert isinstance(metrics, dict), "metrics must be a dict"

    required_keys = {"loss", "accuracy", "reward_chosen", "reward_rejected", "margin"}
    assert required_keys == set(metrics.keys()), (
        f"metrics keys mismatch: expected {required_keys}, got {set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 8: OnlineDPOLoss accuracy == 1.0 when pi_chosen >> pi_rejected
# ---------------------------------------------------------------------------

def test_dpo_loss_accuracy_perfect():
    B = 4
    loss_fn = OnlineDPOLoss(beta=1.0)

    # Make chosen log-probs >> rejected so reward_chosen >> reward_rejected
    pi_chosen = torch.full((B,), 100.0, requires_grad=True)
    pi_rejected = torch.full((B,), -100.0, requires_grad=True)
    ref_chosen = torch.zeros(B)
    ref_rejected = torch.zeros(B)

    _, metrics = loss_fn(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
    assert metrics["accuracy"] == 1.0, (
        f"Expected accuracy=1.0 with pi_chosen >> pi_rejected, got {metrics['accuracy']}"
    )


# ---------------------------------------------------------------------------
# Test 9: OnlineDPOLoss gradients flow through loss
# ---------------------------------------------------------------------------

def test_dpo_loss_gradients_flow():
    B = 4
    loss_fn = OnlineDPOLoss(beta=0.1)

    pi_c = torch.randn(B, requires_grad=True)
    pi_r = torch.randn(B, requires_grad=True)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    loss, _ = loss_fn(pi_c, pi_r, ref_c, ref_r)
    loss.backward()

    assert pi_c.grad is not None, "Gradient did not flow to pi_chosen"
    assert pi_r.grad is not None, "Gradient did not flow to pi_rejected"
    assert torch.isfinite(pi_c.grad).all(), "pi_chosen grad contains non-finite values"
    assert torch.isfinite(pi_r.grad).all(), "pi_rejected grad contains non-finite values"


# ---------------------------------------------------------------------------
# Test 10: OnlineDPOTrainer.freeze_ref freezes all ref params
# ---------------------------------------------------------------------------

def test_freeze_ref_freezes_all_params():
    trainer, _, ref = make_trainer()

    # Ensure at least one param is trainable before freeze
    for p in ref.parameters():
        p.requires_grad_(True)

    trainer.freeze_ref()

    for name, param in ref.named_parameters():
        assert not param.requires_grad, (
            f"Ref param '{name}' still has requires_grad=True after freeze_ref()"
        )


# ---------------------------------------------------------------------------
# Test 11: OnlineDPOTrainer.online_step returns correct keys
# ---------------------------------------------------------------------------

def test_online_step_returns_correct_keys():
    trainer, _, _ = make_trainer()
    B = 4

    pi_c = torch.randn(B, requires_grad=True)
    pi_r = torch.randn(B, requires_grad=True)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    # Recompute with grad context
    pi_c2 = torch.randn(B)
    pi_r2 = torch.randn(B)

    # Supply tensors that actually need grad through the policy model
    # Use a simple proxy: wrap in a linear pass so policy.parameters() get grad
    policy = trainer.policy_model
    x = torch.randn(B, policy.in_features)
    out = policy(x)  # (B, out_features) — just to ensure params are in graph
    # For log-probs, use mean of output as proxy scalar per batch item
    pi_chosen = out.mean(dim=1)   # (B,)
    pi_rejected = -out.mean(dim=1)  # (B,) — different from chosen

    metrics = trainer.online_step(pi_chosen, pi_rejected, ref_c, ref_r)

    assert isinstance(metrics, dict)
    required = {"loss", "accuracy", "reward_chosen", "reward_rejected", "margin"}
    assert required == set(metrics.keys()), (
        f"Expected keys {required}, got {set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 12: OnlineDPOTrainer.online_step loss is finite
# ---------------------------------------------------------------------------

def test_online_step_loss_finite():
    trainer, _, _ = make_trainer()
    B = 4
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    policy = trainer.policy_model
    x = torch.randn(B, policy.in_features)
    out = policy(x)
    pi_chosen = out.mean(dim=1)
    pi_rejected = -out.mean(dim=1)

    metrics = trainer.online_step(pi_chosen, pi_rejected, ref_c, ref_r)
    assert torch.isfinite(torch.tensor(metrics["loss"])), (
        f"Loss is not finite: {metrics['loss']}"
    )


# ---------------------------------------------------------------------------
# Test 13: OnlineDPOTrainer.online_step updates policy parameters
# ---------------------------------------------------------------------------

def test_online_step_updates_policy():
    trainer, policy, _ = make_trainer()
    B = 4
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    # Snapshot initial parameters
    params_before = {
        name: param.detach().clone()
        for name, param in policy.named_parameters()
    }

    x = torch.randn(B, policy.in_features)
    out = policy(x)
    pi_chosen = out.mean(dim=1)
    pi_rejected = -out.mean(dim=1)

    trainer.online_step(pi_chosen, pi_rejected, ref_c, ref_r)

    # At least one parameter must have changed
    changed = any(
        not torch.equal(params_before[name], param.detach())
        for name, param in policy.named_parameters()
    )
    assert changed, "No policy parameters were updated after online_step"


# ---------------------------------------------------------------------------
# Test 14: CompletionSampler.sample output shape (B * n_samples, T)
# ---------------------------------------------------------------------------

def test_sample_output_shape():
    B, T, V = 3, 7, 64
    n_samples = 4
    sampler = CompletionSampler(vocab_size=V, temperature=1.0)
    logits = torch.randn(B, T, V)
    tokens = sampler.sample(logits, n_samples)
    expected = (B * n_samples, T)
    assert tokens.shape == expected, f"Expected {expected}, got {tokens.shape}"
    assert tokens.dtype == torch.long, f"Expected LongTensor, got {tokens.dtype}"


# ---------------------------------------------------------------------------
# Bonus: build_pairs with explicit group_size override
# ---------------------------------------------------------------------------

def test_build_pairs_with_explicit_group_size():
    B, K = 3, 5  # different from config default of 4
    config = OnlineDPOConfig(n_completions=4)
    builder = OnlinePairBuilder(config)
    rewards = torch.arange(float(B * K))  # deterministic
    chosen, rejected = builder.build_pairs(rewards, group_size=K)
    assert chosen.shape == (B,)
    assert rejected.shape == (B,)
    assert (rewards[chosen] >= rewards[rejected]).all()
