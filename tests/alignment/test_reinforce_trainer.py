"""Tests for src/alignment/reinforce_trainer.py."""

import pytest
import torch
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.reinforce_trainer import (
    ReinforceConfig,
    compute_reinforce_loss,
    compute_rloo_baseline,
    compute_kl_penalty,
    sample_rollout,
    ReinforceTrainer,
)

# ---------------------------------------------------------------------------
# Shared tiny model config
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

MAX_NEW_TOKENS = 4
PROMPT_LEN = 8


def _make_model():
    return AureliusTransformer(TINY_CFG)


def _make_prompt():
    return torch.randint(0, TINY_CFG.vocab_size, (1, PROMPT_LEN))


# ---------------------------------------------------------------------------
# 1. ReinforceConfig defaults
# ---------------------------------------------------------------------------

def test_reinforce_config_defaults():
    cfg = ReinforceConfig()
    assert cfg.n_samples == 4
    assert cfg.kl_coeff == 0.05
    assert cfg.gamma == 1.0
    assert cfg.normalize_rewards is True
    assert cfg.max_new_tokens == 32
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 2. compute_reinforce_loss — returns scalar
# ---------------------------------------------------------------------------

def test_compute_reinforce_loss_shape():
    B, T = 4, 8
    log_probs = torch.randn(B, T)
    rewards = torch.randn(B)
    loss = compute_reinforce_loss(log_probs, rewards)
    assert loss.shape == torch.Size([]), "Expected scalar loss"


# ---------------------------------------------------------------------------
# 3. compute_reinforce_loss — no baseline correctness
# ---------------------------------------------------------------------------

def test_compute_reinforce_loss_no_baseline():
    B, T = 3, 5
    log_probs = torch.ones(B, T) * -0.5  # sum per seq = -2.5
    rewards = torch.tensor([1.0, 2.0, 3.0])
    loss = compute_reinforce_loss(log_probs, rewards)
    # Expected: -mean(rewards * seq_log_probs) = -mean([1*-2.5, 2*-2.5, 3*-2.5])
    #         = -mean([-2.5, -5.0, -7.5]) = -(-5.0) = 5.0
    expected = -((rewards * log_probs.sum(dim=-1)).mean())
    assert torch.isclose(loss, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. compute_reinforce_loss — baseline subtracted from rewards
# ---------------------------------------------------------------------------

def test_compute_reinforce_loss_with_baseline():
    B, T = 4, 6
    log_probs = torch.ones(B, T) * -1.0
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    baseline = torch.tensor([0.5, 0.5, 0.5, 0.5])
    loss = compute_reinforce_loss(log_probs, rewards, baseline)
    advantages = rewards - baseline
    expected = -((advantages * log_probs.sum(dim=-1)).mean())
    assert torch.isclose(loss, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. compute_rloo_baseline — baseline[i] = mean of others
# ---------------------------------------------------------------------------

def test_compute_rloo_baseline_mean():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    baseline = compute_rloo_baseline(rewards)
    assert baseline.shape == rewards.shape
    # baseline[0] = (2+3+4)/3 = 3.0
    assert torch.isclose(baseline[0], torch.tensor(3.0), atol=1e-5)
    # baseline[1] = (1+3+4)/3 = 8/3
    assert torch.isclose(baseline[1], torch.tensor(8.0 / 3.0), atol=1e-5)
    # baseline[2] = (1+2+4)/3 = 7/3
    assert torch.isclose(baseline[2], torch.tensor(7.0 / 3.0), atol=1e-5)
    # baseline[3] = (1+2+3)/3 = 2.0
    assert torch.isclose(baseline[3], torch.tensor(2.0), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. compute_rloo_baseline — G=1 returns zeros
# ---------------------------------------------------------------------------

def test_compute_rloo_baseline_g1():
    rewards = torch.tensor([5.0])
    baseline = compute_rloo_baseline(rewards)
    assert baseline.shape == (1,)
    assert baseline[0].item() == 0.0


# ---------------------------------------------------------------------------
# 7. compute_kl_penalty — shape (B, T)
# ---------------------------------------------------------------------------

def test_compute_kl_penalty_shape():
    B, T = 3, 10
    log_probs_policy = torch.randn(B, T)
    log_probs_ref = torch.randn(B, T)
    kl = compute_kl_penalty(log_probs_policy, log_probs_ref)
    assert kl.shape == (B, T)


# ---------------------------------------------------------------------------
# 8. compute_kl_penalty — same log_probs gives zero KL
# ---------------------------------------------------------------------------

def test_compute_kl_penalty_zero_for_same():
    B, T = 4, 8
    log_probs = torch.randn(B, T)
    kl = compute_kl_penalty(log_probs, log_probs)
    assert torch.allclose(kl, torch.zeros(B, T), atol=1e-6)


# ---------------------------------------------------------------------------
# 9. sample_rollout — correct output shapes
# ---------------------------------------------------------------------------

def test_sample_rollout_shapes():
    model = _make_model()
    model.train(False)
    prompt = _make_prompt()
    with torch.no_grad():
        gen_ids, log_probs = sample_rollout(model, prompt, MAX_NEW_TOKENS)
    assert gen_ids.shape == (1, MAX_NEW_TOKENS), f"Expected (1, {MAX_NEW_TOKENS}), got {gen_ids.shape}"
    assert log_probs.shape == (1, MAX_NEW_TOKENS), f"Expected (1, {MAX_NEW_TOKENS}), got {log_probs.shape}"


# ---------------------------------------------------------------------------
# 10. ReinforceTrainer.train_step — returns correct keys
# ---------------------------------------------------------------------------

def test_reinforce_trainer_train_step_keys():
    policy = _make_model()
    ref = _make_model()
    for p in ref.parameters():
        p.requires_grad_(False)

    reward_fn = lambda tokens: 1.0

    cfg = ReinforceConfig(
        n_samples=2,
        max_new_tokens=MAX_NEW_TOKENS,
        normalize_rewards=False,
    )
    optimizer = optim.SGD(policy.parameters(), lr=1e-3)
    trainer = ReinforceTrainer(policy, ref, reward_fn, cfg, optimizer)

    prompt = _make_prompt()
    result = trainer.train_step(prompt)

    assert "loss" in result
    assert "mean_reward" in result
    assert "mean_kl" in result


# ---------------------------------------------------------------------------
# 11. ReinforceTrainer.train_step — loss is nonzero
# ---------------------------------------------------------------------------

def test_reinforce_trainer_train_step_loss_nonzero():
    policy = _make_model()
    ref = _make_model()
    for p in ref.parameters():
        p.requires_grad_(False)

    # Use a non-constant reward so advantages differ across samples
    import random
    call_count = [0]
    def varying_reward(tokens):
        call_count[0] += 1
        return float(call_count[0])

    cfg = ReinforceConfig(
        n_samples=4,
        max_new_tokens=MAX_NEW_TOKENS,
        normalize_rewards=True,
    )
    optimizer = optim.SGD(policy.parameters(), lr=1e-3)
    trainer = ReinforceTrainer(policy, ref, varying_reward, cfg, optimizer)

    prompt = _make_prompt()
    result = trainer.train_step(prompt)

    assert result["loss"] != 0.0, f"Expected nonzero loss, got {result['loss']}"
