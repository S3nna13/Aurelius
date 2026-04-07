"""Tests for GRPO implementation."""
import math
import torch
import pytest
from src.alignment.grpo import (
    compute_advantages,
    grpo_loss,
    compute_sequence_log_probs,
    GRPOConfig,
    GRPOTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def test_compute_advantages_normalized():
    """Advantages must have mean ~0 and std ~1."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = compute_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std().item() - 1.0) < 0.1


def test_compute_advantages_single_rollout():
    """Single rollout should return 0 advantage (no contrast possible)."""
    rewards = torch.tensor([5.0])
    adv = compute_advantages(rewards)
    assert adv.shape == (1,)
    # With std=0, advantage is 0
    assert adv[0].item() == pytest.approx(0.0, abs=1e-5)


def test_grpo_loss_scalar():
    """grpo_loss must return a finite scalar."""
    log_probs_new = torch.tensor([-2.0, -3.0, -1.5, -2.5])
    log_probs_old = log_probs_new.clone().detach()
    advantages = torch.tensor([1.0, -0.5, 0.8, -1.3])
    loss = grpo_loss(log_probs_new, log_probs_old, advantages)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_grpo_loss_on_policy_equals_negative_advantage_mean():
    """When ratio=1 (old==new log probs), loss = -mean(advantages) if not clipped."""
    log_probs = torch.tensor([-1.0, -2.0, -3.0])
    advantages = torch.tensor([1.0, -1.0, 2.0])
    loss = grpo_loss(log_probs, log_probs.detach(), advantages, clip_eps=1.0)
    # ratio=1, no clipping: loss = -mean(advantages)
    expected = -advantages.mean().item()
    assert abs(loss.item() - expected) < 1e-5


def test_compute_sequence_log_probs_shape(small_model):
    """compute_sequence_log_probs must return a scalar."""
    ids = torch.randint(0, 256, (1, 16))
    lp = compute_sequence_log_probs(small_model, ids, response_start=8)
    assert lp.ndim == 0
    assert torch.isfinite(lp)
    assert lp < 0


def test_grpo_trainer_step_updates_weights(small_model):
    """GRPOTrainer.step must update model weights."""
    # Reward fn: prefer shorter responses
    def reward_fn(prompt, response):
        return -len(response) / 100.0

    # Minimal tokenizer mock
    class FakeTok:
        def decode(self, ids): return "x" * len(ids)

    cfg = GRPOConfig(num_rollouts=4, num_steps=1, max_new_tokens=4, batch_size=1)
    trainer = GRPOTrainer(small_model, reward_fn, cfg)

    before = {n: p.clone() for n, p in small_model.named_parameters()}
    prompt_ids = torch.randint(0, 256, (1, 8))

    metrics = trainer.step(prompt_ids, "test prompt", FakeTok())

    assert "loss" in metrics
    assert math.isfinite(metrics["loss"])

    changed = any(
        not torch.equal(before[n], p)
        for n, p in small_model.named_parameters()
        if p.requires_grad
    )
    assert changed, "No weights updated after GRPO step"
