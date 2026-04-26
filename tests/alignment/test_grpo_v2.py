"""Tests for GRPOv2: Enhanced GRPO with Dr. GRPO corrections and clip-higher."""

import math

import pytest
import torch

from src.alignment.grpo_v2 import (
    GRPOv2Config,
    GRPOv2Trainer,
    clip_higher_ratio,
    compute_grpo_advantages,
    grpo_loss,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def default_config():
    return GRPOv2Config(group_size=4, use_dr_grpo=True, use_reference_free=False)


# ---------------------------------------------------------------------------
# compute_grpo_advantages tests
# ---------------------------------------------------------------------------


def test_compute_advantages_basic():
    """G=4, non-uniform rewards → advantages have mean ≈ 0."""
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3])
    config = GRPOv2Config(group_size=4, use_dr_grpo=True)
    adv = compute_grpo_advantages(rewards, config)
    assert adv.shape == (4,)
    assert abs(adv.mean().item()) < 1e-5, f"mean not ≈ 0: {adv.mean().item()}"


def test_compute_advantages_g1_returns_zero():
    """G=1 → advantage must be 0 (no contrast possible)."""
    rewards = torch.tensor([3.7])
    config = GRPOv2Config(group_size=1, use_dr_grpo=True)
    adv = compute_grpo_advantages(rewards, config)
    assert adv.shape == (1,)
    assert adv[0].item() == pytest.approx(0.0, abs=1e-7)


def test_compute_advantages_all_equal_rewards():
    """All same reward → all advantages = 0 (std = 0 case)."""
    rewards = torch.tensor([2.0, 2.0, 2.0, 2.0])
    config = GRPOv2Config(group_size=4, use_dr_grpo=True)
    adv = compute_grpo_advantages(rewards, config)
    assert torch.allclose(adv, torch.zeros(4), atol=1e-6)


# ---------------------------------------------------------------------------
# grpo_loss tests
# ---------------------------------------------------------------------------


def test_grpo_loss_returns_tensor(default_config):
    """grpo_loss returns a differentiable scalar tensor."""
    log_probs = torch.tensor([-1.0, -2.0, -1.5, -0.5], requires_grad=True)
    ref_log_probs = torch.tensor([-1.2, -2.1, -1.4, -0.6])
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3])

    loss, _ = grpo_loss(log_probs, ref_log_probs, rewards, default_config)

    assert loss.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(loss), "loss must be finite"
    # Must be differentiable
    loss.backward()
    assert log_probs.grad is not None


def test_grpo_loss_metrics_keys(default_config):
    """grpo_loss returns dict with all required metric keys."""
    log_probs = torch.tensor([-1.0, -2.0, -1.5, -0.5])
    ref_log_probs = torch.tensor([-1.2, -2.1, -1.4, -0.6])
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3])

    _, metrics = grpo_loss(log_probs, ref_log_probs, rewards, default_config)

    required_keys = {"policy_loss", "kl_loss", "mean_reward", "mean_advantage", "reward_std"}
    assert required_keys.issubset(metrics.keys()), f"Missing keys: {required_keys - metrics.keys()}"


def test_grpo_loss_reference_free():
    """reference_free=True with ref_log_probs=None must work without error."""
    config = GRPOv2Config(group_size=4, use_reference_free=True)
    log_probs = torch.tensor([-1.0, -2.0, -1.5, -0.5])
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3])

    loss, metrics = grpo_loss(log_probs, None, rewards, config)

    assert torch.isfinite(loss)
    assert metrics["kl_loss"] == pytest.approx(0.0, abs=1e-7)


def test_grpo_loss_kl_positive():
    """When policy log probs >> ref log probs, kl_loss should be positive."""
    config = GRPOv2Config(group_size=4, beta=1.0, use_reference_free=False)
    # policy much higher than ref → positive KL
    log_probs = torch.tensor([-0.1, -0.2, -0.1, -0.2])
    ref_log_probs = torch.tensor([-5.0, -5.0, -5.0, -5.0])
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3])

    _, metrics = grpo_loss(log_probs, ref_log_probs, rewards, config)

    assert metrics["kl_loss"] > 0, f"Expected positive kl_loss, got {metrics['kl_loss']}"


# ---------------------------------------------------------------------------
# clip_higher_ratio tests
# ---------------------------------------------------------------------------


def test_clip_higher_ratio_shape():
    """clip_higher_ratio output has the same shape as inputs."""
    ratio = torch.tensor([0.8, 1.0, 1.2, 1.5])
    advantage = torch.tensor([1.0, -0.5, 0.8, -1.3])
    out = clip_higher_ratio(ratio, advantage, clip_low=0.8, clip_high=1.4)
    assert out.shape == ratio.shape


# ---------------------------------------------------------------------------
# GRPOv2Trainer tests
# ---------------------------------------------------------------------------


def test_grpo_trainer_train_step_keys(small_model):
    """train_step returns dict with 'loss', 'mean_reward', 'reward_std'."""
    torch.manual_seed(42)

    config = GRPOv2Config(
        group_size=2,
        use_reference_free=True,
        use_dr_grpo=True,
    )

    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-5)

    def reward_fn(completion: str) -> float:
        return float(len(completion)) / 100.0

    def encode(text: str) -> list[int]:
        return [ord(c) % 256 for c in text[:4]] or [0]

    def decode(ids: list[int]) -> str:
        return "".join(chr(max(32, i)) for i in ids)

    trainer = GRPOv2Trainer(
        model=small_model,
        ref_model=None,
        optimizer=optimizer,
        reward_fn=reward_fn,
        tokenizer_encode=encode,
        tokenizer_decode=decode,
        config=config,
        max_seq_len=16,
    )

    metrics = trainer.train_step(["hello", "world"])

    assert "loss" in metrics
    assert "mean_reward" in metrics
    assert "reward_std" in metrics
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["mean_reward"])


# ---------------------------------------------------------------------------
# Dr. GRPO unbiased std test
# ---------------------------------------------------------------------------


def test_dr_grpo_unbiased_std():
    """Dr. GRPO uses ddof=1 (Bessel correction). Verify with G=2."""
    # G=2: biased std (ddof=0) != unbiased std (ddof=1)
    rewards = torch.tensor([1.0, 3.0])  # mean=2.0, diff=1.0
    # ddof=0: std = sqrt(((1-2)^2 + (3-2)^2) / 2) = sqrt(1.0) = 1.0
    # ddof=1: std = sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2.0) ≈ 1.4142

    config_dr = GRPOv2Config(group_size=2, use_dr_grpo=True, min_group_for_std=2)
    config_std = GRPOv2Config(group_size=2, use_dr_grpo=False, min_group_for_std=2)

    adv_dr = compute_grpo_advantages(rewards, config_dr)
    adv_std = compute_grpo_advantages(rewards, config_std)

    # Dr. GRPO (ddof=1): A = (r - 2) / (sqrt(2) + eps)
    expected_dr_0 = (1.0 - 2.0) / (math.sqrt(2.0) + 1e-8)
    expected_dr_1 = (3.0 - 2.0) / (math.sqrt(2.0) + 1e-8)

    assert adv_dr[0].item() == pytest.approx(expected_dr_0, rel=1e-5)
    assert adv_dr[1].item() == pytest.approx(expected_dr_1, rel=1e-5)

    # Standard (ddof=0): A = (r - 2) / (1.0 + eps)
    expected_std_0 = (1.0 - 2.0) / (1.0 + 1e-8)
    expected_std_1 = (3.0 - 2.0) / (1.0 + 1e-8)

    assert adv_std[0].item() == pytest.approx(expected_std_0, rel=1e-5)
    assert adv_std[1].item() == pytest.approx(expected_std_1, rel=1e-5)

    # The two must differ (key property of Dr. GRPO)
    assert not torch.allclose(adv_dr, adv_std, atol=1e-5), (
        "Dr. GRPO and standard GRPO should produce different advantages for G=2"
    )
