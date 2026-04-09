"""Tests for RLVR: Reinforcement Learning with Verifiable Rewards."""
import math
import torch
import pytest

from src.alignment.rlvr import (
    RLVRConfig,
    VerifiableReward,
    MathReward,
    FormatReward,
    CompositeReward,
    sample_completions,
    compute_rlvr_loss,
    RLVRTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )


@pytest.fixture
def policy_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def ref_model(small_cfg):
    torch.manual_seed(1)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def prompt_ids():
    return torch.tensor([[10, 20, 30]], dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. RLVRConfig defaults
# ---------------------------------------------------------------------------

def test_rlvr_config_defaults():
    """RLVRConfig has the correct default values."""
    cfg = RLVRConfig()
    assert cfg.n_samples == 8
    assert cfg.max_new_tokens == 256
    assert cfg.temperature == pytest.approx(0.8)
    assert cfg.kl_coeff == pytest.approx(0.04)
    assert cfg.clip_ratio == pytest.approx(0.2)
    assert cfg.normalize_rewards is True


# ---------------------------------------------------------------------------
# 2-4. MathReward
# ---------------------------------------------------------------------------

def test_math_reward_correct():
    """MathReward returns 1.0 for exact numeric match."""
    reward = MathReward()
    score = reward("", "The answer is 42", "42")
    assert score == pytest.approx(1.0)


def test_math_reward_wrong():
    """MathReward returns 0.0 for clearly wrong answer."""
    reward = MathReward()
    score = reward("", "The answer is 99", "42")
    assert score == pytest.approx(0.0)


def test_math_reward_near():
    """MathReward returns 0.5 for answer within 10% of truth."""
    reward = MathReward()
    # 42 * 1.05 = 44.1, which is within 10% of 42
    score = reward("", "The answer is 44.1", "42")
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5-6. FormatReward
# ---------------------------------------------------------------------------

def test_format_reward_boxed_and_long():
    r"""FormatReward returns 1.0 when completion has \boxed{ and length > 50."""
    reward = FormatReward()
    completion = r"We reason carefully about the problem. \boxed{42} is the final answer."
    assert len(completion) > 50
    score = reward("", completion, "42")
    assert score == pytest.approx(1.0)


def test_format_reward_short_no_boxed():
    """FormatReward returns 0.0 for short completion with no boxed."""
    reward = FormatReward()
    score = reward("", "42", "42")
    assert len("42") <= 50
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. CompositeReward weighted sum
# ---------------------------------------------------------------------------

def test_composite_reward_weighted_sum():
    """CompositeReward returns correctly weighted sum normalized by total weight."""
    math_r = MathReward()
    format_r = FormatReward()
    # math gets weight 2, format gets weight 1 → total weight = 3
    composite = CompositeReward([(math_r, 2.0), (format_r, 1.0)])

    completion = "42"  # correct answer, short, no boxed
    score = composite("", completion, "42")
    # math=1.0, format=0.0 → (1.0*2 + 0.0*1) / 3 = 2/3
    assert score == pytest.approx(2.0 / 3.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 8. sample_completions output shapes
# ---------------------------------------------------------------------------

def test_sample_completions_shapes(policy_model, prompt_ids):
    """sample_completions returns tensors of the expected shapes."""
    n_samples = 2
    max_new_tokens = 4
    completions, log_probs = sample_completions(
        policy_model, prompt_ids, n_samples=n_samples,
        max_new_tokens=max_new_tokens, temperature=1.0,
    )
    assert completions.shape == (n_samples, max_new_tokens)
    assert log_probs.shape == (n_samples, max_new_tokens)


# ---------------------------------------------------------------------------
# 9-10. compute_rlvr_loss
# ---------------------------------------------------------------------------

def test_compute_rlvr_loss_scalar():
    """compute_rlvr_loss returns a finite scalar loss."""
    n, T = 4, 8
    log_probs = torch.randn(n, T, requires_grad=True)
    ref_log_probs = torch.randn(n, T)
    rewards = torch.tensor([0.0, 0.5, 1.0, 0.5])
    cfg = RLVRConfig()
    loss, _ = compute_rlvr_loss(log_probs, ref_log_probs, rewards, cfg)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert log_probs.grad is not None


def test_compute_rlvr_loss_dict_keys():
    """compute_rlvr_loss returns dict with required keys."""
    n, T = 2, 4
    log_probs = torch.randn(n, T, requires_grad=True)
    ref_log_probs = torch.randn(n, T)
    rewards = torch.tensor([0.0, 1.0])
    cfg = RLVRConfig()
    _, metrics = compute_rlvr_loss(log_probs, ref_log_probs, rewards, cfg)
    assert "policy_loss" in metrics
    assert "kl_loss" in metrics
    assert "mean_reward" in metrics


# ---------------------------------------------------------------------------
# 11-12. RLVRTrainer.train_step
# ---------------------------------------------------------------------------

def test_rlvr_trainer_train_step_keys(policy_model, ref_model, prompt_ids):
    """RLVRTrainer.train_step returns dict with required keys."""
    cfg = RLVRConfig(n_samples=2, max_new_tokens=4)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    reward_fn = MathReward()

    trainer = RLVRTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        config=cfg,
        optimizer=optimizer,
    )
    result = trainer.train_step(prompt_ids, "What is 2+2?", "4")
    assert "loss" in result
    assert "mean_reward" in result
    assert "n_samples" in result


def test_rlvr_trainer_n_samples_matches_config(policy_model, ref_model, prompt_ids):
    """RLVRTrainer.train_step n_samples in output matches config."""
    cfg = RLVRConfig(n_samples=2, max_new_tokens=4)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    reward_fn = FormatReward()

    trainer = RLVRTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        config=cfg,
        optimizer=optimizer,
    )
    result = trainer.train_step(prompt_ids, "Solve:", "0")
    assert result["n_samples"] == cfg.n_samples
    assert math.isfinite(result["loss"])
