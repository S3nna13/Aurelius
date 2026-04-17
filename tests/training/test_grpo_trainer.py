"""Tests for GRPO (Group Relative Policy Optimization) trainer.

15 tests covering GroupRewardNormalizer, GRPOLoss, GroupSampler, GRPOTrainer,
LengthReward, and UniqueTokenReward. Uses a tiny model to keep runtime short.
"""
from __future__ import annotations

import math
import importlib
import sys
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure the src package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.training.grpo_trainer import (
    GroupRewardNormalizer,
    GRPOLoss,
    GroupSampler,
    GRPOTrainer,
    LengthReward,
    UniqueTokenReward,
)


# ---------------------------------------------------------------------------
# Tiny model fixture
# Vocab=16, d_model=8 -- deliberate micro-size for speed.
# ---------------------------------------------------------------------------

VOCAB  = 16
D_MODEL = 8
SEQ_LEN = 4
BATCH   = 2
GROUP   = 2
MAX_NEW = 4


class TinyLM(nn.Module):
    """Tiny embedding + linear language model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.proj  = nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, T, V) logits."""
        x = self.embed(input_ids)
        return self.proj(x)


@pytest.fixture()
def tiny_lm() -> TinyLM:
    torch.manual_seed(42)
    return TinyLM()


@pytest.fixture()
def ref_lm() -> TinyLM:
    torch.manual_seed(7)
    return TinyLM()


@pytest.fixture()
def normalizer() -> GroupRewardNormalizer:
    return GroupRewardNormalizer()


@pytest.fixture()
def grpo_loss() -> GRPOLoss:
    return GRPOLoss(clip_eps=0.2, kl_coeff=0.01)


# ===========================================================================
# GroupRewardNormalizer tests
# ===========================================================================

def test_normalizer_mean_approx_zero(normalizer: GroupRewardNormalizer) -> None:
    """Normalized advantages should have mean ~0."""
    rewards = torch.tensor([0.1, 0.5, 0.8, 1.0])
    adv = normalizer.normalize(rewards)
    assert adv.shape == rewards.shape
    assert abs(adv.mean().item()) < 1e-5, f"mean={adv.mean().item()}"


def test_normalizer_std_approx_one(normalizer: GroupRewardNormalizer) -> None:
    """Normalized advantages should have std ~1."""
    rewards = torch.tensor([0.1, 0.5, 0.8, 1.0])
    adv = normalizer.normalize(rewards)
    # std of population-normalized values should be ~1
    std_val = adv.std(unbiased=False).item()
    assert abs(std_val - 1.0) < 1e-4, f"std={std_val}"


def test_normalizer_uniform_returns_zeros(normalizer: GroupRewardNormalizer) -> None:
    """Uniform rewards -> all-zero advantages (no gradient signal)."""
    rewards = torch.ones(4) * 0.5
    adv = normalizer.normalize(rewards)
    assert torch.all(adv == 0.0), f"expected zeros, got {adv}"


def test_normalizer_clip_within_range(normalizer: GroupRewardNormalizer) -> None:
    """clip_advantages should keep every value within [-clip_range, clip_range]."""
    adv = torch.tensor([-50.0, -5.0, 0.0, 5.0, 50.0])
    clipped = normalizer.clip_advantages(adv, clip_range=10.0)
    assert clipped.min().item() >= -10.0
    assert clipped.max().item() <=  10.0


# ===========================================================================
# GRPOLoss tests
# ===========================================================================

def test_grpoloss_output_types(grpo_loss: GRPOLoss) -> None:
    """Loss and kl_penalty are scalars; clip_fraction in [0, 1]."""
    B, G = 2, 3
    policy_logps  = torch.randn(B, G, requires_grad=True)
    ref_logps     = torch.randn(B, G)
    advantages    = torch.randn(B, G)

    loss, kl, cf = grpo_loss(policy_logps, ref_logps, advantages)

    assert loss.shape    == torch.Size([])
    assert kl.shape      == torch.Size([])
    assert cf.shape      == torch.Size([])
    assert 0.0 <= cf.item() <= 1.0


def test_grpoloss_clip_fraction_nonzero_when_ratio_far(grpo_loss: GRPOLoss) -> None:
    """When policy deviates strongly from ref, clip_fraction should be > 0."""
    B, G = 2, 4
    # Large positive difference -> large ratio -> clipping fires
    policy_logps = torch.full((B, G),  5.0, requires_grad=True)
    ref_logps    = torch.full((B, G), -5.0)
    advantages   = torch.ones(B, G)

    _, _, cf = grpo_loss(policy_logps, ref_logps, advantages)
    assert cf.item() > 0.0, f"clip_fraction should be > 0, got {cf.item()}"


def test_grpoloss_grad_flows_to_policy_logps(grpo_loss: GRPOLoss) -> None:
    """Backward pass must propagate gradient to policy_logps."""
    B, G = 2, 2
    policy_logps = torch.randn(B, G, requires_grad=True)
    ref_logps    = torch.randn(B, G)
    advantages   = torch.randn(B, G)

    loss, _, _ = grpo_loss(policy_logps, ref_logps, advantages)
    loss.backward()

    assert policy_logps.grad is not None
    assert not torch.all(policy_logps.grad == 0), "gradient should be non-zero"


# ===========================================================================
# GroupSampler tests
# ===========================================================================

def test_groupsampler_returns_g_responses(tiny_lm: TinyLM) -> None:
    """sample_group should return exactly GROUP responses."""
    sampler = GroupSampler(tiny_lm, group_size=GROUP, temperature=1.0)
    prompt  = torch.randint(0, VOCAB, (1, SEQ_LEN))
    responses, log_probs = sampler.sample_group(prompt, max_new_tokens=MAX_NEW)

    assert len(responses) == GROUP


def test_groupsampler_logprobs_shape(tiny_lm: TinyLM) -> None:
    """log_probs should have shape (G,)."""
    sampler = GroupSampler(tiny_lm, group_size=GROUP, temperature=1.0)
    prompt  = torch.randint(0, VOCAB, (1, SEQ_LEN))
    _, log_probs = sampler.sample_group(prompt, max_new_tokens=MAX_NEW)

    assert log_probs.shape == (GROUP,), f"expected ({GROUP},), got {log_probs.shape}"


def test_groupsampler_temperature_effect(tiny_lm: TinyLM) -> None:
    """Low temperature should concentrate probability; high temperature spreads it.

    We measure entropy of sampled distributions over many runs.
    """
    torch.manual_seed(0)

    def collect_token_counts(temp: float, n_trials: int = 20) -> torch.Tensor:
        sampler = GroupSampler(tiny_lm, group_size=4, temperature=temp)
        prompt  = torch.randint(0, VOCAB, (1, SEQ_LEN))
        counts  = torch.zeros(VOCAB)
        for _ in range(n_trials):
            responses, _ = sampler.sample_group(prompt, max_new_tokens=MAX_NEW)
            for r in responses:
                for tok in r.tolist():
                    counts[tok] += 1
        return counts

    low_counts  = collect_token_counts(0.1)
    high_counts = collect_token_counts(2.0)

    def entropy(counts: torch.Tensor) -> float:
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    assert entropy(high_counts) >= entropy(low_counts), (
        "Higher temperature should yield >= entropy than lower temperature"
    )


# ===========================================================================
# GRPOTrainer tests
# ===========================================================================

@pytest.fixture()
def trainer_fixture(tiny_lm: TinyLM, ref_lm: TinyLM):
    optimizer = torch.optim.Adam(tiny_lm.parameters(), lr=1e-3)
    trainer   = GRPOTrainer(
        policy_model   = tiny_lm,
        ref_model      = ref_lm,
        optimizer      = optimizer,
        group_size     = GROUP,
        max_new_tokens = MAX_NEW,
    )
    prompt = torch.randint(0, VOCAB, (1, SEQ_LEN))
    reward_fn = LengthReward(target_length=MAX_NEW)
    return trainer, prompt, reward_fn


def test_trainer_returns_expected_keys(trainer_fixture) -> None:
    """train_step dict must contain all five expected keys."""
    trainer, prompt, reward_fn = trainer_fixture
    result = trainer.train_step(prompt, reward_fn)

    expected = {"loss", "kl_penalty", "clip_fraction", "mean_reward", "reward_std"}
    assert set(result.keys()) == expected


def test_trainer_mean_reward_finite(trainer_fixture) -> None:
    """mean_reward must be a finite float."""
    trainer, prompt, reward_fn = trainer_fixture
    result = trainer.train_step(prompt, reward_fn)
    assert math.isfinite(result["mean_reward"])


def test_trainer_reward_std_nonnegative(trainer_fixture) -> None:
    """reward_std must be >= 0."""
    trainer, prompt, reward_fn = trainer_fixture
    result = trainer.train_step(prompt, reward_fn)
    assert result["reward_std"] >= 0.0


def test_trainer_ref_model_params_frozen(trainer_fixture) -> None:
    """All parameters of ref_model must have requires_grad=False."""
    trainer, _, _ = trainer_fixture
    for name, param in trainer.ref_model.named_parameters():
        assert not param.requires_grad, (
            f"ref_model param {name} should be frozen"
        )


# ===========================================================================
# LengthReward tests
# ===========================================================================

def test_length_reward_exact_target() -> None:
    """reward = 1.0 when response length == target_length."""
    rw = LengthReward(target_length=5)
    ids = torch.arange(5)
    assert rw(ids) == 1.0


def test_length_reward_double_target_is_zero() -> None:
    """reward = 0.0 when length is 2x target (|deviation| == target)."""
    rw = LengthReward(target_length=5)
    ids = torch.arange(10)   # length = 10 = 2 * target
    assert rw(ids) == 0.0


# ===========================================================================
# UniqueTokenReward tests
# ===========================================================================

def test_unique_token_reward_all_unique() -> None:
    """reward = 1.0 when every token is distinct."""
    rw  = UniqueTokenReward()
    ids = torch.arange(8)  # all unique
    assert rw(ids) == 1.0


def test_unique_token_reward_in_range() -> None:
    """reward in [0, 1] for a typical (partially-repeated) sequence."""
    rw  = UniqueTokenReward()
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2])  # some repetitions
    val = rw(ids)
    assert 0.0 <= val <= 1.0


# ===========================================================================
# Full training loop test
# ===========================================================================

def test_full_training_loop_finite_loss(tiny_lm: TinyLM, ref_lm: TinyLM) -> None:
    """Three-step training loop must complete without error; loss stays finite."""
    torch.manual_seed(99)
    optimizer = torch.optim.Adam(tiny_lm.parameters(), lr=1e-3)
    trainer   = GRPOTrainer(
        policy_model   = tiny_lm,
        ref_model      = ref_lm,
        optimizer      = optimizer,
        group_size     = GROUP,
        max_new_tokens = MAX_NEW,
    )
    reward_fn = UniqueTokenReward()
    prompt    = torch.randint(0, VOCAB, (1, SEQ_LEN))

    for step in range(3):
        result = trainer.train_step(prompt, reward_fn)
        assert math.isfinite(result["loss"]), (
            f"Step {step}: loss is not finite: {result['loss']}"
        )
