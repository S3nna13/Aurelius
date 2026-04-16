"""Tests for src/alignment/rlhf_utils.py.

All tests use tiny tensors (B=2, T=6, VOCAB=16) and pure PyTorch — no
HuggingFace or other external ML libraries.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.rlhf_utils import (
    RLHFConfig,
    RLHFTrainer,
    clip_rewards,
    compute_advantages_gae,
    compute_kl_penalty,
    compute_returns,
    entropy_bonus,
    ppo_policy_loss,
    ppo_value_loss,
    whiten_rewards,
)

# ---------------------------------------------------------------------------
# Common fixtures / helpers
# ---------------------------------------------------------------------------

B, T, VOCAB, D_IN = 2, 6, 16, 8

torch.manual_seed(0)


def make_log_probs(b=B, t=T) -> torch.Tensor:
    """Random log-probabilities in (-5, 0)."""
    return -torch.rand(b, t) * 5


def make_rewards(b=B, t=T) -> torch.Tensor:
    return torch.randn(b, t)


def make_values(b=B, t=T) -> torch.Tensor:
    return torch.randn(b, t)


# ---------------------------------------------------------------------------
# Minimal wrappers so nn.Linear can serve as a language-model / critic
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Wraps nn.Linear(D_IN, VOCAB) to accept integer token ids.

    forward(input_ids: LongTensor (B, T)) -> logits (B, T, VOCAB)
    """

    def __init__(self, vocab: int = VOCAB, d_in: int = D_IN) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_in)
        self.proj = nn.Linear(d_in, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))  # (B, T, VOCAB)


class TinyCritic(nn.Module):
    """Wraps nn.Linear(D_IN, 1) to accept integer token ids.

    forward(input_ids: LongTensor (B, T)) -> values (B, T, 1)
    """

    def __init__(self, d_in: int = D_IN) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_in)
        self.proj = nn.Linear(d_in, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))  # (B, T, 1)


@pytest.fixture()
def tiny_trainer() -> RLHFTrainer:
    torch.manual_seed(42)
    policy = TinyLM()
    ref_policy = TinyLM()
    critic = TinyCritic()
    cfg = RLHFConfig()
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=1e-3
    )
    return RLHFTrainer(policy, ref_policy, critic, cfg, opt)


# ---------------------------------------------------------------------------
# 1. RLHFConfig defaults
# ---------------------------------------------------------------------------

def test_rlhf_config_defaults():
    cfg = RLHFConfig()
    assert cfg.kl_coef == pytest.approx(0.1)
    assert cfg.clip_ratio == pytest.approx(0.2)
    assert cfg.vf_coef == pytest.approx(0.1)
    assert cfg.entropy_coef == pytest.approx(0.01)
    assert cfg.gamma == pytest.approx(1.0)
    assert cfg.lam == pytest.approx(0.95)
    assert cfg.reward_scale == pytest.approx(1.0)
    assert cfg.reward_clip is None


# ---------------------------------------------------------------------------
# 2. compute_kl_penalty — shape
# ---------------------------------------------------------------------------

def test_compute_kl_penalty_shape():
    lp = make_log_probs()
    ref = make_log_probs()
    kl = compute_kl_penalty(lp, ref)
    assert kl.shape == (B, T)


# ---------------------------------------------------------------------------
# 3. compute_kl_penalty — zero when log_probs are identical
# ---------------------------------------------------------------------------

def test_compute_kl_penalty_zero_when_same():
    lp = make_log_probs()
    kl = compute_kl_penalty(lp, lp)
    assert torch.allclose(kl, torch.zeros_like(kl))


# ---------------------------------------------------------------------------
# 4. clip_rewards — actually clips
# ---------------------------------------------------------------------------

def test_clip_rewards_clips_correctly():
    rewards = torch.tensor([[-3.0, 2.5, 0.0], [1.0, -1.5, 4.0]])
    clip_val = 2.0
    clipped = clip_rewards(rewards, clip_val)
    assert clipped.max().item() <= clip_val + 1e-6
    assert clipped.min().item() >= -clip_val - 1e-6
    # Exact check on a known value
    assert clipped[0, 0].item() == pytest.approx(-2.0)
    assert clipped[1, 2].item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 5. clip_rewards — no-op when clip_val is None
# ---------------------------------------------------------------------------

def test_clip_rewards_noop_when_none():
    rewards = torch.tensor([[-10.0, 10.0]])
    out = clip_rewards(rewards, clip_val=None)
    assert torch.equal(out, rewards)


# ---------------------------------------------------------------------------
# 6. whiten_rewards — mean ≈ 0
# ---------------------------------------------------------------------------

def test_whiten_rewards_mean_approx_zero():
    rewards = make_rewards()
    whitened = whiten_rewards(rewards)
    assert whitened.mean().abs().item() < 1e-5


# ---------------------------------------------------------------------------
# 7. whiten_rewards — std ≈ 1
# ---------------------------------------------------------------------------

def test_whiten_rewards_std_approx_one():
    rewards = make_rewards()
    whitened = whiten_rewards(rewards)
    # std over all elements should be close to 1
    assert abs(whitened.std().item() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 8. compute_returns — shape
# ---------------------------------------------------------------------------

def test_compute_returns_shape():
    rewards = make_rewards()
    returns = compute_returns(rewards, gamma=1.0)
    assert returns.shape == (B, T)


# ---------------------------------------------------------------------------
# 9. compute_returns — terminal position equals its own reward (gamma=1)
# ---------------------------------------------------------------------------

def test_compute_returns_terminal_equals_reward():
    rewards = make_rewards()
    returns = compute_returns(rewards, gamma=1.0)
    # At the last timestep G_{T-1} = r_{T-1}
    assert torch.allclose(returns[:, -1], rewards[:, -1])


# ---------------------------------------------------------------------------
# 10. compute_advantages_gae — shape
# ---------------------------------------------------------------------------

def test_compute_advantages_gae_shape():
    rewards = make_rewards()
    values = make_values()
    adv = compute_advantages_gae(rewards, values, gamma=0.99, lam=0.95)
    assert adv.shape == (B, T)


# ---------------------------------------------------------------------------
# 11. ppo_policy_loss — scalar
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_is_scalar():
    lp = make_log_probs()
    old_lp = make_log_probs()
    adv = torch.randn(B, T)
    loss = ppo_policy_loss(lp, old_lp, adv)
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# 12. ppo_policy_loss — non-positive when advantages positive and ratio ≈ 1
# ---------------------------------------------------------------------------

def test_ppo_policy_loss_nonpositive_pos_advantages_near_one():
    # When log_probs == old_log_probs, ratio == 1.
    # With all-positive advantages, surrogate = advantages > 0, so loss = -mean < 0.
    lp = make_log_probs()
    old_lp = lp.clone()
    adv = torch.abs(torch.randn(B, T)) + 0.1   # strictly positive
    loss = ppo_policy_loss(lp, old_lp, adv)
    assert loss.item() <= 0.0


# ---------------------------------------------------------------------------
# 13. ppo_value_loss — scalar
# ---------------------------------------------------------------------------

def test_ppo_value_loss_is_scalar():
    values = make_values()
    old_values = make_values()
    returns = make_values()
    loss = ppo_value_loss(values, old_values, returns)
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# 14. entropy_bonus — scalar and positive
# ---------------------------------------------------------------------------

def test_entropy_bonus_scalar_and_positive():
    # log-probs are negative, so -mean(log_probs) should be positive
    lp = -torch.abs(torch.randn(B, T))   # all strictly negative
    ent = entropy_bonus(lp)
    assert ent.shape == ()
    assert ent.item() > 0.0


# ---------------------------------------------------------------------------
# 15. RLHFTrainer.compute_loss — correct keys and scalar values
# ---------------------------------------------------------------------------

def test_rlhf_trainer_compute_loss_keys(tiny_trainer):
    input_ids = torch.randint(0, VOCAB, (B, 4))
    response_ids = torch.randint(0, VOCAB, (B, T))
    rewards = torch.randn(B, T)

    result = tiny_trainer.compute_loss(input_ids, response_ids, rewards)

    expected_keys = {"total_loss", "policy_loss", "value_loss", "kl_penalty", "entropy"}
    assert set(result.keys()) == expected_keys
    for key, val in result.items():
        assert val.shape == (), f"{key} should be a scalar but got shape {val.shape}"


# ---------------------------------------------------------------------------
# 16. RLHFTrainer.compute_loss — total_loss is differentiable
# ---------------------------------------------------------------------------

def test_rlhf_trainer_total_loss_differentiable(tiny_trainer):
    input_ids = torch.randint(0, VOCAB, (B, 4))
    response_ids = torch.randint(0, VOCAB, (B, T))
    rewards = torch.randn(B, T)

    result = tiny_trainer.compute_loss(input_ids, response_ids, rewards)
    result["total_loss"].backward()   # should not raise

    # At least some policy gradients must exist
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in tiny_trainer.policy.parameters()
    )
    assert has_grad
