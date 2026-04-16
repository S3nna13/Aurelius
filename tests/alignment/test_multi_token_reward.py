"""Tests for Multi-Token Reward Model (src/alignment/multi_token_reward.py).

Covers:
  1.  MultiTokenRMConfig defaults
  2.  TokenRewardHead output shape
  3.  MultiTokenRewardModel.forward tuple shapes
  4.  sequence_reward is scalar per batch item
  5.  get_process_rewards step count
  6.  compute_advantage: last token has highest advantage (equal positive rewards)
  7.  MultiTokenRMLoss with only seq_labels is finite
  8.  MultiTokenRMLoss with both labels combines losses correctly
  9.  Metrics dict has 'total_loss' key
  10. Gradient flows through token reward model
  11. attention_mask zeros out masked positions
"""

import math
import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.multi_token_reward import (
    MultiTokenRMConfig,
    TokenRewardHead,
    MultiTokenRewardModel,
    MultiTokenRMLoss,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB_SIZE = 256


@pytest.fixture(scope="module")
def aurelius_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=D_MODEL,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def rm_config():
    return MultiTokenRMConfig(d_model=D_MODEL)


@pytest.fixture(scope="module")
def reward_model(aurelius_cfg, rm_config):
    torch.manual_seed(0)
    def backbone_fn():
        return AureliusTransformer(aurelius_cfg)
    return MultiTokenRewardModel(backbone_fn, rm_config)


# ---------------------------------------------------------------------------
# Test 1: MultiTokenRMConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = MultiTokenRMConfig()
    assert cfg.d_model == 64
    assert cfg.dropout == 0.0
    assert cfg.discount == 0.99
    assert cfg.aggregate == "mean"
    assert cfg.token_weight == 0.5
    assert cfg.seq_weight == 0.5


# ---------------------------------------------------------------------------
# Test 2: TokenRewardHead output shape is (batch, seq)
# ---------------------------------------------------------------------------

def test_token_reward_head_shape():
    B, T, D = 3, 10, D_MODEL
    head = TokenRewardHead(D)
    hidden = torch.randn(B, T, D)
    out = head(hidden)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: MultiTokenRewardModel.forward returns (batch,seq) and (batch,)
# ---------------------------------------------------------------------------

def test_forward_shapes(reward_model):
    B, T = 2, 8
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    token_rewards, seq_reward = reward_model(input_ids)
    assert token_rewards.shape == (B, T), (
        f"token_rewards shape: expected ({B}, {T}), got {token_rewards.shape}"
    )
    assert seq_reward.shape == (B,), (
        f"seq_reward shape: expected ({B},), got {seq_reward.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: sequence_reward is scalar per batch item
# ---------------------------------------------------------------------------

def test_seq_reward_scalar_per_item(reward_model):
    B, T = 4, 12
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    _, seq_reward = reward_model(input_ids)
    # Must be 1-D with exactly B elements
    assert seq_reward.dim() == 1
    assert seq_reward.size(0) == B


# ---------------------------------------------------------------------------
# Test 5: get_process_rewards returns correct number of steps
# ---------------------------------------------------------------------------

def test_get_process_rewards_step_count(reward_model):
    T = 12
    input_ids = torch.randint(0, VOCAB_SIZE, (1, T))
    # Three boundaries → 4 steps: [0,4), [4,8), [8,10), [10,12)
    boundaries = [4, 8, 10]
    step_rewards = reward_model.get_process_rewards(input_ids, boundaries)
    expected_steps = len(boundaries) + 1  # 4
    assert step_rewards.shape == (expected_steps,), (
        f"Expected ({expected_steps},), got {step_rewards.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: compute_advantage — last token has highest advantage (equal +rewards)
# ---------------------------------------------------------------------------

def test_compute_advantage_last_token_highest(reward_model):
    # All-ones rewards: undiscounted, the last token should have the highest
    # advantage because it bears no future discount penalty.
    B, T = 1, 5
    token_rewards = torch.ones(B, T)
    advantages = reward_model.compute_advantage(token_rewards, discount=0.99)
    assert advantages.shape == (B, T)
    last_adv = advantages[0, -1].item()
    for t in range(T - 1):
        assert advantages[0, t].item() >= last_adv - 1e-6, (
            f"Advantage at position {t} ({advantages[0,t].item():.6f}) "
            f"should be >= last ({last_adv:.6f})"
        )
    # Specifically confirm the last position equals the reward (no future)
    assert math.isclose(last_adv, token_rewards[0, -1].item(), rel_tol=1e-5), (
        f"Last advantage {last_adv} should equal reward {token_rewards[0,-1].item()}"
    )


# ---------------------------------------------------------------------------
# Test 7: MultiTokenRMLoss with only seq_labels returns finite loss
# ---------------------------------------------------------------------------

def test_loss_seq_labels_only():
    B, T = 2, 8
    token_rewards = torch.randn(B, T)
    seq_rewards = torch.randn(B)
    seq_labels = torch.randn(B)

    loss_fn = MultiTokenRMLoss()
    loss, metrics = loss_fn.forward(token_rewards, seq_rewards, None, seq_labels)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert math.isfinite(metrics["total_loss"])


# ---------------------------------------------------------------------------
# Test 8: MultiTokenRMLoss with both labels combines losses correctly
# ---------------------------------------------------------------------------

def test_loss_both_labels_combined():
    B, T = 2, 8
    torch.manual_seed(42)
    token_rewards = torch.randn(B, T)
    seq_rewards = torch.randn(B)
    token_labels = torch.randn(B, T)
    seq_labels = torch.randn(B)

    token_weight = 0.3
    seq_weight = 0.7
    loss_fn = MultiTokenRMLoss(token_weight=token_weight, seq_weight=seq_weight)

    loss, metrics = loss_fn.forward(token_rewards, seq_rewards, token_labels, seq_labels)

    expected = (
        token_weight * metrics["token_loss"]
        + seq_weight * metrics["seq_loss"]
    )
    assert math.isclose(loss.item(), expected, rel_tol=1e-5), (
        f"total_loss {loss.item():.6f} != expected {expected:.6f}"
    )
    assert math.isclose(metrics["total_loss"], loss.item(), rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 9: Metrics dict has 'total_loss' key
# ---------------------------------------------------------------------------

def test_metrics_has_total_loss_key():
    B, T = 2, 6
    token_rewards = torch.randn(B, T)
    seq_rewards = torch.randn(B)
    seq_labels = torch.randn(B)

    loss_fn = MultiTokenRMLoss()
    _, metrics = loss_fn.forward(token_rewards, seq_rewards, None, seq_labels)

    assert "total_loss" in metrics, f"'total_loss' not found in metrics: {metrics.keys()}"


# ---------------------------------------------------------------------------
# Test 10: Gradient flows through token reward model (backward works)
# ---------------------------------------------------------------------------

def test_gradient_flows(aurelius_cfg, rm_config):
    torch.manual_seed(1)

    def backbone_fn():
        return AureliusTransformer(aurelius_cfg)

    model = MultiTokenRewardModel(backbone_fn, rm_config)
    model.train()

    B, T = 2, 8
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    token_rewards, seq_rewards = model(input_ids)

    # Scalar loss
    loss = token_rewards.sum() + seq_rewards.sum()
    loss.backward()

    # At least one parameter in the reward head should have a non-None gradient
    grad_exists = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.reward_head.parameters()
    )
    assert grad_exists, "No gradient flowed through the reward head parameters."


# ---------------------------------------------------------------------------
# Test 11: attention_mask zeros out masked positions
# ---------------------------------------------------------------------------

def test_attention_mask_zeros_masked_positions(reward_model):
    B, T = 2, 8
    torch.manual_seed(7)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))

    # Mask the last two token positions for every batch item.
    attention_mask = torch.ones(B, T)
    attention_mask[:, -2:] = 0.0  # last 2 positions are padding

    token_rewards_masked, _ = reward_model(input_ids, attention_mask=attention_mask)

    # Masked positions must be exactly zero.
    for b in range(B):
        for t in [T - 2, T - 1]:
            val = token_rewards_masked[b, t].item()
            assert val == 0.0, (
                f"Expected token_rewards[{b},{t}] == 0 (masked), got {val}"
            )
