"""Tests for PRIME: Process Reward Model via Implicit Reward (arXiv:2502.01456).

12 focused tests covering:
  1.  compute_implicit_rewards output shape (B, T)
  2.  equal log_probs → zero implicit reward
  3.  better than ref (log_prob > ref) → positive reward
  4.  mask zeros out padding positions
  5.  aggregate credit_mode="last" picks last valid token
  6.  aggregate credit_mode="mean" averages over valid tokens
  7.  forward returns (Tensor, dict) tuple
  8.  forward output has shape (B, T)
  9.  outcome_rewards are added to the step aggregate
  10. normalize=True → rewards have unit std
  11. no NaN/Inf with extreme log_probs (-100, 0)
  12. gradient flows through compute_implicit_rewards
"""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.prime import PRIMEConfig, PRIMEReward

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T = 4, 10  # default batch size and sequence length


@pytest.fixture
def default_module():
    """PRIMEReward with default config."""
    return PRIMEReward()


@pytest.fixture
def mean_module():
    """PRIMEReward with credit_mode='mean'."""
    return PRIMEReward(PRIMEConfig(credit_mode="mean", normalize=False))


@pytest.fixture
def last_module():
    """PRIMEReward with credit_mode='last' and normalize=False."""
    return PRIMEReward(PRIMEConfig(credit_mode="last", normalize=False))


@pytest.fixture
def full_mask():
    """All-ones mask — no padding."""
    return torch.ones(B, T, dtype=torch.long)


@pytest.fixture
def partial_mask():
    """Each row has a different number of valid tokens."""
    mask = torch.ones(B, T, dtype=torch.long)
    # Row 0: all valid (10), row 1: 8, row 2: 6, row 3: 4
    for i, valid in enumerate([10, 8, 6, 4]):
        mask[i, valid:] = 0
    return mask


# ---------------------------------------------------------------------------
# Test 1 — output shape (B, T)
# ---------------------------------------------------------------------------


def test_compute_implicit_rewards_shape(default_module, full_mask):
    """compute_implicit_rewards must return a (B, T) tensor."""
    log_probs = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)
    out = default_module.compute_implicit_rewards(log_probs, ref_log_probs, full_mask)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2 — equal log_probs → zero reward
# ---------------------------------------------------------------------------


def test_equal_log_probs_zero_reward(default_module, full_mask):
    """When policy == reference, the implicit reward must be exactly zero."""
    log_probs = torch.randn(B, T)
    out = default_module.compute_implicit_rewards(log_probs, log_probs, full_mask)
    assert torch.allclose(out, torch.zeros_like(out)), (
        "Equal log_probs should yield zero implicit rewards."
    )


# ---------------------------------------------------------------------------
# Test 3 — policy better than ref → positive reward
# ---------------------------------------------------------------------------


def test_positive_reward_when_policy_better(default_module, full_mask):
    """log_prob > ref_log_prob must produce positive implicit rewards."""
    ref_log_probs = torch.full((B, T), -2.0)
    log_probs = torch.full((B, T), -1.0)  # policy assigns higher prob
    out = default_module.compute_implicit_rewards(log_probs, ref_log_probs, full_mask)
    assert (out > 0).all(), "All rewards should be positive when policy > ref."


# ---------------------------------------------------------------------------
# Test 4 — mask zeros out padding
# ---------------------------------------------------------------------------


def test_mask_zeros_padding(default_module, partial_mask):
    """Positions where mask==0 must have reward exactly 0."""
    log_probs = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)
    out = default_module.compute_implicit_rewards(log_probs, ref_log_probs, partial_mask)
    padding_rewards = out[partial_mask == 0]
    assert torch.allclose(padding_rewards, torch.zeros_like(padding_rewards)), (
        "Padded positions must have zero reward."
    )


# ---------------------------------------------------------------------------
# Test 5 — aggregate credit_mode="last"
# ---------------------------------------------------------------------------


def test_aggregate_last_picks_last_valid(last_module, partial_mask):
    """credit_mode='last' must return the reward at the last valid token."""
    # Build known token_rewards: row i has value float(i) at every position
    token_rewards = torch.zeros(B, T)
    for i in range(B):
        token_rewards[i] = float(i + 1)
    token_rewards = token_rewards * partial_mask.float()

    step = last_module.aggregate_step_rewards(token_rewards, partial_mask)

    assert step.shape == (B,), f"Expected (B,)={B}, got {step.shape}"
    # For each row the last valid token has value float(i+1)
    expected = torch.tensor([float(i + 1) for i in range(B)])
    assert torch.allclose(step, expected), f"Expected {expected.tolist()}, got {step.tolist()}"


# ---------------------------------------------------------------------------
# Test 6 — aggregate credit_mode="mean"
# ---------------------------------------------------------------------------


def test_aggregate_mean_averages_valid(mean_module, partial_mask):
    """credit_mode='mean' must equal the mean of valid token rewards."""
    torch.manual_seed(7)
    token_rewards = torch.randn(B, T) * partial_mask.float()

    step = mean_module.aggregate_step_rewards(token_rewards, partial_mask)

    assert step.shape == (B,)
    valid_counts = [10, 8, 6, 4]
    for i, n in enumerate(valid_counts):
        expected_mean = token_rewards[i, :n].mean()
        assert torch.isclose(step[i], expected_mean, atol=1e-5), (
            f"Row {i}: expected {expected_mean.item():.6f}, got {step[i].item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 7 — forward returns (Tensor, dict)
# ---------------------------------------------------------------------------


def test_forward_returns_tuple(default_module, full_mask):
    """forward() must return a (Tensor, dict) tuple."""
    log_probs = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)
    outcome_rewards = torch.randn(B)

    result = default_module(log_probs, ref_log_probs, outcome_rewards, full_mask)

    assert isinstance(result, tuple) and len(result) == 2, "forward must return a 2-tuple."
    dense, metrics = result
    assert isinstance(dense, torch.Tensor), "First element must be a Tensor."
    assert isinstance(metrics, dict), "Second element must be a dict."


# ---------------------------------------------------------------------------
# Test 8 — forward dense_rewards shape (B, T)
# ---------------------------------------------------------------------------


def test_forward_output_shape(default_module, full_mask):
    """forward() dense_rewards must have shape (B, T)."""
    log_probs = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)
    outcome_rewards = torch.randn(B)

    dense, _ = default_module(log_probs, ref_log_probs, outcome_rewards, full_mask)

    assert dense.shape == (B, T), f"Expected ({B}, {T}), got {dense.shape}"


# ---------------------------------------------------------------------------
# Test 9 — outcome_rewards contribute to aggregate
# ---------------------------------------------------------------------------


def test_outcome_rewards_added(full_mask):
    """Outcome rewards must shift the combined reward by their value."""
    module_no_norm = PRIMEReward(PRIMEConfig(normalize=False, credit_mode="mean"))

    log_probs = torch.zeros(B, T)
    ref_log_probs = torch.zeros(B, T)

    outcome_a = torch.zeros(B)
    outcome_b = torch.ones(B) * 5.0

    dense_a, metrics_a = module_no_norm(log_probs, ref_log_probs, outcome_a, full_mask)
    dense_b, metrics_b = module_no_norm(log_probs, ref_log_probs, outcome_b, full_mask)

    # dense reward at valid positions should differ by exactly 5
    diff = (dense_b - dense_a) * full_mask.float()
    expected_diff = outcome_b.unsqueeze(1).expand(B, T) * full_mask.float()
    assert torch.allclose(diff, expected_diff, atol=1e-5), (
        "Outcome reward must shift dense rewards by outcome value."
    )


# ---------------------------------------------------------------------------
# Test 10 — normalize=True → unit std
# ---------------------------------------------------------------------------


def test_normalize_unit_std(full_mask):
    """With normalize=True the token rewards (before outcome) must have unit std."""
    module = PRIMEReward(PRIMEConfig(normalize=True, credit_mode="mean"))

    torch.manual_seed(99)
    log_probs = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)
    outcome_rewards = torch.zeros(B)  # zero outcome so we isolate implicit rewards

    dense, _ = module(log_probs, ref_log_probs, outcome_rewards, full_mask)

    # Compute std over all valid (non-zero-mask) positions
    valid_rewards = dense[full_mask == 1]
    std = valid_rewards.std()
    # Should be close to 1 — allow tolerance due to finite-sample normalization
    assert torch.isclose(std, torch.tensor(1.0), atol=0.1), (
        f"Expected std ≈ 1.0 after normalization, got {std.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 11 — no NaN/Inf with extreme log_probs
# ---------------------------------------------------------------------------


def test_no_nan_inf_extreme_log_probs(default_module, full_mask):
    """No NaN or Inf should appear with extreme log_prob values (-100, 0)."""
    log_probs = torch.full((B, T), -100.0)
    ref_log_probs = torch.zeros(B, T)
    outcome_rewards = torch.zeros(B)

    dense, metrics = default_module(log_probs, ref_log_probs, outcome_rewards, full_mask)

    assert not torch.isnan(dense).any(), "NaN detected in dense_rewards."
    assert not torch.isinf(dense).any(), "Inf detected in dense_rewards."
    for key, val in metrics.items():
        assert math.isfinite(val), f"Metric '{key}' is not finite: {val}"


# ---------------------------------------------------------------------------
# Test 12 — gradient flows through compute_implicit_rewards
# ---------------------------------------------------------------------------


def test_gradient_flows(default_module, full_mask):
    """Gradients must flow back through compute_implicit_rewards to log_probs."""
    log_probs = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.randn(B, T)

    rewards = default_module.compute_implicit_rewards(log_probs, ref_log_probs, full_mask)
    loss = rewards.sum()
    loss.backward()

    assert log_probs.grad is not None, "No gradient on log_probs."
    assert not torch.isnan(log_probs.grad).any(), "NaN in gradients."
    assert (log_probs.grad != 0).any(), "All gradients are zero — no signal."
