"""Tests for GRPO v3: Group Relative Policy Optimization (Shao et al. 2024).

Covers:
  1.  GRPOConfig default values
  2.  GroupRewardNormalizer.normalize output shape  (B*G,)
  3.  Normalised advantages have zero mean per group
  4.  Normalised advantages have unit std per group (when rewards are non-uniform)
  5.  normalize_batch input/output shapes
  6.  All-equal rewards per group → zero advantages
  7.  GRPOLoss.clip_ratio clips correctly (asymmetric upper/lower)
  8.  policy_loss returns a scalar finite value
  9.  kl_penalty returns a scalar finite value
  10. kl_penalty is non-negative in expectation (zero when policies are equal)
  11. GRPOLoss.forward returns the four required metric keys
  12. GRPOLoss.forward total_loss == policy_loss + beta * kl_loss
  13. GRPOLoss gradient flows through log_probs
  14. GRPOTrainer.freeze_ref freezes all reference model parameters
  15. GRPOTrainer.grpo_step returns the four required metric keys
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from aurelius.alignment.grpo_v3 import (
    GRPOConfig,
    GroupRewardNormalizer,
    GRPOLoss,
    GRPOTrainer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> GRPOConfig:
    return GRPOConfig(beta=0.04, group_size=4, epsilon_low=0.2, epsilon_high=0.3)


@pytest.fixture
def normalizer(cfg: GRPOConfig) -> GroupRewardNormalizer:
    return GroupRewardNormalizer(group_size=cfg.group_size)


@pytest.fixture
def loss_fn(cfg: GRPOConfig) -> GRPOLoss:
    return GRPOLoss(cfg)


def _simple_model(n_params: int = 8) -> nn.Linear:
    """Return a tiny trainable linear model."""
    return nn.Linear(n_params, 1, bias=False)


def _flat_log_probs(B: int, G: int, requires_grad: bool = False) -> torch.Tensor:
    t = torch.randn(B * G)
    if requires_grad:
        t = t.detach().requires_grad_(True)
    return t


# ---------------------------------------------------------------------------
# 1. GRPOConfig defaults
# ---------------------------------------------------------------------------

class TestGRPOConfig:
    def test_defaults(self) -> None:
        c = GRPOConfig()
        assert c.beta == pytest.approx(0.04)
        assert c.group_size == 8
        assert c.clip_ratio == pytest.approx(0.2)
        assert c.epsilon_low == pytest.approx(0.2)
        assert c.epsilon_high == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 2-6. GroupRewardNormalizer
# ---------------------------------------------------------------------------

class TestGroupRewardNormalizer:
    def test_normalize_output_shape(self, normalizer: GroupRewardNormalizer) -> None:
        """normalize() must return a flat (B*G,) tensor."""
        B, G = 3, 4
        rewards = torch.randn(B * G)
        adv = normalizer.normalize(rewards)
        assert adv.shape == (B * G,), f"expected ({B * G},), got {adv.shape}"

    def test_normalize_zero_mean_per_group(self, normalizer: GroupRewardNormalizer) -> None:
        """Each group of G advantages must have mean ≈ 0."""
        B, G = 5, 4
        rewards = torch.randn(B * G)
        adv = normalizer.normalize(rewards)
        adv_2d = adv.reshape(B, G)
        group_means = adv_2d.mean(dim=1)
        assert torch.allclose(group_means, torch.zeros(B), atol=1e-5), (
            f"group means not zero: {group_means}"
        )

    def test_normalize_unit_std_per_group(self, normalizer: GroupRewardNormalizer) -> None:
        """Each group of G advantages must have std ≈ 1 (when rewards vary)."""
        B, G = 4, 4
        # Ensure each group has distinct rewards to avoid degenerate std = 0
        rewards = torch.arange(float(B * G))  # strictly increasing
        adv = normalizer.normalize(rewards)
        adv_2d = adv.reshape(B, G)
        group_stds = adv_2d.std(dim=1)  # unbiased std
        assert torch.allclose(group_stds, torch.ones(B), atol=1e-5), (
            f"group stds not unit: {group_stds}"
        )

    def test_normalize_batch_shape(self, normalizer: GroupRewardNormalizer) -> None:
        """normalize_batch() must accept (B, G) and return (B, G)."""
        B, G = 6, 4
        rewards_2d = torch.randn(B, G)
        adv = normalizer.normalize_batch(rewards_2d)
        assert adv.shape == (B, G), f"expected ({B},{G}), got {adv.shape}"

    def test_normalize_batch_zero_mean(self, normalizer: GroupRewardNormalizer) -> None:
        """normalize_batch(): each row has mean ≈ 0."""
        B, G = 3, 4
        rewards_2d = torch.randn(B, G)
        adv = normalizer.normalize_batch(rewards_2d)
        row_means = adv.mean(dim=1)
        assert torch.allclose(row_means, torch.zeros(B), atol=1e-5)

    def test_all_equal_rewards_gives_zero_advantages(
        self, normalizer: GroupRewardNormalizer
    ) -> None:
        """When all rewards in a group are equal, advantages must be 0."""
        B, G = 3, 4
        rewards = torch.ones(B * G)  # all equal within every group
        adv = normalizer.normalize(rewards)
        assert torch.allclose(adv, torch.zeros(B * G), atol=1e-6), (
            f"expected zeros, got {adv}"
        )


# ---------------------------------------------------------------------------
# 7. GRPOLoss.clip_ratio — asymmetric clipping
# ---------------------------------------------------------------------------

class TestClipRatio:
    def test_clips_below_lower_bound(self, loss_fn: GRPOLoss) -> None:
        """Ratios below 1-epsilon_low must be clipped up."""
        ratio = torch.tensor([0.5])  # well below 1 - 0.2 = 0.8
        clipped = loss_fn.clip_ratio(ratio)
        expected_lo = 1.0 - loss_fn.config.epsilon_low
        assert clipped.item() == pytest.approx(expected_lo, abs=1e-6)

    def test_clips_above_upper_bound(self, loss_fn: GRPOLoss) -> None:
        """Ratios above 1+epsilon_high must be clipped down."""
        ratio = torch.tensor([2.0])  # well above 1 + 0.3 = 1.3
        clipped = loss_fn.clip_ratio(ratio)
        expected_hi = 1.0 + loss_fn.config.epsilon_high
        assert clipped.item() == pytest.approx(expected_hi, abs=1e-6)

    def test_asymmetric_bounds_differ(self, loss_fn: GRPOLoss) -> None:
        """epsilon_low != epsilon_high → lower and upper bounds are not equal."""
        lo = 1.0 - loss_fn.config.epsilon_low
        hi = 1.0 + loss_fn.config.epsilon_high
        assert lo != hi, "asymmetric test requires epsilon_low != epsilon_high"

    def test_in_range_ratio_unchanged(self, loss_fn: GRPOLoss) -> None:
        """A ratio inside [1-eps_low, 1+eps_high] must not be clipped."""
        ratio = torch.tensor([1.0])
        clipped = loss_fn.clip_ratio(ratio)
        assert clipped.item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. policy_loss
# ---------------------------------------------------------------------------

class TestPolicyLoss:
    def test_scalar_and_finite(self, loss_fn: GRPOLoss) -> None:
        B = 16
        log_probs = torch.randn(B)
        old_log_probs = torch.randn(B)
        advantages = torch.randn(B)
        loss = loss_fn.policy_loss(log_probs, old_log_probs, advantages)
        assert loss.shape == (), f"expected scalar, got shape {loss.shape}"
        assert math.isfinite(loss.item()), f"policy_loss not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# 9-10. kl_penalty
# ---------------------------------------------------------------------------

class TestKLPenalty:
    def test_scalar_and_finite(self, loss_fn: GRPOLoss) -> None:
        B = 16
        log_probs = torch.randn(B)
        ref_log_probs = torch.randn(B)
        kl = loss_fn.kl_penalty(log_probs, ref_log_probs)
        assert kl.shape == (), f"expected scalar, got shape {kl.shape}"
        assert math.isfinite(kl.item()), f"kl_penalty not finite: {kl.item()}"

    def test_zero_when_policies_equal(self, loss_fn: GRPOLoss) -> None:
        """KL(p||p) = 0."""
        log_probs = torch.randn(32)
        kl = loss_fn.kl_penalty(log_probs, log_probs)
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self, loss_fn: GRPOLoss) -> None:
        """The approximation exp(x) - x - 1 >= 0 for all real x."""
        for _ in range(5):
            log_probs = torch.randn(64)
            ref_log_probs = torch.randn(64)
            kl = loss_fn.kl_penalty(log_probs, ref_log_probs)
            assert kl.item() >= -1e-6, f"kl_penalty negative: {kl.item()}"


# ---------------------------------------------------------------------------
# 11-12. GRPOLoss.forward
# ---------------------------------------------------------------------------

class TestGRPOLossForward:
    def test_required_keys(self, loss_fn: GRPOLoss) -> None:
        B = 8
        log_probs = torch.randn(B, requires_grad=True)
        old_lp = torch.randn(B)
        ref_lp = torch.randn(B)
        adv = torch.randn(B)
        _, metrics = loss_fn(log_probs, old_lp, ref_lp, adv)
        for key in ("policy_loss", "kl_loss", "total_loss", "mean_advantage"):
            assert key in metrics, f"missing key '{key}' in metrics"

    def test_total_loss_equals_sum(self, loss_fn: GRPOLoss) -> None:
        """total_loss must equal policy_loss + beta * kl_loss."""
        B = 8
        log_probs = torch.randn(B, requires_grad=True)
        old_lp = torch.randn(B)
        ref_lp = torch.randn(B)
        adv = torch.randn(B)
        _, metrics = loss_fn(log_probs, old_lp, ref_lp, adv)
        expected = metrics["policy_loss"] + loss_fn.config.beta * metrics["kl_loss"]
        assert metrics["total_loss"] == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# 13. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_through_log_probs(self, loss_fn: GRPOLoss) -> None:
        B = 8
        log_probs = torch.randn(B, requires_grad=True)
        old_lp = torch.randn(B).detach()
        ref_lp = torch.randn(B).detach()
        adv = torch.randn(B).detach()
        loss, _ = loss_fn(log_probs, old_lp, ref_lp, adv)
        loss.backward()
        assert log_probs.grad is not None, "gradient is None"
        assert torch.isfinite(log_probs.grad).all(), "gradient contains non-finite values"


# ---------------------------------------------------------------------------
# 14. GRPOTrainer.freeze_ref
# ---------------------------------------------------------------------------

class TestGRPOTrainerFreezeRef:
    def test_freeze_ref_sets_no_grad(self, cfg: GRPOConfig) -> None:
        policy = _simple_model()
        ref = _simple_model()
        opt = torch.optim.SGD(policy.parameters(), lr=1e-3)
        loss_fn = GRPOLoss(cfg)
        norm = GroupRewardNormalizer(cfg.group_size)
        trainer = GRPOTrainer(policy, ref, opt, cfg, loss_fn, norm)

        # Before freezing, ref params should have grad by default
        trainer.freeze_ref()

        for name, param in ref.named_parameters():
            assert not param.requires_grad, (
                f"ref param '{name}' still has requires_grad=True after freeze_ref()"
            )


# ---------------------------------------------------------------------------
# 15. GRPOTrainer.grpo_step
# ---------------------------------------------------------------------------

class TestGRPOTrainerStep:
    def test_grpo_step_returns_correct_keys(self, cfg: GRPOConfig) -> None:
        B, G = 2, cfg.group_size
        policy = _simple_model()
        ref = _simple_model()
        opt = torch.optim.SGD(policy.parameters(), lr=1e-3)
        loss_fn = GRPOLoss(cfg)
        norm = GroupRewardNormalizer(G)
        trainer = GRPOTrainer(policy, ref, opt, cfg, loss_fn, norm)
        trainer.freeze_ref()

        log_probs = _flat_log_probs(B, G, requires_grad=True)
        old_lp = _flat_log_probs(B, G)
        ref_lp = _flat_log_probs(B, G)
        rewards = torch.randn(B * G)

        metrics = trainer.grpo_step(log_probs, old_lp, ref_lp, rewards)

        for key in ("policy_loss", "kl_loss", "total_loss", "mean_advantage"):
            assert key in metrics, f"missing key '{key}' in grpo_step metrics"

    def test_grpo_step_metrics_are_finite(self, cfg: GRPOConfig) -> None:
        """All metric values returned by grpo_step must be finite floats."""
        B, G = 2, cfg.group_size
        policy = _simple_model()
        ref = _simple_model()
        opt = torch.optim.SGD(policy.parameters(), lr=1e-3)
        loss_fn = GRPOLoss(cfg)
        norm = GroupRewardNormalizer(G)
        trainer = GRPOTrainer(policy, ref, opt, cfg, loss_fn, norm)
        trainer.freeze_ref()

        log_probs = _flat_log_probs(B, G, requires_grad=True)
        old_lp = _flat_log_probs(B, G)
        ref_lp = _flat_log_probs(B, G)
        rewards = torch.randn(B * G)

        metrics = trainer.grpo_step(log_probs, old_lp, ref_lp, rewards)

        for key, val in metrics.items():
            assert math.isfinite(val), f"metric '{key}' is not finite: {val}"
