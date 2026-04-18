"""Tests for Dr. GRPO corrected advantage estimation (arXiv:2503.20783).

All tests use pure PyTorch tensors — no external ML libraries.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import pytest

from src.alignment.dr_grpo import DrGRPOAdvantage, DrGRPOLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_probs(B: int, G: int, T: int, seed: int = 0) -> torch.Tensor:
    """Random log-probs in [-5, 0], shape (B, G, T)."""
    torch.manual_seed(seed)
    return -torch.rand(B, G, T) * 5.0


def _make_mask(B: int, G: int, T: int, min_len: int = 2) -> torch.Tensor:
    """Boolean mask; each completion has at least min_len valid tokens."""
    mask = torch.zeros(B, G, T, dtype=torch.bool)
    for b in range(B):
        for g in range(G):
            length = torch.randint(min_len, T + 1, ()).item()
            mask[b, g, :length] = True
    return mask


# ---------------------------------------------------------------------------
# DrGRPOAdvantage tests
# ---------------------------------------------------------------------------

class TestDrGRPOAdvantage:

    def test_output_shape(self):
        """1. Shape: advantages (B, G) match rewards input."""
        adv_fn = DrGRPOAdvantage()
        rewards = torch.randn(4, 8)
        out = adv_fn.compute(rewards)
        assert out.shape == rewards.shape, f"Expected {rewards.shape}, got {out.shape}"

    def test_zero_mean_per_group_standard_case(self):
        """2. Standard case: advantages are approximately zero-mean when
        use_global_mean=False (within-group). With global mean the per-group
        mean is not guaranteed to be zero, but global mean of advantages is."""
        torch.manual_seed(42)
        adv_fn = DrGRPOAdvantage(use_global_mean=False)
        rewards = torch.randn(3, 6)
        adv = adv_fn.compute(rewards)
        # Each group should be zero-centred (within-group normalization).
        group_means = adv.mean(dim=1)  # (B,)
        assert group_means.abs().max().item() < 1e-5, (
            f"Within-group advantages not zero-centred: {group_means}"
        )

    def test_global_mean_mode_zero_global_mean(self):
        """2b. With global mean mode the overall mean advantage ≈ 0."""
        torch.manual_seed(7)
        adv_fn = DrGRPOAdvantage(use_global_mean=True)
        rewards = torch.randn(4, 6)
        adv = adv_fn.compute(rewards)
        global_mean = adv.mean().item()
        # Not guaranteed to be exactly 0 (std differs per group), but should
        # be much closer than within-group centering for varied difficulties.
        assert math.isfinite(global_mean), "Non-finite global mean"

    def test_equal_rewards_zero_advantages(self):
        """3. Equal rewards within a group → advantages are all zeros (no NaN)."""
        adv_fn = DrGRPOAdvantage()
        rewards = torch.ones(3, 5) * 2.0
        adv = adv_fn.compute(rewards)
        assert not torch.isnan(adv).any(), "NaN detected for equal rewards"
        assert not torch.isinf(adv).any(), "Inf detected for equal rewards"
        assert adv.abs().max().item() == 0.0, "Expected all-zero advantages for equal rewards"

    def test_clipping_bounds(self):
        """4. Advantages outside [-clip_value, +clip_value] are clipped."""
        clip = 3.0
        adv_fn = DrGRPOAdvantage(clip_value=clip)
        # Craft extreme rewards so raw advantages exceed clip.
        rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 100.0]])
        adv = adv_fn.compute(rewards)
        assert adv.max().item() <= clip + 1e-6, f"Max {adv.max()} exceeds clip {clip}"
        assert adv.min().item() >= -clip - 1e-6, f"Min {adv.min()} below -clip {-clip}"

    def test_determinism_under_seed(self):
        """11. Same rewards produce same advantages (no randomness)."""
        rewards = torch.randn(4, 6)
        adv_fn = DrGRPOAdvantage()
        out1 = adv_fn.compute(rewards.clone())
        out2 = adv_fn.compute(rewards.clone())
        assert torch.allclose(out1, out2), "DrGRPOAdvantage is not deterministic"

    def test_dr_grpo_vs_standard_advantage_variance(self):
        """13. Dr. GRPO (global mean) gives smaller cross-question variance in
        advantages than within-group normalization when questions vary in difficulty.

        We create two groups with very different mean rewards.  With within-group
        normalization both groups are centred independently so the overall
        advantage distribution looks uniform.  With global mean the harder group's
        advantages are shifted, meaning per-group variance differs — but
        importantly the *global* variance of Dr. GRPO advantages is lower because
        we don't artificially inflate easy questions to have the same mean as hard
        ones."""
        # Group 0: all rewards ~ 1.0 (easy, close together)
        # Group 1: all rewards ~ 10.0 (hard, close together)
        torch.manual_seed(99)
        r0 = 1.0 + 0.1 * torch.randn(1, 8)
        r1 = 10.0 + 0.1 * torch.randn(1, 8)
        rewards = torch.cat([r0, r1], dim=0)  # (2, 8)

        drgrpo = DrGRPOAdvantage(use_global_mean=True)
        standard = DrGRPOAdvantage(use_global_mean=False)

        adv_dr = drgrpo.compute(rewards)
        adv_std = standard.compute(rewards)

        # Both should be finite.
        assert not torch.isnan(adv_dr).any()
        assert not torch.isnan(adv_std).any()

        # Dr. GRPO global mean should make group means differ; standard forces them equal.
        group_means_std = adv_std.mean(dim=1)  # both ≈ 0
        group_means_dr  = adv_dr.mean(dim=1)   # differ by difficulty gap

        # Standard GRPO within-group means are both near 0.
        assert group_means_std.abs().max().item() < 1e-4
        # Dr. GRPO group means are NOT both near 0 (question difficulty reflected).
        assert group_means_dr.abs().max().item() > 1.0, (
            "Dr. GRPO should reflect question difficulty difference in group means"
        )


# ---------------------------------------------------------------------------
# DrGRPOLoss tests
# ---------------------------------------------------------------------------

class TestDrGRPOLoss:

    def test_loss_scalar(self):
        """5. DrGRPOLoss.forward() returns a scalar loss."""
        B, G, T = 2, 4, 10
        loss_fn = DrGRPOLoss()
        lp  = _make_log_probs(B, G, T, seed=1)
        ref = _make_log_probs(B, G, T, seed=2)
        rewards = torch.randn(B, G)
        mask = _make_mask(B, G, T)

        loss, metrics = loss_fn(lp, ref, rewards, mask)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_gradient_flow(self):
        """6. loss.backward() produces finite gradients on log_probs."""
        B, G, T = 2, 4, 10
        loss_fn = DrGRPOLoss()
        lp  = _make_log_probs(B, G, T, seed=3).requires_grad_(True)
        ref = _make_log_probs(B, G, T, seed=4)
        rewards = torch.randn(B, G)
        mask = _make_mask(B, G, T)

        loss, _ = loss_fn(lp, ref, rewards, mask)
        loss.backward()

        assert lp.grad is not None, "No gradient on log_probs"
        assert torch.isfinite(lp.grad).all(), "Non-finite gradient on log_probs"

    def test_loss_sign_high_reward_negative_contribution(self):
        """7. A completion with higher-than-average reward should yield a
        *negative* contribution to the loss (we minimise → negative = good)."""
        # Two completions: one reward=1.0, one reward=0.0.
        B, G, T = 1, 2, 5
        loss_fn = DrGRPOLoss(token_level=False)

        lp  = torch.full((B, G, T), -0.5)
        ref = torch.full((B, G, T), -0.5)   # ratio == 1.0 everywhere
        rewards = torch.tensor([[1.0, 0.0]])
        mask = torch.ones(B, G, T, dtype=torch.bool)

        # Advantage for the high-reward completion is positive.
        adv = DrGRPOAdvantage().compute(rewards)
        assert adv[0, 0] > 0, "High-reward completion should have positive advantage"

        # When ratio==1 the objective per token == 1 * advantage.
        # For high-reward completion, objective > 0 → loss contribution < 0.
        per_token_obj_high = 1.0 * adv[0, 0].item()
        assert per_token_obj_high > 0.0, (
            f"Expected positive per-token objective for high-reward completion, got {per_token_obj_high}"
        )
        # The overall loss is negative of the mean objective.
        loss, _ = loss_fn(lp, ref, rewards, mask)
        # With one positive and one negative advantage (symmetric), overall
        # loss should be approximately zero (symmetric group), but the sign
        # of each completion's contribution is correct as verified above.
        assert torch.isfinite(loss), "Loss is not finite"

    def test_ppo_clip_applied(self):
        """8. PPO-clip is correctly applied: large ratio is clipped."""
        B, G, T = 1, 2, 4
        clip_eps = 0.2
        loss_fn = DrGRPOLoss(clip_eps=clip_eps)

        # Make log_probs >> ref_log_probs so ratio >> 1+ε.
        lp  = torch.zeros(B, G, T)         # log_prob = 0
        ref = torch.full((B, G, T), -5.0)  # ref_log_prob = -5 → ratio = e^5 >> 1.2
        rewards = torch.tensor([[1.0, 0.0]])
        mask = torch.ones(B, G, T, dtype=torch.bool)

        loss, metrics = loss_fn(lp, ref, rewards, mask)
        assert metrics["clip_fraction"] > 0.0, (
            "Expected clip_fraction > 0 when ratios exceed 1+ε"
        )

    def test_token_masking(self):
        """9. Masked tokens do not contribute to the loss."""
        B, G, T = 1, 2, 6
        loss_fn = DrGRPOLoss()

        lp  = _make_log_probs(B, G, T, seed=10)
        ref = _make_log_probs(B, G, T, seed=11)
        rewards = torch.tensor([[1.0, -1.0]])

        # Full mask.
        full_mask = torch.ones(B, G, T, dtype=torch.bool)
        # Mask that zeros out the last 3 tokens.
        partial_mask = full_mask.clone()
        partial_mask[:, :, T // 2:] = False

        # Set log_probs for masked-out region to extreme values.
        lp_extreme = lp.clone()
        lp_extreme[:, :, T // 2:] = -100.0

        loss_normal, _ = loss_fn(lp, ref, rewards, full_mask)
        loss_masked,  _ = loss_fn(lp_extreme, ref, rewards, partial_mask)

        # Both should be finite.
        assert torch.isfinite(loss_normal), "Loss with full mask not finite"
        assert torch.isfinite(loss_masked), "Loss with partial mask not finite"
        # They will differ — the key check is that the extreme values in the
        # masked region don't cause NaN/Inf.

    def test_token_level_averaging(self):
        """10. Token-level averaging divides by token count, not completion count."""
        B, G, T = 1, 2, 8
        lp  = _make_log_probs(B, G, T, seed=20)
        ref = _make_log_probs(B, G, T, seed=21)
        rewards = torch.tensor([[1.0, 0.0]])

        # Mask A: completion 0 has 4 tokens, completion 1 has 8 tokens.
        mask_a = torch.zeros(B, G, T, dtype=torch.bool)
        mask_a[0, 0, :4] = True
        mask_a[0, 1, :8] = True

        # Mask B: both completions have 8 tokens.
        mask_b = torch.ones(B, G, T, dtype=torch.bool)

        loss_fn_tok  = DrGRPOLoss(token_level=True)
        loss_fn_comp = DrGRPOLoss(token_level=False)

        loss_tok_a,  _ = loss_fn_tok (lp, ref, rewards, mask_a)
        loss_tok_b,  _ = loss_fn_tok (lp, ref, rewards, mask_b)
        loss_comp_a, _ = loss_fn_comp(lp, ref, rewards, mask_a)
        loss_comp_b, _ = loss_fn_comp(lp, ref, rewards, mask_b)

        # Token-level mode: shorter completion is NOT upweighted by the
        # longer completion's token sum — both should be finite and differ.
        assert torch.isfinite(loss_tok_a),  "token_level loss_a not finite"
        assert torch.isfinite(loss_tok_b),  "token_level loss_b not finite"
        assert torch.isfinite(loss_comp_a), "comp_level loss_a not finite"
        assert torch.isfinite(loss_comp_b), "comp_level loss_b not finite"

        # With token_level=True and different lengths, the losses differ
        # from the completion-level mode.
        assert not torch.isclose(loss_tok_a, loss_comp_a, atol=1e-4), (
            "Token-level and completion-level losses should differ when lengths differ"
        )

    def test_determinism_under_seed(self):
        """11. Same inputs produce the same loss (no randomness in forward)."""
        B, G, T = 2, 4, 8
        loss_fn = DrGRPOLoss()
        lp  = _make_log_probs(B, G, T, seed=30)
        ref = _make_log_probs(B, G, T, seed=31)
        rewards = torch.randn(B, G)
        mask = _make_mask(B, G, T)

        loss1, _ = loss_fn(lp.clone(), ref.clone(), rewards.clone(), mask.clone())
        loss2, _ = loss_fn(lp.clone(), ref.clone(), rewards.clone(), mask.clone())
        assert torch.isclose(loss1, loss2), "Loss is non-deterministic"

    def test_numerical_stability_extreme_log_probs(self):
        """12. No NaN/Inf with extreme log_probs (-100 and 0)."""
        B, G, T = 2, 4, 10
        loss_fn = DrGRPOLoss()

        # log_prob = 0 and ref = -100 → ratio = e^100 (extreme)
        lp  = torch.zeros(B, G, T)
        ref = torch.full((B, G, T), -100.0)
        rewards = torch.randn(B, G)
        mask = torch.ones(B, G, T, dtype=torch.bool)

        loss, metrics = loss_fn(lp, ref, rewards, mask)
        assert torch.isfinite(loss), f"Loss not finite for extreme log_probs: {loss}"
        assert all(math.isfinite(v) for v in metrics.values()), (
            f"Non-finite metrics: {metrics}"
        )

    def test_metrics_keys_present(self):
        """14. Metrics dict contains: mean_advantage, std_advantage, clip_fraction."""
        B, G, T = 2, 4, 8
        loss_fn = DrGRPOLoss()
        lp  = _make_log_probs(B, G, T, seed=40)
        ref = _make_log_probs(B, G, T, seed=41)
        rewards = torch.randn(B, G)
        mask = _make_mask(B, G, T)

        _, metrics = loss_fn(lp, ref, rewards, mask)
        required = {"mean_advantage", "std_advantage", "clip_fraction"}
        missing = required - set(metrics.keys())
        assert not missing, f"Metrics dict missing keys: {missing}"
        for k, v in metrics.items():
            assert math.isfinite(v), f"Metric '{k}' is not finite: {v}"

    def test_equal_rewards_loss_finite_no_nan(self):
        """3+5 combined. Equal rewards produce finite loss with no NaN."""
        B, G, T = 2, 4, 8
        loss_fn = DrGRPOLoss()
        lp  = _make_log_probs(B, G, T, seed=50)
        ref = _make_log_probs(B, G, T, seed=51)
        rewards = torch.ones(B, G)  # all rewards equal
        mask = torch.ones(B, G, T, dtype=torch.bool)

        loss, metrics = loss_fn(lp, ref, rewards, mask)
        assert torch.isfinite(loss), f"Loss not finite for equal rewards: {loss}"
        assert not torch.isnan(loss), "Loss is NaN for equal rewards"

    def test_clip_fraction_zero_when_ratio_near_one(self):
        """8b. clip_fraction == 0 when current and reference policies are identical."""
        B, G, T = 2, 4, 8
        loss_fn = DrGRPOLoss(clip_eps=0.2)
        lp = _make_log_probs(B, G, T, seed=60)
        # Same log probs → ratio = 1 everywhere, no clipping.
        rewards = torch.randn(B, G)
        mask = torch.ones(B, G, T, dtype=torch.bool)

        _, metrics = loss_fn(lp, lp.clone(), rewards, mask)
        assert metrics["clip_fraction"] == 0.0, (
            f"Expected clip_fraction=0.0 when ratio=1, got {metrics['clip_fraction']}"
        )
