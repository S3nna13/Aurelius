"""Tests for src/alignment/rlcd.py — RLCD (arXiv:2307.15217).

Pure native PyTorch only.  No HuggingFace / einops / trl / etc.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.rlcd import RLCDConfig, RLCDLoss, RLCDPairGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log_probs(B: int, T: int, value: float = -1.0) -> torch.Tensor:
    """Return a (B, T) tensor filled with *value*."""
    return torch.full((B, T), value)


def _all_ones_mask(B: int, T: int) -> torch.Tensor:
    return torch.ones(B, T)


# ---------------------------------------------------------------------------
# RLCDConfig
# ---------------------------------------------------------------------------


class TestRLCDConfig:
    def test_defaults(self):
        cfg = RLCDConfig()
        assert cfg.beta == pytest.approx(0.1)
        assert "helpful" in cfg.positive_prompt.lower()
        assert "harmful" in cfg.negative_prompt.lower()

    def test_custom(self):
        cfg = RLCDConfig(beta=0.5, positive_prompt="good", negative_prompt="bad")
        assert cfg.beta == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RLCDLoss — shape & basics
# ---------------------------------------------------------------------------


class TestRLCDLossShape:
    """Test 1 — loss is a scalar."""

    def test_output_is_scalar(self):
        B, T = 4, 16
        loss_fn = RLCDLoss()
        lp = _make_log_probs(B, T)
        mask = _all_ones_mask(B, T)
        loss, metrics = loss_fn(lp, lp, lp, lp, mask, mask)
        assert loss.shape == torch.Size([]), "loss must be 0-dimensional (scalar)"

    def test_metrics_keys(self):
        """Test 10 — metrics dict contains required keys."""
        B, T = 2, 8
        loss_fn = RLCDLoss()
        lp = _make_log_probs(B, T)
        mask = _all_ones_mask(B, T)
        _, metrics = loss_fn(lp, lp, lp, lp, mask, mask)
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics


# ---------------------------------------------------------------------------
# Test 2 — Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_finite_grads_after_backward(self):
        B, T = 3, 10
        loss_fn = RLCDLoss()
        pos_lp = _make_log_probs(B, T).requires_grad_(True)
        neg_lp = _make_log_probs(B, T).requires_grad_(True)
        ref_lp = _make_log_probs(B, T)
        mask = _all_ones_mask(B, T)

        loss, _ = loss_fn(pos_lp, neg_lp, ref_lp, ref_lp, mask, mask)
        loss.backward()

        assert pos_lp.grad is not None
        assert neg_lp.grad is not None
        assert torch.isfinite(pos_lp.grad).all(), "pos gradient contains non-finite values"
        assert torch.isfinite(neg_lp.grad).all(), "neg gradient contains non-finite values"


# ---------------------------------------------------------------------------
# Test 3 — Loss ≈ log(2) when policy == reference
# ---------------------------------------------------------------------------


class TestLossAtReference:
    def test_loss_approx_log2_when_policy_equals_reference(self):
        """When π_θ = π_ref the log-ratio is 0 → margin = 0 → loss = -log σ(0) = log 2."""
        B, T = 8, 20
        loss_fn = RLCDLoss()
        lp = _make_log_probs(B, T, value=-2.0)
        mask = _all_ones_mask(B, T)

        loss, _ = loss_fn(lp, lp, lp, lp, mask, mask)
        assert loss.item() == pytest.approx(math.log(2), abs=1e-5)


# ---------------------------------------------------------------------------
# Test 4 — Loss decreases when positives outperform negatives
# ---------------------------------------------------------------------------


class TestLossDirection:
    def test_better_positives_give_lower_loss(self):
        """Positive completions closer to reference than negatives → lower loss."""
        B, T = 4, 12
        loss_fn = RLCDLoss()
        ref_lp = _make_log_probs(B, T, value=-1.0)
        mask = _all_ones_mask(B, T)

        # Baseline: policy == reference (loss = log 2)
        base_loss, _ = loss_fn(ref_lp, ref_lp, ref_lp, ref_lp, mask, mask)

        # Policy improves on positives (higher log-prob) and degrades on negatives
        pos_lp_better = _make_log_probs(B, T, value=-0.5)  # higher than ref
        neg_lp_worse = _make_log_probs(B, T, value=-2.0)  # lower than ref

        better_loss, _ = loss_fn(pos_lp_better, neg_lp_worse, ref_lp, ref_lp, mask, mask)
        assert better_loss.item() < base_loss.item(), (
            f"Expected lower loss when pos > ref and neg < ref, "
            f"got {better_loss.item():.4f} vs baseline {base_loss.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Token masking
# ---------------------------------------------------------------------------


class TestTokenMasking:
    def test_masked_tokens_dont_affect_loss(self):
        """Zeroing out extra tokens via mask should not change the loss."""
        B, T = 2, 10
        loss_fn = RLCDLoss()

        pos_lp = torch.randn(B, T)
        neg_lp = torch.randn(B, T)
        ref_lp = torch.randn(B, T)

        # First 5 tokens are valid
        mask_short = torch.zeros(B, T)
        mask_short[:, :5] = 1.0

        # All tokens valid, but the last 5 have the same values → same mean
        # Replicate: use identical values for tokens 5-9 as tokens 0-4
        pos_lp_full = pos_lp.clone()
        pos_lp_full[:, 5:] = pos_lp[:, :5]
        neg_lp_full = neg_lp.clone()
        neg_lp_full[:, 5:] = neg_lp[:, :5]
        ref_lp_full = ref_lp.clone()
        ref_lp_full[:, 5:] = ref_lp[:, :5]
        mask_full = torch.ones(B, T)

        loss_short, _ = loss_fn(pos_lp, neg_lp, ref_lp, ref_lp, mask_short, mask_short)
        loss_full, _ = loss_fn(
            pos_lp_full, neg_lp_full, ref_lp_full, ref_lp_full, mask_full, mask_full
        )

        assert loss_short.item() == pytest.approx(loss_full.item(), abs=1e-5), (
            "Masking extra (identical) tokens changed the loss"
        )


# ---------------------------------------------------------------------------
# Tests 6 & 7 — RLCDPairGenerator.prepare_inputs
# ---------------------------------------------------------------------------


class TestPairGeneratorPrepareInputs:
    def test_prepend_order(self):
        """Test 6 — output starts with prompt tokens then instruction tokens."""
        gen = RLCDPairGenerator()
        B, T_inst, T_p = 2, 6, 3

        inst = torch.arange(T_inst).unsqueeze(0).expand(B, -1)  # (2, 6)
        p_pos = torch.tensor([100, 101, 102])  # (3,)
        p_neg = torch.tensor([200, 201, 202])  # (3,)

        pos_out, neg_out = gen.prepare_inputs(inst, p_pos, p_neg)

        # First T_p tokens of every row should be the prompt
        assert (pos_out[:, :T_p] == p_pos.unsqueeze(0)).all()
        assert (neg_out[:, :T_p] == p_neg.unsqueeze(0)).all()
        # Remaining tokens should be the instruction
        assert (pos_out[:, T_p:] == inst).all()
        assert (neg_out[:, T_p:] == inst).all()

    def test_output_shape(self):
        """Test 7 — output shape is (B, T_p + T_inst)."""
        gen = RLCDPairGenerator()
        B, T_inst, T_p = 3, 8, 4

        inst = torch.zeros(B, T_inst, dtype=torch.long)
        p_pos = torch.zeros(T_p, dtype=torch.long)
        p_neg = torch.zeros(T_p, dtype=torch.long)

        pos_out, neg_out = gen.prepare_inputs(inst, p_pos, p_neg)

        assert pos_out.shape == (B, T_p + T_inst)
        assert neg_out.shape == (B, T_p + T_inst)


# ---------------------------------------------------------------------------
# Test 8 — Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_loss(self):
        B, T = 4, 16
        loss_fn = RLCDLoss()

        def _run(seed: int) -> float:
            torch.manual_seed(seed)
            pos_lp = torch.randn(B, T)
            neg_lp = torch.randn(B, T)
            ref_lp = torch.randn(B, T)
            mask = _all_ones_mask(B, T)
            loss, _ = loss_fn(pos_lp, neg_lp, ref_lp, ref_lp, mask, mask)
            return loss.item()

        assert _run(42) == pytest.approx(_run(42), abs=0.0)
        assert _run(7) == pytest.approx(_run(7), abs=0.0)


# ---------------------------------------------------------------------------
# Test 9 — Numerical stability (extreme log-probs)
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_no_nan_inf_with_extreme_log_probs(self):
        B, T = 4, 16
        loss_fn = RLCDLoss()
        mask = _all_ones_mask(B, T)

        for val in [-100.0, 0.0, -50.0]:
            lp = _make_log_probs(B, T, value=val)
            loss, metrics = loss_fn(lp, lp, lp, lp, mask, mask)
            assert torch.isfinite(loss), f"loss not finite for log_prob={val}"
            assert all(math.isfinite(v) for v in metrics.values()), (
                f"metrics contain non-finite values for log_prob={val}"
            )


# ---------------------------------------------------------------------------
# Test 11 — Beta scaling
# ---------------------------------------------------------------------------


class TestBetaScaling:
    def test_higher_beta_stronger_kl_penalty(self):
        """Higher β should produce a larger |margin| → further from log(2)."""
        B, T = 4, 12
        mask = _all_ones_mask(B, T)

        pos_lp = _make_log_probs(B, T, value=-0.5)
        neg_lp = _make_log_probs(B, T, value=-2.0)
        ref_lp = _make_log_probs(B, T, value=-1.0)

        loss_small_beta, _ = RLCDLoss(RLCDConfig(beta=0.01)).forward(
            pos_lp, neg_lp, ref_lp, ref_lp, mask, mask
        )
        loss_large_beta, _ = RLCDLoss(RLCDConfig(beta=1.0)).forward(
            pos_lp, neg_lp, ref_lp, ref_lp, mask, mask
        )

        # Larger β means the margin is larger → -log σ(margin) is smaller (more confident)
        assert loss_large_beta.item() < loss_small_beta.item(), (
            "Higher beta should produce lower loss when policy prefers positives over negatives"
        )


# ---------------------------------------------------------------------------
# Test 12 — Sequence length tolerance (different T+ vs T-)
# ---------------------------------------------------------------------------


class TestDifferentSequenceLengths:
    def test_pos_neg_different_lengths(self):
        """pos and neg completions can have different padded lengths via masking."""
        B = 3
        T_pos, T_neg = 12, 8
        loss_fn = RLCDLoss()

        pos_lp = torch.randn(B, T_pos)
        ref_pos_lp = torch.randn(B, T_pos)
        neg_lp = torch.randn(B, T_neg)
        ref_neg_lp = torch.randn(B, T_neg)
        pos_mask = _all_ones_mask(B, T_pos)
        neg_mask = _all_ones_mask(B, T_neg)

        loss, metrics = loss_fn(pos_lp, neg_lp, ref_pos_lp, ref_neg_lp, pos_mask, neg_mask)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 13 — Batch size 1
# ---------------------------------------------------------------------------


class TestBatchSizeOne:
    def test_single_example(self):
        B, T = 1, 16
        loss_fn = RLCDLoss()
        lp = _make_log_probs(B, T, value=-1.5)
        ref = _make_log_probs(B, T, value=-2.0)
        mask = _all_ones_mask(B, T)

        loss, metrics = loss_fn(lp, lp, ref, ref, mask, mask)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)
        assert all(math.isfinite(v) for v in metrics.values())

    def test_prepare_inputs_batch_size_one(self):
        gen = RLCDPairGenerator()
        inst = torch.tensor([[1, 2, 3]])  # (1, 3)
        p_pos = torch.tensor([10, 11])  # (2,)
        p_neg = torch.tensor([20, 21])  # (2,)

        pos_out, neg_out = gen.prepare_inputs(inst, p_pos, p_neg)
        assert pos_out.shape == (1, 5)
        assert neg_out.shape == (1, 5)
