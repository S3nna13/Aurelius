"""Tests for src/alignment/bco.py — BCO-0 (Jung et al., arXiv:2404.04656).

Pure native PyTorch; no scipy, sklearn, HuggingFace, trl, etc.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.alignment.bco import BCOLoss, BCOTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logps(
    batch_size: int = 4,
    w_offset: float = 0.0,
    l_offset: float = 0.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random log-prob tensors for testing.

    Returns (lp_w, lp_l, lp_ref_w, lp_ref_l) all shape (batch_size,).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    base = torch.randn(batch_size, generator=g) - 5.0  # realistic log-prob range
    lp_w = base + w_offset
    lp_l = base + l_offset
    lp_ref_w = base.clone()
    lp_ref_l = base.clone()
    return lp_w, lp_l, lp_ref_w, lp_ref_l


# ---------------------------------------------------------------------------
# Test 1: Loss is a scalar
# ---------------------------------------------------------------------------


def test_loss_is_scalar():
    loss_fn = BCOLoss(beta=0.1)
    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps()
    loss, _ = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Test 2: Gradients are finite
# ---------------------------------------------------------------------------


def test_gradients_finite():
    loss_fn = BCOLoss(beta=0.1)
    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps()
    lp_w = lp_w.requires_grad_(True)
    lp_l = lp_l.requires_grad_(True)
    loss, _ = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    loss.backward()
    assert lp_w.grad is not None
    assert lp_l.grad is not None
    assert torch.isfinite(lp_w.grad).all(), "lp_w gradient contains non-finite values"
    assert torch.isfinite(lp_l.grad).all(), "lp_l gradient contains non-finite values"


# ---------------------------------------------------------------------------
# Test 3: Determinism
# ---------------------------------------------------------------------------


def test_determinism():
    loss_fn = BCOLoss(beta=0.1)
    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps(seed=7)
    loss1, m1 = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    loss2, m2 = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    assert torch.equal(loss1, loss2), "Loss not deterministic"
    assert m1 == m2, "Metrics not deterministic"


# ---------------------------------------------------------------------------
# Test 4: batch_size=1 works
# ---------------------------------------------------------------------------


def test_batch_size_one():
    loss_fn = BCOLoss(beta=0.1)
    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps(batch_size=1)
    loss, metrics = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    assert loss.shape == ()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 5: Large margin → small loss
# ---------------------------------------------------------------------------


def test_large_margin_small_loss():
    """When y_w >> y_l (large preference margin), loss should be near 0."""
    loss_fn = BCOLoss(beta=0.1)
    B = 8
    lp_w = torch.zeros(B)
    lp_l = torch.full((B,), -50.0)  # policy strongly prefers w
    lp_ref_w = torch.zeros(B)
    lp_ref_l = torch.zeros(B)
    loss, _ = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    assert loss.item() < 0.05, f"Expected small loss for large margin, got {loss.item():.4f}"


# ---------------------------------------------------------------------------
# Test 6: Zero margin → loss ≈ 2·log(2)
# ---------------------------------------------------------------------------


def test_zero_margin_loss_equals_2log2():
    """When diff=0, σ(0)=0.5, loss = -2·log(0.5) = 2·log(2) ≈ 1.3863."""
    loss_fn = BCOLoss(beta=0.1)
    B = 16
    zeros = torch.zeros(B)
    loss, _ = loss_fn(zeros, zeros, zeros, zeros)
    expected = 2.0 * math.log(2.0)
    assert abs(loss.item() - expected) < 1e-5, (
        f"Expected {expected:.6f} at zero margin, got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7: classifier_score ∈ (0, 1)
# ---------------------------------------------------------------------------


def test_classifier_score_in_unit_interval():
    loss_fn = BCOLoss(beta=0.1)
    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps()
    _, metrics = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    cs = metrics["classifier_score"]
    assert 0.0 < cs < 1.0, f"classifier_score={cs} out of (0, 1)"


# ---------------------------------------------------------------------------
# Test 8: classifier_score > 0.5 when y_w is clearly preferred
# ---------------------------------------------------------------------------


def test_classifier_score_above_half_when_preferred():
    """When the policy assigns clearly higher log-probs to y_w, score > 0.5."""
    loss_fn = BCOLoss(beta=0.1)
    B = 8
    lp_w = torch.zeros(B)
    lp_l = torch.full((B,), -10.0)
    lp_ref_w = torch.zeros(B)
    lp_ref_l = torch.zeros(B)
    _, metrics = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    assert metrics["classifier_score"] > 0.5, (
        f"Expected classifier_score > 0.5, got {metrics['classifier_score']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 9: BCO loss ≈ 2× DPO loss for same β
# ---------------------------------------------------------------------------


def test_bco_loss_approx_twice_dpo_loss():
    """BCO-0 loss = -2·log σ(diff/β) = 2 × DPO loss (which = -log σ(diff/β) / β·β)."""
    beta = 0.1
    loss_fn = BCOLoss(beta=beta)
    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps(batch_size=32, seed=99)

    bco_loss, _ = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)

    # DPO loss: -log σ(β * diff)   but BCO uses diff/β as the scaled margin
    # Standard DPO: margin = β * ((lp_w - lp_ref_w) - (lp_l - lp_ref_l))
    # BCO-0:        scaled = diff / β  where diff = (lp_w - lp_ref_w) - (lp_l - lp_ref_l)
    # So BCO scaled = DPO margin / β²  — they share the SAME diff argument only when β=1.
    #
    # The 2× relationship holds when comparing at the SAME scaled argument:
    #   BCO  = -2 · log σ(s)
    #   DPO  = -1 · log σ(s)   (where s = diff/β for BCO, or β·diff for DPO)
    #
    # Compute DPO loss at the same scaled values for comparison:
    diff = (lp_w - lp_ref_w) - (lp_l - lp_ref_l)
    scaled = diff / beta
    dpo_equiv = -F.logsigmoid(scaled).mean()

    ratio = bco_loss.item() / dpo_equiv.item()
    assert abs(ratio - 2.0) < 1e-4, f"Expected BCO ≈ 2×DPO (ratio=2.0), got ratio={ratio:.6f}"


# ---------------------------------------------------------------------------
# Test 10: No NaN/Inf on extreme log-probs (±100)
# ---------------------------------------------------------------------------


def test_no_nan_inf_extreme_logprobs():
    loss_fn = BCOLoss(beta=0.1)
    B = 4
    lp_w = torch.full((B,), -100.0)
    lp_l = torch.full((B,), -100.0)
    lp_ref_w = torch.full((B,), 100.0)
    lp_ref_l = torch.full((B,), 100.0)
    loss, metrics = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    assert torch.isfinite(loss), f"loss={loss.item()} is not finite on extreme inputs"
    for k, v in metrics.items():
        assert math.isfinite(v), f"metric {k}={v} is not finite on extreme inputs"


# ---------------------------------------------------------------------------
# Test 11: No NaN/Inf on zero inputs
# ---------------------------------------------------------------------------


def test_no_nan_inf_zero_inputs():
    loss_fn = BCOLoss(beta=0.1)
    B = 4
    zeros = torch.zeros(B)
    loss, metrics = loss_fn(zeros, zeros, zeros, zeros)
    assert torch.isfinite(loss)
    for k, v in metrics.items():
        assert math.isfinite(v), f"metric {k}={v} is not finite on zero inputs"


# ---------------------------------------------------------------------------
# Test 12: beta=0 raises ValueError (no silent division by zero)
# ---------------------------------------------------------------------------


def test_beta_zero_raises():
    with pytest.raises(ValueError, match="beta > 0"):
        BCOLoss(beta=0.0)


def test_beta_negative_raises():
    with pytest.raises(ValueError, match="beta > 0"):
        BCOLoss(beta=-0.5)


# ---------------------------------------------------------------------------
# Test 13: reward_margin in metrics is the mean diff value
# ---------------------------------------------------------------------------


def test_reward_margin_is_mean_diff():
    beta = 0.1
    loss_fn = BCOLoss(beta=beta)
    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps(batch_size=8, seed=13)
    _, metrics = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)

    expected_diff = ((lp_w - lp_ref_w) - (lp_l - lp_ref_l)).mean().item()
    assert abs(metrics["reward_margin"] - expected_diff) < 1e-5, (
        f"reward_margin={metrics['reward_margin']:.6f} != mean diff={expected_diff:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 14: accuracy = fraction of correct rankings (diff > 0)
# ---------------------------------------------------------------------------


def test_accuracy_correct_ranking():
    beta = 0.1
    loss_fn = BCOLoss(beta=beta)
    B = 10
    # Craft inputs where exactly 7/10 have diff > 0
    # diff = (lp_w - lp_ref_w) - (lp_l - lp_ref_l)
    # Set lp_ref_w = lp_ref_l = 0 for simplicity, then diff = lp_w - lp_l
    lp_w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    lp_l = torch.zeros(B)
    lp_ref_w = torch.zeros(B)
    lp_ref_l = torch.zeros(B)
    _, metrics = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)
    assert abs(metrics["accuracy"] - 0.7) < 1e-5, (
        f"Expected accuracy=0.7, got {metrics['accuracy']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 15: BCOTrainer.compute_loss matches BCOLoss directly
# ---------------------------------------------------------------------------


def test_bco_trainer_compute_loss_matches_bco_loss():
    beta = 0.2
    trainer = BCOTrainer(beta=beta)
    loss_fn = BCOLoss(beta=beta)

    lp_w, lp_l, lp_ref_w, lp_ref_l = _make_logps(batch_size=6, seed=55)

    batch = {
        "lp_w": lp_w,
        "lp_l": lp_l,
        "lp_ref_w": lp_ref_w,
        "lp_ref_l": lp_ref_l,
    }

    trainer_loss, trainer_metrics = trainer.compute_loss(batch)
    direct_loss, direct_metrics = loss_fn(lp_w, lp_l, lp_ref_w, lp_ref_l)

    assert torch.allclose(trainer_loss, direct_loss), (
        f"BCOTrainer loss {trainer_loss.item()} != BCOLoss {direct_loss.item()}"
    )
    for k in direct_metrics:
        assert abs(trainer_metrics[k] - direct_metrics[k]) < 1e-6, (
            f"metric {k}: trainer={trainer_metrics[k]}, direct={direct_metrics[k]}"
        )


# ---------------------------------------------------------------------------
# Test 16: BCOTrainer raises KeyError for missing batch key
# ---------------------------------------------------------------------------


def test_bco_trainer_missing_key_raises():
    trainer = BCOTrainer(beta=0.1)
    incomplete_batch = {
        "lp_w": torch.zeros(4),
        "lp_l": torch.zeros(4),
        # Missing lp_ref_w, lp_ref_l
    }
    with pytest.raises(KeyError, match="lp_ref_w"):
        trainer.compute_loss(incomplete_batch)
