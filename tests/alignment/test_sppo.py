"""Tests for src/alignment/sppo.py — Self-Play Preference Optimization.

Covers all 15 required test cases from the SPPO implementation spec.
Pure native PyTorch; no scipy, sklearn, HuggingFace, trl, peft, etc.
"""

from __future__ import annotations

import math

import torch

from src.alignment.sppo import SPPOLoss, SPPOTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 4,
    *,
    lp_w: float = -1.0,
    lp_l: float = -3.0,
    ref_lp_w: float = -2.0,
    ref_lp_l: float = -2.0,
    requires_grad: bool = False,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a synthetic preference batch."""

    def _t(v: float) -> torch.Tensor:
        return torch.full((B,), v, dtype=torch.float32, device=device, requires_grad=requires_grad)

    return {
        "log_probs_w": _t(lp_w),
        "log_probs_l": _t(lp_l),
        "ref_log_probs_w": _t(ref_lp_w),
        "ref_log_probs_l": _t(ref_lp_l),
    }


def _default_inputs(B: int = 4, device: str = "cpu"):
    """Return (log_probs_w, log_probs_l, ref_log_probs_w, ref_log_probs_l)."""
    batch = _make_batch(B=B, device=device)
    return (
        batch["log_probs_w"],
        batch["log_probs_l"],
        batch["ref_log_probs_w"],
        batch["ref_log_probs_l"],
    )


# ---------------------------------------------------------------------------
# Test 1 — Loss is a scalar
# ---------------------------------------------------------------------------


def test_loss_is_scalar():
    criterion = SPPOLoss(T=0.1)
    loss, _ = criterion(*_default_inputs(B=4))
    assert loss.shape == torch.Size([]), "Loss must be a scalar (0-d tensor)"


# ---------------------------------------------------------------------------
# Test 2 — Gradients are finite (backward works)
# ---------------------------------------------------------------------------


def test_gradients_are_finite():
    lp_w = torch.tensor([-1.0, -2.0], requires_grad=True)
    lp_l = torch.tensor([-3.0, -4.0], requires_grad=True)
    ref_w = torch.tensor([-2.0, -2.0])
    ref_l = torch.tensor([-2.0, -2.0])

    criterion = SPPOLoss(T=0.1)
    loss, _ = criterion(lp_w, lp_l, ref_w, ref_l)
    loss.backward()

    assert lp_w.grad is not None, "Gradient for log_probs_w is None"
    assert lp_l.grad is not None, "Gradient for log_probs_l is None"
    assert torch.isfinite(lp_w.grad).all(), "log_probs_w gradient is non-finite"
    assert torch.isfinite(lp_l.grad).all(), "log_probs_l gradient is non-finite"


# ---------------------------------------------------------------------------
# Test 3 — Determinism
# ---------------------------------------------------------------------------


def test_determinism():
    criterion = SPPOLoss(T=0.1)
    inputs = _default_inputs(B=8)
    loss1, _ = criterion(*inputs)
    loss2, _ = criterion(*inputs)
    assert torch.equal(loss1, loss2), "Loss is not deterministic"


# ---------------------------------------------------------------------------
# Test 4 — batch_size = 1
# ---------------------------------------------------------------------------


def test_batch_size_one():
    criterion = SPPOLoss(T=0.1)
    loss, metrics = criterion(*_default_inputs(B=1))
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 5 — Large reward margin → low loss
# ---------------------------------------------------------------------------


def test_large_reward_margin_low_loss():
    """When y_w is clearly better (large diff), loss should be near 0."""
    criterion = SPPOLoss(T=0.1)
    # diff = (lp_w - ref_w) - (lp_l - ref_l) = 0 - (-100) = 100
    lp_w = torch.zeros(4)
    lp_l = torch.full((4,), -100.0)
    ref_w = torch.zeros(4)
    ref_l = torch.zeros(4)
    loss, _ = criterion(lp_w, lp_l, ref_w, ref_l)
    assert loss.item() < 0.01, f"Expected loss near 0, got {loss.item():.6f}"


# ---------------------------------------------------------------------------
# Test 6 — Tied preferences → loss ≈ log(2)
# ---------------------------------------------------------------------------


def test_tied_preferences_loss_equals_log2():
    """When y_w = y_l (diff = 0), p_hat = 0.5 → loss = log(2)."""
    criterion = SPPOLoss(T=0.1)
    zeros = torch.zeros(8)
    loss, _ = criterion(zeros, zeros, zeros, zeros)
    expected = math.log(2)
    assert abs(loss.item() - expected) < 1e-5, (
        f"Expected loss ≈ log(2) = {expected:.6f}, got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — T → 0: loss ≈ 0 for correct rankings
# ---------------------------------------------------------------------------


def test_very_small_T_near_zero_loss():
    """With T → 0 and positive margin, loss → 0 (deterministic preference)."""
    criterion = SPPOLoss(T=1e-6)
    # diff = 1.0 (y_w is better)
    lp_w = torch.ones(4)
    lp_l = torch.zeros(4)
    ref_w = torch.zeros(4)
    ref_l = torch.zeros(4)
    loss, _ = criterion(lp_w, lp_l, ref_w, ref_l)
    assert loss.item() < 1e-4, f"With T→0 and correct ranking, loss should be ~0, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 8 — Higher T → softer preferences (higher loss for same margin)
# ---------------------------------------------------------------------------


def test_higher_T_means_higher_loss():
    """For a fixed positive reward margin, higher T yields higher loss."""
    lp_w = torch.ones(4)
    lp_l = torch.zeros(4)
    ref_w = torch.zeros(4)
    ref_l = torch.zeros(4)

    loss_low_T, _ = SPPOLoss(T=0.01)(lp_w, lp_l, ref_w, ref_l)
    loss_high_T, _ = SPPOLoss(T=1.0)(lp_w, lp_l, ref_w, ref_l)

    assert loss_low_T.item() < loss_high_T.item(), (
        f"Expected loss(T=0.01) < loss(T=1.0), got "
        f"{loss_low_T.item():.6f} vs {loss_high_T.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 9 — Win probability in [0, 1]
# ---------------------------------------------------------------------------


def test_win_prob_in_unit_interval():
    criterion = SPPOLoss(T=0.1)
    for lp_w_val, lp_l_val in [(-1.0, -3.0), (-3.0, -1.0), (0.0, 0.0)]:
        lp_w = torch.full((4,), lp_w_val)
        lp_l = torch.full((4,), lp_l_val)
        ref = torch.zeros(4)
        _, metrics = criterion(lp_w, lp_l, ref, ref)
        wp = metrics["win_prob"].item()
        assert 0.0 <= wp <= 1.0, f"win_prob={wp} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 10 — Win probability > 0.5 when y_w is better
# ---------------------------------------------------------------------------


def test_win_prob_gt_half_when_yw_better():
    criterion = SPPOLoss(T=0.1)
    # diff > 0 → p_hat > 0.5
    lp_w = torch.zeros(4)  # reward = 0 - 0 = 0
    lp_l = torch.full((4,), -2.0)  # reward = -2 - 0 = -2
    ref = torch.zeros(4)
    _, metrics = criterion(lp_w, lp_l, ref, ref)
    assert metrics["win_prob"].item() > 0.5, (
        f"win_prob should be > 0.5 when y_w is better, got {metrics['win_prob'].item()}"
    )


# ---------------------------------------------------------------------------
# Test 11 — No NaN/Inf on extreme log_probs (±100)
# ---------------------------------------------------------------------------


def test_no_nan_inf_on_extreme_log_probs():
    criterion = SPPOLoss(T=0.1)
    for val in [100.0, -100.0]:
        lp_w = torch.full((4,), val)
        lp_l = torch.full((4,), -val)
        ref = torch.zeros(4)
        loss, metrics = criterion(lp_w, lp_l, ref, ref)
        assert torch.isfinite(loss), f"Loss is non-finite for val={val}"
        assert torch.isfinite(metrics["win_prob"]), "win_prob is non-finite"
        assert torch.isfinite(metrics["reward_margin"]), "reward_margin is non-finite"


# ---------------------------------------------------------------------------
# Test 12 — No NaN/Inf on equal log_probs (all zeros)
# ---------------------------------------------------------------------------


def test_no_nan_inf_on_zero_log_probs():
    criterion = SPPOLoss(T=0.1)
    zeros = torch.zeros(4)
    loss, metrics = criterion(zeros, zeros, zeros, zeros)
    assert torch.isfinite(loss), "Loss is non-finite when all log_probs are 0"
    assert torch.isfinite(metrics["win_prob"]), "win_prob non-finite on zeros"
    assert torch.isfinite(metrics["reward_margin"]), "reward_margin non-finite on zeros"


# ---------------------------------------------------------------------------
# Test 13 — SPPOTrainer.compute_loss matches SPPOLoss directly
# ---------------------------------------------------------------------------


def test_trainer_compute_loss_matches_sppoloss():
    T = 0.1
    trainer = SPPOTrainer(T=T)
    criterion = SPPOLoss(T=T)

    batch = _make_batch(B=6)
    trainer_loss = trainer.compute_loss(batch, T=T)
    direct_loss, _ = criterion(
        batch["log_probs_w"],
        batch["log_probs_l"],
        batch["ref_log_probs_w"],
        batch["ref_log_probs_l"],
        T=T,
    )
    assert torch.allclose(trainer_loss, direct_loss), (
        f"Trainer loss {trainer_loss.item()} != direct SPPOLoss {direct_loss.item()}"
    )


# ---------------------------------------------------------------------------
# Test 14 — metrics_dict contains 'win_prob' and 'reward_margin'
# ---------------------------------------------------------------------------


def test_metrics_dict_keys():
    criterion = SPPOLoss(T=0.1)
    _, metrics = criterion(*_default_inputs(B=4))
    assert "win_prob" in metrics, "metrics missing 'win_prob'"
    assert "reward_margin" in metrics, "metrics missing 'reward_margin'"


# ---------------------------------------------------------------------------
# Test 15 — Gradient flows through both log_probs_w and log_probs_l
# ---------------------------------------------------------------------------


def test_gradient_flows_through_both_inputs():
    lp_w = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    lp_l = torch.tensor([-4.0, -5.0, -6.0], requires_grad=True)
    ref_w = torch.zeros(3)
    ref_l = torch.zeros(3)

    criterion = SPPOLoss(T=0.1)
    loss, _ = criterion(lp_w, lp_l, ref_w, ref_l)
    loss.backward()

    assert lp_w.grad is not None and (lp_w.grad.abs() > 0).any(), (
        "No gradient flows through log_probs_w"
    )
    assert lp_l.grad is not None and (lp_l.grad.abs() > 0).any(), (
        "No gradient flows through log_probs_l"
    )
    # Gradient signs: ∂L/∂lp_w < 0 (increasing lp_w reduces loss)
    #                 ∂L/∂lp_l > 0 (increasing lp_l increases loss)
    assert (lp_w.grad < 0).all(), "Expected negative gradient for log_probs_w"
    assert (lp_l.grad > 0).all(), "Expected positive gradient for log_probs_l"
