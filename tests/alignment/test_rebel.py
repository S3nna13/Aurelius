"""Tests for REBEL alignment loss (Gao et al., 2024, arXiv:2404.16767).

All tests use pure PyTorch — no scipy, sklearn, HuggingFace, or trl.
"""

from __future__ import annotations

import pytest
import torch

from src.alignment.rebel import REBELLoss, REBELTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(batch_size: int = 4, seed: int = 0):
    """Return a reproducible batch of tensors."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return dict(
        log_probs_w=torch.randn(batch_size, generator=rng),
        log_probs_l=torch.randn(batch_size, generator=rng),
        ref_log_probs_w=torch.randn(batch_size, generator=rng),
        ref_log_probs_l=torch.randn(batch_size, generator=rng),
        reward_w=torch.randn(batch_size, generator=rng) + 1.0,   # winner typically higher
        reward_l=torch.randn(batch_size, generator=rng) - 1.0,
    )


# ---------------------------------------------------------------------------
# 1. Loss is scalar
# ---------------------------------------------------------------------------

def test_loss_is_scalar():
    loss_fn = REBELLoss(beta=0.1)
    b = _make_batch()
    loss, _ = loss_fn(**b)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 2. Gradients are finite
# ---------------------------------------------------------------------------

def test_gradients_finite():
    loss_fn = REBELLoss(beta=0.1)
    b = _make_batch()
    lp_w = b["log_probs_w"].requires_grad_(True)
    lp_l = b["log_probs_l"].requires_grad_(True)
    loss, _ = loss_fn(
        log_probs_w=lp_w,
        log_probs_l=lp_l,
        ref_log_probs_w=b["ref_log_probs_w"],
        ref_log_probs_l=b["ref_log_probs_l"],
        reward_w=b["reward_w"],
        reward_l=b["reward_l"],
    )
    loss.backward()
    assert lp_w.grad is not None
    assert lp_l.grad is not None
    assert torch.all(torch.isfinite(lp_w.grad))
    assert torch.all(torch.isfinite(lp_l.grad))


# ---------------------------------------------------------------------------
# 3. Determinism — same inputs → same loss
# ---------------------------------------------------------------------------

def test_determinism():
    loss_fn = REBELLoss(beta=0.1)
    b = _make_batch(seed=42)
    loss_a, _ = loss_fn(**b)
    loss_b, _ = loss_fn(**b)
    assert loss_a.item() == loss_b.item()


# ---------------------------------------------------------------------------
# 4. batch_size = 1 works
# ---------------------------------------------------------------------------

def test_batch_size_one():
    loss_fn = REBELLoss(beta=0.1)
    b = _make_batch(batch_size=1)
    loss, metrics = loss_fn(**b)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 5. Loss = 0 when Δ = target exactly  (perfect regression)
# ---------------------------------------------------------------------------

def test_loss_zero_when_delta_equals_target():
    """Construct inputs so that Δ_w - Δ_l == (r_w - r_l) / β exactly."""
    beta = 0.5
    # Let Δ_w = 0.8, Δ_l = 0.2  →  Δ = 0.6
    # Set r_w - r_l = Δ * β = 0.3  →  target = 0.6
    ref_w = torch.tensor([0.0])
    ref_l = torch.tensor([0.0])
    lp_w = torch.tensor([0.8])    # Δ_w = 0.8 - 0.0 = 0.8
    lp_l = torch.tensor([0.2])    # Δ_l = 0.2 - 0.0 = 0.2  → Δ = 0.6
    r_w = torch.tensor([0.3])
    r_l = torch.tensor([0.0])     # reward_margin = 0.3  → target = 0.3/0.5 = 0.6

    loss_fn = REBELLoss(beta=beta)
    loss, _ = loss_fn(lp_w, lp_l, ref_w, ref_l, r_w, r_l)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Loss increases as Δ deviates from target
# ---------------------------------------------------------------------------

def test_loss_increases_with_deviation():
    beta = 0.1
    ref = torch.zeros(1)
    r_w = torch.tensor([1.0])
    r_l = torch.tensor([0.0])   # target = 10.0

    loss_fn = REBELLoss(beta=beta)

    # Δ = 0  (large deviation from target=10)
    loss_far, _ = loss_fn(torch.tensor([0.0]), torch.tensor([0.0]), ref, ref, r_w, r_l)
    # Δ = 9  (smaller deviation)
    loss_near, _ = loss_fn(torch.tensor([9.0]), torch.tensor([0.0]), ref, ref, r_w, r_l)

    assert loss_far.item() > loss_near.item()


# ---------------------------------------------------------------------------
# 7. Positive reward margin → positive target
# ---------------------------------------------------------------------------

def test_positive_reward_margin_gives_positive_target():
    beta = 0.1
    loss_fn = REBELLoss(beta=beta)
    r_w = torch.tensor([2.0])
    r_l = torch.tensor([1.0])  # margin = 1.0 > 0 → target = 10.0
    ref = torch.zeros(1)
    _, metrics = loss_fn(ref, ref, ref, ref, r_w, r_l)
    assert metrics["target"].item() > 0.0


# ---------------------------------------------------------------------------
# 8. Negative reward margin → negative target
# ---------------------------------------------------------------------------

def test_negative_reward_margin_gives_negative_target():
    beta = 0.1
    loss_fn = REBELLoss(beta=beta)
    r_w = torch.tensor([0.5])
    r_l = torch.tensor([2.0])  # margin = -1.5 → target < 0
    ref = torch.zeros(1)
    _, metrics = loss_fn(ref, ref, ref, ref, r_w, r_l)
    assert metrics["target"].item() < 0.0


# ---------------------------------------------------------------------------
# 9. lambda_reg=0 — no regularisation (pure MSE)
# ---------------------------------------------------------------------------

def test_lambda_reg_zero_means_no_regularisation():
    """With λ=0, large Δ_w / Δ_l should not inflate loss beyond MSE."""
    beta = 0.1
    # Construct a situation where only the regression term matters:
    # Δ_w = 100, Δ_l = 0  →  Δ = 100; target = 100/0.1 = 1000 → regression huge
    # But compare two loss_fns: one with λ=0, one with λ=1 → same regression term
    lp_w = torch.tensor([100.0])
    lp_l = torch.tensor([0.0])
    ref = torch.zeros(1)
    r_w = torch.tensor([100.0])
    r_l = torch.tensor([0.0])

    loss_no_reg = REBELLoss(beta=beta, lambda_reg=0.0)
    loss_with_reg = REBELLoss(beta=beta, lambda_reg=1.0)

    val_no_reg, _ = loss_no_reg(lp_w, lp_l, ref, ref, r_w, r_l)
    val_with_reg, _ = loss_with_reg(lp_w, lp_l, ref, ref, r_w, r_l)

    # With reg, loss must be strictly larger
    assert val_with_reg.item() > val_no_reg.item()


# ---------------------------------------------------------------------------
# 10. lambda_reg>0 increases loss when Δ_w, Δ_l are large
# ---------------------------------------------------------------------------

def test_lambda_reg_increases_loss_for_large_deltas():
    """Same regression error but bigger Δ_w/Δ_l → larger regularised loss."""
    beta = 1.0
    # Both setups have Δ = target = 0 (zero regression error).
    # Setup A: ref = log_probs → Δ_w = Δ_l = 0  (small deviations)
    # Setup B: log_probs and ref both large but equal magnitude → same Δ,
    #          but Δ_w and Δ_l individually are large.
    lp_w_a = torch.tensor([0.5])
    lp_l_a = torch.tensor([0.5])    # Δ_w=0, Δ_l=0, Δ=0
    ref_w_a = torch.tensor([0.5])
    ref_l_a = torch.tensor([0.5])
    r_w_a = r_l_a = torch.tensor([0.0])   # target = 0, Δ = 0 → perfect

    lp_w_b = torch.tensor([50.0])
    lp_l_b = torch.tensor([50.0])   # Δ_w=40, Δ_l=40, Δ=0
    ref_w_b = torch.tensor([10.0])
    ref_l_b = torch.tensor([10.0])
    r_w_b = r_l_b = torch.tensor([0.0])   # target = 0, Δ = 0 → perfect

    loss_fn = REBELLoss(beta=beta, lambda_reg=0.1)

    loss_a, _ = loss_fn(lp_w_a, lp_l_a, ref_w_a, ref_l_a, r_w_a, r_l_a)
    loss_b, _ = loss_fn(lp_w_b, lp_l_b, ref_w_b, ref_l_b, r_w_b, r_l_b)

    assert loss_b.item() > loss_a.item(), (
        f"Regularised loss should be larger when Δ_w/Δ_l are large, "
        f"got loss_a={loss_a.item():.4f} loss_b={loss_b.item():.4f}"
    )


# ---------------------------------------------------------------------------
# 11. No NaN/Inf on extreme log_probs (±100)
# ---------------------------------------------------------------------------

def test_no_nan_inf_on_extreme_log_probs():
    loss_fn = REBELLoss(beta=0.1)
    extremes = torch.tensor([100.0, -100.0, 50.0, -50.0])
    loss, metrics = loss_fn(
        log_probs_w=extremes,
        log_probs_l=-extremes,
        ref_log_probs_w=torch.zeros(4),
        ref_log_probs_l=torch.zeros(4),
        reward_w=torch.ones(4),
        reward_l=-torch.ones(4),
    )
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"
    for k, v in metrics.items():
        assert torch.all(torch.isfinite(v)), f"Non-finite metric '{k}': {v}"


# ---------------------------------------------------------------------------
# 12. No NaN/Inf when rewards are equal (target = 0)
# ---------------------------------------------------------------------------

def test_no_nan_inf_equal_rewards():
    loss_fn = REBELLoss(beta=0.1)
    b = _make_batch()
    b["reward_w"] = torch.zeros(4)
    b["reward_l"] = torch.zeros(4)
    loss, metrics = loss_fn(**b)
    assert torch.isfinite(loss)
    assert torch.all(metrics["target"] == 0.0)


# ---------------------------------------------------------------------------
# 13. MSE symmetry: equal error in + and - direction gives equal loss
# ---------------------------------------------------------------------------

def test_mse_symmetry_around_target():
    """Loss must be symmetric: (Δ - target)² = (target - Δ)² ."""
    beta = 0.1
    loss_fn = REBELLoss(beta=beta)
    ref = torch.zeros(1)
    r_w = torch.tensor([1.0])
    r_l = torch.tensor([0.0])   # target = 10.0

    # Δ = 10 + ε  (overshoot)
    eps = 3.0
    loss_over, _ = loss_fn(torch.tensor([10.0 + eps]), torch.tensor([0.0]), ref, ref, r_w, r_l)
    # Δ = 10 - ε  (undershoot)
    loss_under, _ = loss_fn(torch.tensor([10.0 - eps]), torch.tensor([0.0]), ref, ref, r_w, r_l)

    assert loss_over.item() == pytest.approx(loss_under.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# 14. REBELTrainer.compute_loss matches REBELLoss directly
# ---------------------------------------------------------------------------

def test_trainer_compute_loss_matches_rebel_loss():
    beta = 0.2
    lambda_reg = 0.05
    b = _make_batch(seed=7)

    loss_fn = REBELLoss(beta=beta, lambda_reg=lambda_reg)
    trainer = REBELTrainer(beta=beta, lambda_reg=lambda_reg)

    expected_loss, _ = loss_fn(**b)
    actual_loss = trainer.compute_loss(b)

    assert actual_loss.item() == pytest.approx(expected_loss.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# 15. reward_margin in metrics == r_w - r_l
# ---------------------------------------------------------------------------

def test_reward_margin_metric_equals_r_w_minus_r_l():
    loss_fn = REBELLoss(beta=0.1)
    r_w = torch.tensor([3.0, 1.5, -0.5])
    r_l = torch.tensor([1.0, 1.0,  0.5])
    ref = torch.zeros(3)
    _, metrics = loss_fn(ref, ref, ref, ref, r_w, r_l)
    expected = r_w - r_l
    assert torch.allclose(metrics["reward_margin"], expected)


# ---------------------------------------------------------------------------
# Extra: invalid beta raises loudly
# ---------------------------------------------------------------------------

def test_invalid_beta_raises():
    with pytest.raises(ValueError, match="beta"):
        REBELLoss(beta=-0.1)


# ---------------------------------------------------------------------------
# Extra: missing batch key raises loudly
# ---------------------------------------------------------------------------

def test_trainer_raises_on_missing_key():
    trainer = REBELTrainer()
    incomplete = {"log_probs_w": torch.tensor([1.0])}
    with pytest.raises(KeyError):
        trainer.compute_loss(incomplete)
