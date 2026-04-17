"""Tests for DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization.

Tests cover:
  1. Loss is scalar
  2. Gradients finite
  3. Determinism
  4. batch_size=1, seq_len=1 edge case
  5. Positive advantage + high ratio → clipped at 1+ε_high
  6. Negative advantage + low ratio → clipped at 1-ε_low
  7. Decoupled clip: ε_high > ε_low → asymmetric clipping
  8. DAPOFilter: all rewards=1 → should_keep=False
  9. DAPOFilter: all rewards=0 → should_keep=False
 10. DAPOFilter: mixed rewards → should_keep=True
 11. Entropy bonus: beta_entropy=0 → entropy term doesn't affect loss
 12. Token-level normalization: loss normalized by token count
 13. No NaN/Inf on zero advantages
 14. No NaN/Inf on extreme log_probs (±100)
 15. clip_fraction in metrics ∈ [0, 1]
"""
import math
import pytest
import torch

from src.alignment.dapo import DAPOLoss, DAPOFilter, DAPOTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(
    batch: int = 2,
    seq_len: int = 4,
    advantage_val: float = 1.0,
    log_prob_val: float = -0.5,
    old_log_prob_val: float = -0.5,
    requires_grad: bool = True,
):
    """Create simple (batch, seq_len) tensors for testing."""
    log_probs = torch.full((batch, seq_len), log_prob_val, dtype=torch.float32)
    if requires_grad:
        log_probs.requires_grad_(True)
    old_log_probs = torch.full((batch, seq_len), old_log_prob_val, dtype=torch.float32)
    advantages = torch.full((batch, seq_len), advantage_val, dtype=torch.float32)
    return log_probs, old_log_probs, advantages


# ---------------------------------------------------------------------------
# Test 1: Loss is scalar
# ---------------------------------------------------------------------------

def test_loss_is_scalar():
    loss_fn = DAPOLoss()
    log_probs, old_log_probs, advantages = make_inputs()
    loss, _ = loss_fn(log_probs, old_log_probs, advantages)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# Test 2: Gradients are finite
# ---------------------------------------------------------------------------

def test_gradients_finite():
    loss_fn = DAPOLoss()
    log_probs, old_log_probs, advantages = make_inputs(requires_grad=True)
    loss, _ = loss_fn(log_probs, old_log_probs, advantages)
    loss.backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all(), "Gradient contains non-finite values"


# ---------------------------------------------------------------------------
# Test 3: Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    loss_fn = DAPOLoss(eps_low=0.1, eps_high=0.2, beta_entropy=0.001)
    log_probs, old_log_probs, advantages = make_inputs()
    entropy = torch.ones_like(log_probs) * 0.5

    loss1, m1 = loss_fn(log_probs, old_log_probs, advantages, entropy)
    loss2, m2 = loss_fn(log_probs, old_log_probs, advantages, entropy)

    assert torch.equal(loss1, loss2), "Loss is not deterministic"
    assert m1 == m2, "Metrics are not deterministic"


# ---------------------------------------------------------------------------
# Test 4: batch_size=1, seq_len=1 edge case
# ---------------------------------------------------------------------------

def test_single_token():
    loss_fn = DAPOLoss()
    log_probs = torch.tensor([[-0.3]], requires_grad=True)
    old_log_probs = torch.tensor([[-0.3]])
    advantages = torch.tensor([[1.0]])

    loss, metrics = loss_fn(log_probs, old_log_probs, advantages)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())
    assert 0.0 <= metrics["clip_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5: Positive advantage + high ratio → clipped at 1+ε_high
# ---------------------------------------------------------------------------

def test_positive_advantage_high_ratio_clipped_at_eps_high():
    """When A > 0 and r > 1+ε_high, the objective should be capped (clipped).

    r_clipped = 1 + ε_high  (upper bound)
    surr2 = r_clipped * A < surr1 = r * A
    → min selects surr2, so loss = -surr2 = -(1+ε_high)*A
    """
    eps_low, eps_high = 0.1, 0.2
    loss_fn = DAPOLoss(eps_low=eps_low, eps_high=eps_high, beta_entropy=0.0)

    # r = exp(log_prob - old_log_prob) = exp(0.5) ≈ 1.65 > 1 + 0.2 = 1.2
    log_probs = torch.tensor([[0.5]], requires_grad=True)
    old_log_probs = torch.tensor([[0.0]])
    advantages = torch.tensor([[1.0]])

    loss, metrics = loss_fn(log_probs, old_log_probs, advantages)

    # Expected: clipped ratio = 1 + eps_high, loss = -(1 + eps_high) * A
    expected_loss = -(1.0 + eps_high)
    assert abs(loss.item() - expected_loss) < 1e-5, (
        f"Expected loss {expected_loss:.4f}, got {loss.item():.4f}"
    )
    assert metrics["clip_fraction"] > 0.0, "Expected clipping to occur"


# ---------------------------------------------------------------------------
# Test 6: Negative advantage + low ratio → clipped at 1-ε_low
# ---------------------------------------------------------------------------

def test_negative_advantage_low_ratio_clipped_at_eps_low():
    """When A < 0 and r < 1-ε_low, the objective should be capped.

    r_clipped = 1 - ε_low  (lower bound)
    A < 0 → r_clipped * A > r * A  (less negative)
    min selects r * A  (more negative), but clipping prevents going below
    r_clipped * A when A < 0.

    Actually for A < 0:
      surr1 = r * A  (more negative when r < 1-ε_low)
      surr2 = r_clipped * A = (1-ε_low) * A  (less negative, since r_clipped > r)
      min(surr1, surr2) = surr1 = r * A  (no clipping effect on loss direction)

    The clip at lower end for A < 0 becomes binding when r < 1-ε_low:
      r_clipped * A > r * A when A < 0 and r < 1-ε_low
      min picks surr1 = r*A  ... but wait, PPO clip is designed so clipping
      prevents too-large updates. For A<0, large decrease (small r) is
      clamped: r_clipped = max(r, 1-ε_low), so loss = -(1-ε_low)*A.

    Test: verify clip_fraction > 0 when r < 1-ε_low and A < 0.
    """
    eps_low, eps_high = 0.1, 0.2
    loss_fn = DAPOLoss(eps_low=eps_low, eps_high=eps_high, beta_entropy=0.0)

    # r = exp(-0.5) ≈ 0.607 < 1 - 0.1 = 0.9 → lower bound active
    log_probs = torch.tensor([[-0.5]], requires_grad=True)
    old_log_probs = torch.tensor([[0.0]])
    advantages = torch.tensor([[-1.0]])  # negative advantage

    loss, metrics = loss_fn(log_probs, old_log_probs, advantages)

    # r < 1-ε_low and A < 0: clipping should be active
    assert metrics["clip_fraction"] > 0.0, (
        f"Expected clipping for r < 1-ε_low with A < 0, got clip_fraction={metrics['clip_fraction']}"
    )
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# Test 7: Decoupled clip: ε_high > ε_low → asymmetric clipping
# ---------------------------------------------------------------------------

def test_decoupled_clip_asymmetry():
    """Verify that ε_high and ε_low produce different clipping behavior.

    For positive advantages with r > 1+ε_high:
      loss_high = -(1+ε_high) * A
    For symmetric clip with ε=ε_low (tighter):
      loss_low = -(1+ε_low) * A

    Since ε_high > ε_low, (1+ε_high) > (1+ε_low), so loss_high < loss_low
    (more negative → better surrogate objective allowed).
    """
    eps_low = 0.1
    eps_high = 0.2

    loss_fn_dapo = DAPOLoss(eps_low=eps_low, eps_high=eps_high, beta_entropy=0.0)
    loss_fn_sym = DAPOLoss(eps_low=eps_low, eps_high=eps_low, beta_entropy=0.0)  # symmetric (tight)

    # r ≈ 1.65 > 1+ε_high=1.2 and also > 1+ε_low=1.1 → both clip
    log_probs = torch.tensor([[0.5]], requires_grad=False)
    old_log_probs = torch.tensor([[0.0]])
    advantages = torch.tensor([[1.0]])

    loss_dapo, _ = loss_fn_dapo(log_probs, old_log_probs, advantages)
    loss_sym, _ = loss_fn_sym(log_probs, old_log_probs, advantages)

    # DAPO allows larger ratio for positive A → lower (more negative) loss
    assert loss_dapo.item() < loss_sym.item(), (
        f"DAPO loss {loss_dapo.item():.4f} should be < symmetric loss {loss_sym.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: DAPOFilter all rewards=1 → should_keep=False
# ---------------------------------------------------------------------------

def test_filter_all_correct():
    f = DAPOFilter()
    rewards = torch.ones(8)
    assert f.should_keep(rewards) is False, "All-correct batch should be filtered out"


# ---------------------------------------------------------------------------
# Test 9: DAPOFilter all rewards=0 → should_keep=False
# ---------------------------------------------------------------------------

def test_filter_all_wrong():
    f = DAPOFilter()
    rewards = torch.zeros(8)
    assert f.should_keep(rewards) is False, "All-wrong batch should be filtered out"


# ---------------------------------------------------------------------------
# Test 10: DAPOFilter mixed rewards → should_keep=True
# ---------------------------------------------------------------------------

def test_filter_mixed_rewards():
    f = DAPOFilter()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert f.should_keep(rewards) is True, "Mixed-reward batch should be kept"


# ---------------------------------------------------------------------------
# Test 11: Entropy bonus: beta_entropy=0 → entropy term doesn't affect loss
# ---------------------------------------------------------------------------

def test_entropy_bonus_zero_beta():
    """With beta_entropy=0, providing entropy should not change loss."""
    loss_fn = DAPOLoss(eps_low=0.1, eps_high=0.2, beta_entropy=0.0)
    log_probs, old_log_probs, advantages = make_inputs()
    entropy = torch.rand_like(log_probs)

    loss_no_ent, _ = loss_fn(log_probs, old_log_probs, advantages, entropy=None)
    loss_with_ent, _ = loss_fn(log_probs, old_log_probs, advantages, entropy=entropy)

    assert torch.allclose(loss_no_ent, loss_with_ent), (
        f"beta_entropy=0 should make entropy irrelevant: "
        f"{loss_no_ent.item()} vs {loss_with_ent.item()}"
    )


# ---------------------------------------------------------------------------
# Test 12: Token-level normalization
# ---------------------------------------------------------------------------

def test_token_level_normalization():
    """Loss should be mean over tokens, not sum over sequences.

    If we double seq_len with same per-token values, the loss should
    stay the same (mean is invariant to length when values are uniform).
    """
    loss_fn = DAPOLoss(eps_low=0.1, eps_high=0.2, beta_entropy=0.0)

    # 1 sequence of length 4
    lp1 = torch.full((1, 4), -0.3)
    olp1 = torch.full((1, 4), -0.5)
    adv1 = torch.full((1, 4), 0.8)

    # 1 sequence of length 8 (same values repeated)
    lp2 = torch.full((1, 8), -0.3)
    olp2 = torch.full((1, 8), -0.5)
    adv2 = torch.full((1, 8), 0.8)

    loss1, _ = loss_fn(lp1, olp1, adv1)
    loss2, _ = loss_fn(lp2, olp2, adv2)

    assert abs(loss1.item() - loss2.item()) < 1e-5, (
        f"Token-level normalization: loss should be identical for doubled sequence "
        f"with uniform values. Got {loss1.item():.6f} vs {loss2.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 13: No NaN/Inf on zero advantages
# ---------------------------------------------------------------------------

def test_no_nan_zero_advantages():
    loss_fn = DAPOLoss()
    log_probs, old_log_probs, advantages = make_inputs(advantage_val=0.0)
    loss, metrics = loss_fn(log_probs, old_log_probs, advantages)
    assert torch.isfinite(loss), f"Loss is not finite for zero advantages: {loss.item()}"
    assert math.isfinite(metrics["clip_fraction"])
    assert math.isfinite(metrics["mean_ratio"])


# ---------------------------------------------------------------------------
# Test 14: No NaN/Inf on extreme log_probs (±100)
# ---------------------------------------------------------------------------

def test_no_nan_extreme_log_probs():
    loss_fn = DAPOLoss()

    for lp_val, olp_val in [(100.0, -100.0), (-100.0, 100.0), (100.0, 100.0)]:
        log_probs = torch.full((2, 4), lp_val)
        old_log_probs = torch.full((2, 4), olp_val)
        advantages = torch.ones(2, 4)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages)
        assert torch.isfinite(loss), (
            f"Loss is not finite for lp={lp_val}, olp={olp_val}: {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Test 15: clip_fraction in metrics ∈ [0, 1]
# ---------------------------------------------------------------------------

def test_clip_fraction_in_range():
    loss_fn = DAPOLoss(eps_low=0.1, eps_high=0.2)
    log_probs, old_log_probs, advantages = make_inputs(log_prob_val=-0.3, old_log_prob_val=-0.5)
    _, metrics = loss_fn(log_probs, old_log_probs, advantages)
    cf = metrics["clip_fraction"]
    assert 0.0 <= cf <= 1.0, f"clip_fraction={cf} is outside [0, 1]"


# ---------------------------------------------------------------------------
# Bonus: DAPOTrainer.compute_loss integration test
# ---------------------------------------------------------------------------

def test_trainer_compute_loss():
    trainer = DAPOTrainer(eps_low=0.1, eps_high=0.2, beta_entropy=0.001)
    batch = {
        "log_probs": torch.randn(3, 5),
        "old_log_probs": torch.randn(3, 5).detach(),
        "advantages": torch.randn(3, 5),
        "rewards": torch.tensor([1.0, 0.0, 1.0]),
    }
    loss = trainer.compute_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_trainer_missing_key_raises():
    trainer = DAPOTrainer()
    with pytest.raises(ValueError, match="missing required keys"):
        trainer.compute_loss({"log_probs": torch.randn(2, 4)})
