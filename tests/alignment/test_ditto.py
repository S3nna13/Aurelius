"""Tests for DITTO implementation (src/alignment/ditto.py).

Covers:
  1.  DITTOLoss output is scalar
  2.  Gradients are finite after backward
  3.  Determinism: same inputs → same output
  4.  batch_size=1 works
  5.  Loss decreases when reward margin increases (correct ranking)
  6.  Loss ≈ log(2) when margin = 0 (random preference)
  7.  Reward accuracy > 0.5 when y_w is clearly better
  8.  No NaN/Inf on extreme log_probs (±100)
  9.  No NaN/Inf when all inputs are equal
  10. DITTOReferenceUpdater.update: ref params move toward policy
  11. DITTOReferenceUpdater.update: alpha=1.0 leaves ref unchanged
  12. DITTOReferenceUpdater.update: alpha=0.0 sets ref = policy
  13. DITTOReferenceUpdater.hard_update: ref params exactly equal policy
  14. DITTOTrainer.compute_loss matches DITTOLoss directly
  15. Reference updater preserves requires_grad=False on ref params
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from src.alignment.ditto import DITTOLoss, DITTOReferenceUpdater, DITTOTrainer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_log_probs(B: int, seed: int = 0) -> torch.Tensor:
    """Return shape-(B,) negative values suitable as sequence log-probs."""
    torch.manual_seed(seed)
    return -torch.rand(B) * 10.0  # values in [-10, 0]


def make_simple_model(in_features: int = 8, seed: int = 0) -> nn.Linear:
    """Tiny single-layer model for EMA / hard-update tests."""
    torch.manual_seed(seed)
    return nn.Linear(in_features, in_features, bias=False)


def make_ref_model(policy: nn.Module) -> nn.Module:
    """Deep-copy a model and freeze its parameters."""
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# ---------------------------------------------------------------------------
# Test 1: DITTOLoss output is scalar
# ---------------------------------------------------------------------------


def test_ditto_loss_output_is_scalar():
    criterion = DITTOLoss(beta=0.1)
    B = 4
    lp_w = make_log_probs(B, seed=1)
    lp_l = make_log_probs(B, seed=2)
    ref_w = make_log_probs(B, seed=3)
    ref_l = make_log_probs(B, seed=4)

    loss, metrics = criterion(lp_w, lp_l, ref_w, ref_l)

    assert isinstance(loss, torch.Tensor), "loss must be a Tensor"
    assert loss.ndim == 0, f"loss must be a scalar (0-dim), got ndim={loss.ndim}"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 2: Gradients are finite after backward
# ---------------------------------------------------------------------------


def test_ditto_loss_gradients_finite():
    criterion = DITTOLoss(beta=0.1)
    B = 6
    lp_w = make_log_probs(B, seed=5).requires_grad_(True)
    lp_l = make_log_probs(B, seed=6).requires_grad_(True)
    ref_w = make_log_probs(B, seed=7)
    ref_l = make_log_probs(B, seed=8)

    loss, _ = criterion(lp_w, lp_l, ref_w, ref_l)
    loss.backward()

    for name, t in [("lp_w", lp_w), ("lp_l", lp_l)]:
        assert t.grad is not None, f"No gradient on {name}"
        assert torch.isfinite(t.grad).all(), f"Non-finite gradient on {name}: {t.grad}"


# ---------------------------------------------------------------------------
# Test 3: Determinism — same inputs produce same output
# ---------------------------------------------------------------------------


def test_ditto_loss_determinism():
    criterion = DITTOLoss(beta=0.1)
    B = 8
    lp_w = make_log_probs(B, seed=9)
    lp_l = make_log_probs(B, seed=10)
    ref_w = make_log_probs(B, seed=11)
    ref_l = make_log_probs(B, seed=12)

    loss1, _ = criterion(lp_w, lp_l, ref_w, ref_l)
    loss2, _ = criterion(lp_w, lp_l, ref_w, ref_l)

    assert torch.allclose(loss1, loss2), f"Non-deterministic: {loss1.item()} vs {loss2.item()}"


# ---------------------------------------------------------------------------
# Test 4: batch_size=1 works
# ---------------------------------------------------------------------------


def test_ditto_loss_batch_size_one():
    criterion = DITTOLoss(beta=0.1)
    lp_w = torch.tensor([-1.0])
    lp_l = torch.tensor([-3.0])
    ref_w = torch.tensor([-2.0])
    ref_l = torch.tensor([-2.5])

    loss, metrics = criterion(lp_w, lp_l, ref_w, ref_l)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "reward_margin" in metrics
    assert "reward_accuracy" in metrics


# ---------------------------------------------------------------------------
# Test 5: Loss decreases when reward margin increases
# ---------------------------------------------------------------------------


def test_ditto_loss_decreases_with_larger_margin():
    """A larger positive reward margin should produce a lower loss."""
    criterion = DITTOLoss(beta=0.1)
    B = 4
    ref = torch.zeros(B)  # neutral reference (log-ratio = raw log-prob)

    # Small margin: y_w slightly better than y_l
    lp_w_small = torch.full((B,), -1.0)
    lp_l_small = torch.full((B,), -1.5)
    loss_small, _ = criterion(lp_w_small, lp_l_small, ref, ref)

    # Large margin: y_w much better than y_l
    lp_w_large = torch.full((B,), -0.5)
    lp_l_large = torch.full((B,), -5.0)
    loss_large, _ = criterion(lp_w_large, lp_l_large, ref, ref)

    assert loss_large.item() < loss_small.item(), (
        f"Loss should decrease with larger margin: "
        f"loss_large={loss_large.item():.4f} >= loss_small={loss_small.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Loss ≈ log(2) when margin = 0
# ---------------------------------------------------------------------------


def test_ditto_loss_equals_log2_at_zero_margin():
    """When β * ((lp_w − ref_w) − (lp_l − ref_l)) = 0 for all items,
    loss = −log σ(0) = log(2) ≈ 0.6931."""
    criterion = DITTOLoss(beta=0.1)
    B = 32
    same = torch.zeros(B)

    loss, _ = criterion(same, same, same, same)

    expected = math.log(2)
    assert abs(loss.item() - expected) < 1e-5, (
        f"Expected loss ≈ {expected:.4f} (log 2), got {loss.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: Reward accuracy > 0.5 when y_w is clearly better
# ---------------------------------------------------------------------------


def test_ditto_reward_accuracy_when_winner_is_clear():
    criterion = DITTOLoss(beta=0.1)
    B = 16
    ref = torch.zeros(B)

    # y_w has much higher log-prob under policy than y_l
    lp_w = torch.full((B,), -0.1)
    lp_l = torch.full((B,), -10.0)

    _, metrics = criterion(lp_w, lp_l, ref, ref)
    assert metrics["reward_accuracy"] > 0.5, (
        f"Expected reward_accuracy > 0.5, got {metrics['reward_accuracy']}"
    )


# ---------------------------------------------------------------------------
# Test 8: No NaN/Inf on extreme log_probs (±100)
# ---------------------------------------------------------------------------


def test_ditto_loss_no_nan_on_extreme_inputs():
    criterion = DITTOLoss(beta=0.1)
    B = 4
    lp_w = torch.full((B,), 100.0)
    lp_l = torch.full((B,), -100.0)
    ref_w = torch.full((B,), 100.0)
    ref_l = torch.full((B,), -100.0)

    loss, metrics = criterion(lp_w, lp_l, ref_w, ref_l)

    assert not torch.isnan(loss), f"NaN loss on extreme inputs: {loss}"
    assert not torch.isinf(loss), f"Inf loss on extreme inputs: {loss}"
    assert math.isfinite(metrics["reward_margin"]), "reward_margin not finite"
    assert math.isfinite(metrics["reward_accuracy"]), "reward_accuracy not finite"


# ---------------------------------------------------------------------------
# Test 9: No NaN/Inf when all inputs are equal
# ---------------------------------------------------------------------------


def test_ditto_loss_no_nan_equal_inputs():
    criterion = DITTOLoss(beta=0.1)
    B = 4
    val = torch.full((B,), -5.0)

    loss, metrics = criterion(val, val, val, val)

    assert not torch.isnan(loss), f"NaN loss when inputs are equal: {loss}"
    assert not torch.isinf(loss), f"Inf loss when inputs are equal: {loss}"


# ---------------------------------------------------------------------------
# Test 10: DITTOReferenceUpdater.update moves ref toward policy
# ---------------------------------------------------------------------------


def test_reference_updater_soft_update_moves_toward_policy():
    """After EMA update with alpha < 1, ref params should be closer to policy."""
    torch.manual_seed(42)
    policy = make_simple_model(seed=0)
    ref = make_ref_model(policy)

    # Perturb the ref so it differs from policy
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 5.0)

    # Compute initial distance
    def l2_dist(m1: nn.Module, m2: nn.Module) -> float:
        total = 0.0
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            total += (p1.data - p2.data).pow(2).sum().item()
        return total**0.5

    dist_before = l2_dist(ref, policy)

    updater = DITTOReferenceUpdater(alpha=0.5)
    updater.update(policy, ref)

    dist_after = l2_dist(ref, policy)

    assert dist_after < dist_before, (
        f"EMA update should bring ref closer to policy: "
        f"before={dist_before:.4f}, after={dist_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 11: DITTOReferenceUpdater.update: alpha=1.0 leaves ref unchanged
# ---------------------------------------------------------------------------


def test_reference_updater_alpha_one_unchanged():
    """With alpha=1.0, the EMA leaves ref params exactly unchanged."""
    torch.manual_seed(43)
    policy = make_simple_model(seed=1)
    ref = make_ref_model(policy)

    # Perturb ref
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.ones_like(p) * 3.0)

    # Record original ref params
    ref_params_before = [p.data.clone() for p in ref.parameters()]

    updater = DITTOReferenceUpdater(alpha=1.0)
    updater.update(policy, ref)

    for before, after in zip(ref_params_before, ref.parameters()):
        assert torch.allclose(before, after.data), "alpha=1.0 should leave ref params unchanged"


# ---------------------------------------------------------------------------
# Test 12: DITTOReferenceUpdater.update: alpha=0.0 sets ref = policy
# ---------------------------------------------------------------------------


def test_reference_updater_alpha_zero_copies_policy():
    """With alpha=0.0, ref params become exactly equal to policy params."""
    torch.manual_seed(44)
    policy = make_simple_model(seed=2)
    ref = make_ref_model(policy)

    # Perturb ref so it starts different
    with torch.no_grad():
        for p in ref.parameters():
            p.fill_(999.0)

    updater = DITTOReferenceUpdater(alpha=0.0)
    updater.update(policy, ref)

    for pol_p, ref_p in zip(policy.parameters(), ref.parameters()):
        assert torch.allclose(pol_p.data, ref_p.data), (
            "alpha=0.0 EMA update should set ref = policy"
        )


# ---------------------------------------------------------------------------
# Test 13: DITTOReferenceUpdater.hard_update: ref params exactly equal policy
# ---------------------------------------------------------------------------


def test_reference_updater_hard_update_exact_copy():
    """hard_update must copy all policy params verbatim into ref_model."""
    torch.manual_seed(45)
    policy = make_simple_model(seed=3)
    ref = make_ref_model(policy)

    # Diverge ref
    with torch.no_grad():
        for p in ref.parameters():
            p.fill_(-42.0)

    updater = DITTOReferenceUpdater(alpha=0.9)
    updater.hard_update(policy, ref)

    for pol_p, ref_p in zip(policy.parameters(), ref.parameters()):
        assert torch.allclose(pol_p.data, ref_p.data), (
            "hard_update must make ref params exactly equal to policy params"
        )


# ---------------------------------------------------------------------------
# Test 14: DITTOTrainer.compute_loss matches DITTOLoss directly
# ---------------------------------------------------------------------------


def test_ditto_trainer_compute_loss_matches_criterion():
    """DITTOTrainer.compute_loss should produce the same loss as calling
    DITTOLoss directly with the same inputs."""
    beta = 0.2
    B = 8
    torch.manual_seed(99)
    lp_w = make_log_probs(B, seed=20)
    lp_l = make_log_probs(B, seed=21)
    ref_w = make_log_probs(B, seed=22)
    ref_l = make_log_probs(B, seed=23)

    # Direct criterion
    criterion = DITTOLoss(beta=beta)
    expected_loss, _ = criterion(lp_w, lp_l, ref_w, ref_l)

    # Via trainer
    trainer = DITTOTrainer(beta=beta)
    batch = {
        "log_probs_w": lp_w,
        "log_probs_l": lp_l,
        "ref_log_probs_w": ref_w,
        "ref_log_probs_l": ref_l,
    }
    actual_loss = trainer.compute_loss(batch)

    assert torch.allclose(expected_loss, actual_loss), (
        f"Trainer loss {actual_loss.item()} != direct criterion loss {expected_loss.item()}"
    )


# ---------------------------------------------------------------------------
# Test 15: Reference updater preserves requires_grad=False on ref params
# ---------------------------------------------------------------------------


def test_reference_updater_preserves_no_grad_on_ref():
    """Both soft and hard updates must not accidentally set requires_grad=True
    on reference model parameters."""
    torch.manual_seed(46)
    policy = make_simple_model(seed=4)
    ref = make_ref_model(policy)

    # Confirm starting state
    for p in ref.parameters():
        assert not p.requires_grad, "ref should start with requires_grad=False"

    updater = DITTOReferenceUpdater(alpha=0.95)

    # Soft update
    updater.update(policy, ref)
    for p in ref.parameters():
        assert not p.requires_grad, "soft update must not enable requires_grad on ref params"

    # Hard update
    updater.hard_update(policy, ref)
    for p in ref.parameters():
        assert not p.requires_grad, "hard_update must not enable requires_grad on ref params"
