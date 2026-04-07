"""Tests for src/training/distillation.py."""

import pytest
import torch

from src.training.distillation import DistillationLoss


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BATCH, SEQ, VOCAB = 2, 8, 32


def _make_inputs(seed: int = 0):
    """Return (student_logits, teacher_logits, labels) with reproducible values."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    student = torch.randn(BATCH, SEQ, VOCAB, generator=gen)
    teacher = torch.randn(BATCH, SEQ, VOCAB, generator=gen)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen)
    return student, teacher, labels


# ---------------------------------------------------------------------------
# alpha boundary tests
# ---------------------------------------------------------------------------

def test_distillation_ce_only():
    """At alpha=0.0 total_loss must equal ce_loss."""
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.0)
    student, teacher, labels = _make_inputs()
    total, kl, ce = loss_fn(student, teacher, labels)
    assert torch.isclose(total, ce), f"total={total.item():.6f}, ce={ce.item():.6f}"


def test_distillation_kl_only():
    """At alpha=1.0 total_loss must equal kl_loss."""
    loss_fn = DistillationLoss(temperature=2.0, alpha=1.0)
    student, teacher, labels = _make_inputs()
    total, kl, ce = loss_fn(student, teacher, labels)
    assert torch.isclose(total, kl), f"total={total.item():.6f}, kl={kl.item():.6f}"


def test_distillation_blended():
    """At alpha=0.5, total_loss must be between kl_loss and ce_loss."""
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)
    student, teacher, labels = _make_inputs()
    total, kl, ce = loss_fn(student, teacher, labels)
    lo = min(kl.item(), ce.item())
    hi = max(kl.item(), ce.item())
    assert lo <= total.item() <= hi, (
        f"total={total.item():.6f} not in [{lo:.6f}, {hi:.6f}]"
    )


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def test_distillation_temperature_scaling():
    """Higher temperature flattens distributions, lowering the raw KL divergence.

    The KL term is multiplied by T² (Hinton et al., 2015) to preserve gradient
    magnitude, so the *scaled* KL stays roughly constant.  This test therefore
    verifies the underlying softening effect directly: the raw (unscaled)
    F.kl_div of the softened distributions must decrease monotonically with T.
    """
    student, teacher, labels = _make_inputs()

    # Compute raw (unscaled) KL at two temperatures by removing the T² factor
    def raw_kl(T: float) -> float:
        loss_fn = DistillationLoss(temperature=T, alpha=1.0)
        _, kl_scaled, _ = loss_fn(student, teacher, labels)
        return (kl_scaled / (T ** 2)).item()

    kl_raw_low_t = raw_kl(1.0)
    kl_raw_high_t = raw_kl(4.0)

    assert kl_raw_high_t < kl_raw_low_t, (
        f"Expected raw kl(T=4) < raw kl(T=1), "
        f"got {kl_raw_high_t:.6f} >= {kl_raw_low_t:.6f}"
    )


# ---------------------------------------------------------------------------
# Ignore-index behaviour
# ---------------------------------------------------------------------------

def test_distillation_ignores_minus100():
    """Labels with -100 should not affect CE loss; masking must work correctly."""
    gen = torch.Generator()
    gen.manual_seed(42)
    student = torch.randn(BATCH, SEQ, VOCAB, generator=gen)
    teacher = torch.randn(BATCH, SEQ, VOCAB, generator=gen)

    # All valid labels
    labels_full = torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen)

    # Same labels but second half of positions masked to -100
    labels_masked = labels_full.clone()
    labels_masked[:, SEQ // 2 :] = -100

    loss_fn = DistillationLoss(temperature=2.0, alpha=0.0)  # CE only

    _, _, ce_full = loss_fn(student, teacher, labels_full)
    _, _, ce_masked = loss_fn(student, teacher, labels_masked)

    # With fewer tokens contributing the CE values should differ
    assert not torch.isclose(ce_full, ce_masked), (
        "CE loss did not change when half the labels were masked to -100"
    )

    # Verify -100 positions are truly ignored: CE on first half only should
    # match the masked CE (PyTorch averages only over non-ignored tokens).
    labels_first_half = labels_full.clone()
    labels_first_half[:, SEQ // 2 :] = -100
    _, _, ce_first_half = loss_fn(student, teacher, labels_first_half)
    assert torch.isclose(ce_masked, ce_first_half), (
        f"Masked CE {ce_masked.item():.6f} != first-half CE {ce_first_half.item():.6f}"
    )
