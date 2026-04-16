"""Tests for src/training/long_tail_loss.py.

All tests use tiny tensors (B=2, T=4, N_CLASSES=8) to keep them fast and
dependency-free.  Only pure PyTorch is used.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.training.long_tail_loss import (
    LongTailConfig,
    SeesawLoss,
    balanced_softmax_loss,
    class_balanced_loss,
    compute_class_weights,
    compute_effective_num_weights,
    logit_adjusted_loss,
)

# ---------------------------------------------------------------------------
# Shared tiny-problem constants
# ---------------------------------------------------------------------------
B = 2
T = 4
N_CLASSES = 8

torch.manual_seed(0)


def _make_logits(b=B, t=T, c=N_CLASSES):
    return torch.randn(b, t, c)


def _make_labels(b=B, t=T, c=N_CLASSES):
    """Random valid labels in [0, c)."""
    return torch.randint(0, c, (b, t))


def _make_counts(c=N_CLASSES, rare_idx=0, rare_count=5, common_count=500):
    counts = torch.full((c,), float(common_count))
    counts[rare_idx] = float(rare_count)
    return counts


# ===========================================================================
# 1. LongTailConfig defaults
# ===========================================================================

def test_config_defaults():
    cfg = LongTailConfig()
    assert cfg.n_classes == 50257
    assert cfg.smoothing == 0.0
    assert cfg.focal_gamma == 0.0
    assert cfg.class_freq_power == 0.5
    assert cfg.effective_num_beta == 0.9999
    assert cfg.margin == 0.0


def test_config_custom():
    cfg = LongTailConfig(n_classes=100, smoothing=0.1, focal_gamma=2.0)
    assert cfg.n_classes == 100
    assert cfg.smoothing == 0.1
    assert cfg.focal_gamma == 2.0


# ===========================================================================
# 2. compute_class_weights
# ===========================================================================

def test_class_weights_shape():
    counts = _make_counts()
    w = compute_class_weights(counts)
    assert w.shape == (N_CLASSES,)


def test_class_weights_sum_to_n_classes():
    counts = _make_counts()
    w = compute_class_weights(counts)
    assert abs(w.sum().item() - N_CLASSES) < 1e-4


def test_class_weights_rare_gets_higher_weight():
    counts = _make_counts(rare_idx=0, rare_count=5, common_count=500)
    w = compute_class_weights(counts)
    # Class 0 has far fewer samples, so its weight should be largest
    assert w[0].item() == pytest.approx(w.max().item(), rel=1e-5)
    assert w[0].item() > w[1].item()


def test_class_weights_power_zero_is_uniform():
    counts = _make_counts()
    w = compute_class_weights(counts, power=0.0)
    # power=0 → all weights = 1 / 1 = 1, then normalised → all equal
    assert torch.allclose(w, torch.ones_like(w), atol=1e-5)


# ===========================================================================
# 3. compute_effective_num_weights
# ===========================================================================

def test_effective_num_weights_shape():
    counts = _make_counts()
    w = compute_effective_num_weights(counts)
    assert w.shape == (N_CLASSES,)


def test_effective_num_weights_sum_to_n_classes():
    counts = _make_counts()
    w = compute_effective_num_weights(counts)
    assert abs(w.sum().item() - N_CLASSES) < 1e-4


def test_effective_num_weights_rare_gets_higher_weight():
    counts = _make_counts(rare_idx=2, rare_count=3, common_count=1000)
    w = compute_effective_num_weights(counts)
    assert w[2].item() > w[1].item()


# ===========================================================================
# 4. class_balanced_loss
# ===========================================================================

def test_class_balanced_loss_is_scalar():
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts()
    weights = compute_class_weights(counts)
    loss = class_balanced_loss(logits, labels, weights)
    assert loss.shape == ()
    assert loss.item() > 0.0


def test_class_balanced_loss_handles_ignore_index():
    logits = _make_logits()
    labels = _make_labels()
    labels[0, :] = -100  # mask out the entire first sequence
    counts = _make_counts()
    weights = compute_class_weights(counts)
    loss = class_balanced_loss(logits, labels, weights, ignore_index=-100)
    assert torch.isfinite(loss)


def test_class_balanced_loss_all_ignored_returns_zero():
    logits = _make_logits()
    labels = torch.full((B, T), -100, dtype=torch.long)
    counts = _make_counts()
    weights = compute_class_weights(counts)
    loss = class_balanced_loss(logits, labels, weights)
    assert loss.item() == 0.0


def test_class_balanced_loss_2d_input():
    """(N, C) input (no batch/seq dimension)."""
    N = B * T
    logits = torch.randn(N, N_CLASSES)
    labels = torch.randint(0, N_CLASSES, (N,))
    counts = _make_counts()
    weights = compute_class_weights(counts)
    loss = class_balanced_loss(logits, labels, weights)
    assert loss.shape == ()


# ===========================================================================
# 5. logit_adjusted_loss
# ===========================================================================

def test_logit_adjusted_loss_is_scalar():
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts().float()
    log_prior = (counts / counts.sum()).log()
    loss = logit_adjusted_loss(logits, labels, log_prior)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_logit_adjusted_loss_differs_from_standard_ce():
    """Logit adjustment should produce a different loss than vanilla CE."""
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts().float()
    # Very skewed prior so the adjustment is non-trivial
    counts[0] = 1.0
    counts[1:] = 10000.0
    log_prior = (counts / counts.sum()).log()

    loss_adjusted = logit_adjusted_loss(logits, labels, log_prior, tau=1.0)
    loss_standard = F.cross_entropy(logits.reshape(-1, N_CLASSES), labels.reshape(-1))

    assert not torch.allclose(loss_adjusted, loss_standard, atol=1e-4)


def test_logit_adjusted_loss_tau_zero_equals_standard_ce():
    """With tau=0 no adjustment is applied; loss should match standard CE."""
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts().float()
    log_prior = (counts / counts.sum()).log()

    loss_adjusted = logit_adjusted_loss(logits, labels, log_prior, tau=0.0)
    loss_standard = F.cross_entropy(logits.reshape(-1, N_CLASSES), labels.reshape(-1))

    assert torch.allclose(loss_adjusted, loss_standard, atol=1e-5)


# ===========================================================================
# 6. balanced_softmax_loss
# ===========================================================================

def test_balanced_softmax_loss_is_scalar():
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts()
    loss = balanced_softmax_loss(logits, labels, counts)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_balanced_softmax_loss_uniform_counts_close_to_ce():
    """With uniform class counts the adjustment is constant → loss == CE."""
    logits = _make_logits()
    labels = _make_labels()
    counts = torch.ones(N_CLASSES) * 100.0

    loss_bsm = balanced_softmax_loss(logits, labels, counts)
    loss_ce = F.cross_entropy(logits.reshape(-1, N_CLASSES), labels.reshape(-1))

    # With uniform counts log(n_c / n_total) is the same for all classes,
    # so subtracting it shifts all logits by the same scalar — CE is invariant.
    assert torch.allclose(loss_bsm, loss_ce, atol=1e-5)


# ===========================================================================
# 7. SeesawLoss
# ===========================================================================

def test_seesaw_loss_forward_is_scalar():
    seesaw = SeesawLoss(n_classes=N_CLASSES)
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts()
    loss = seesaw(logits, labels, counts)
    assert loss.shape == ()


def test_seesaw_loss_is_finite():
    seesaw = SeesawLoss(n_classes=N_CLASSES, p=0.8, q=2.0)
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts()
    loss = seesaw(logits, labels, counts)
    assert torch.isfinite(loss)


def test_seesaw_loss_positive():
    seesaw = SeesawLoss(n_classes=N_CLASSES)
    logits = _make_logits()
    labels = _make_labels()
    counts = _make_counts()
    loss = seesaw(logits, labels, counts)
    assert loss.item() > 0.0


def test_seesaw_loss_all_ignored_returns_zero():
    seesaw = SeesawLoss(n_classes=N_CLASSES)
    logits = _make_logits()
    labels = torch.full((B, T), -100, dtype=torch.long)
    counts = _make_counts()
    loss = seesaw(logits, labels, counts)
    assert loss.item() == 0.0
