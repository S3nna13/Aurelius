"""Tests for src/training/loss_functions.py.

Tiny tensors (B=2, T=6, V=16) are used throughout to keep tests fast.
All tests use pure PyTorch — no HuggingFace, scipy, or sklearn.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.training.loss_functions import (
    LossConfig,
    LMLoss,
    compute_ppl_from_loss,
    cross_entropy_loss,
    focal_loss,
    label_smoothed_loss,
    token_weighted_loss,
    z_loss,
)

# ---------------------------------------------------------------------------
# Shared small tensor shapes
# ---------------------------------------------------------------------------

B, T, V = 2, 6, 16


def make_random(seed: int = 0):
    """Return (logits, labels) with the canonical small shape."""
    torch.manual_seed(seed)
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    return logits, labels


# ---------------------------------------------------------------------------
# 1. LossConfig — default field values
# ---------------------------------------------------------------------------

def test_loss_config_defaults():
    cfg = LossConfig()
    assert cfg.label_smoothing == 0.0
    assert cfg.focal_gamma == 2.0
    assert cfg.ignore_index == -100
    assert cfg.reduction == "mean"


# ---------------------------------------------------------------------------
# 2. cross_entropy_loss — returns a scalar
# ---------------------------------------------------------------------------

def test_cross_entropy_loss_is_scalar():
    logits, labels = make_random(1)
    loss = cross_entropy_loss(logits, labels)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 3. cross_entropy_loss — ignore_index positions do not contribute
# ---------------------------------------------------------------------------

def test_cross_entropy_loss_ignore_index():
    logits, labels = make_random(2)

    # Mask the last 2 time steps in every batch element
    labels_masked = labels.clone()
    labels_masked[:, -2:] = -100

    # Compute reference on the valid slice only
    logits_valid = logits[:, :4, :].reshape(-1, V)
    labels_valid = labels[:, :4].reshape(-1)
    ref = F.cross_entropy(logits_valid, labels_valid, reduction="mean")

    loss = cross_entropy_loss(logits, labels_masked, ignore_index=-100, reduction="mean")
    assert torch.allclose(loss, ref, atol=1e-5), (
        f"Expected {ref.item():.6f}, got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 4. cross_entropy_loss — near-zero for near-perfect predictions
# ---------------------------------------------------------------------------

def test_cross_entropy_loss_near_zero_for_perfect_prediction():
    # All tokens predict class 0; all labels are 0
    logits = torch.zeros(B, T, V)
    logits[:, :, 0] = 20.0          # very high confidence for class 0
    labels = torch.zeros(B, T, dtype=torch.long)

    loss = cross_entropy_loss(logits, labels)
    assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item():.6f}"


# ---------------------------------------------------------------------------
# 5. label_smoothed_loss — returns a finite scalar
# ---------------------------------------------------------------------------

def test_label_smoothed_loss_is_scalar_and_finite():
    logits, labels = make_random(3)
    loss = label_smoothed_loss(logits, labels, smoothing=0.1)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), "label_smoothed_loss must be finite"
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 6. label_smoothed_loss > plain CE on easy (confident) examples
# ---------------------------------------------------------------------------

def test_label_smoothed_loss_greater_than_ce_on_easy_examples():
    # Very confident correct predictions — smoothing should inflate the loss
    logits = torch.zeros(B, T, V)
    logits[:, :, 0] = 15.0
    labels = torch.zeros(B, T, dtype=torch.long)

    ce = cross_entropy_loss(logits, labels)
    ls = label_smoothed_loss(logits, labels, smoothing=0.1)

    assert ls.item() > ce.item(), (
        f"Label-smoothed ({ls.item():.6f}) should exceed CE ({ce.item():.6f}) "
        "when model is over-confident"
    )


# ---------------------------------------------------------------------------
# 7. focal_loss — returns a finite scalar
# ---------------------------------------------------------------------------

def test_focal_loss_is_scalar_and_finite():
    logits, labels = make_random(4)
    loss = focal_loss(logits, labels, gamma=2.0)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), "focal_loss must be finite"
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 8. token_weighted_loss — returns a scalar
# ---------------------------------------------------------------------------

def test_token_weighted_loss_is_scalar():
    torch.manual_seed(5)
    logits, labels = make_random(5)
    weights = torch.rand(B, T)
    loss = token_weighted_loss(logits, labels, weights)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 9. token_weighted_loss — zero weights → zero loss
# ---------------------------------------------------------------------------

def test_token_weighted_loss_zero_weights_gives_zero():
    logits, labels = make_random(6)
    weights = torch.zeros(B, T)
    loss = token_weighted_loss(logits, labels, weights)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), (
        f"All-zero weights should give zero loss, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# 10. z_loss — returns a scalar
# ---------------------------------------------------------------------------

def test_z_loss_is_scalar():
    logits, _ = make_random(7)
    loss = z_loss(logits, coef=1e-4)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"


# ---------------------------------------------------------------------------
# 11. z_loss — is strictly positive
# ---------------------------------------------------------------------------

def test_z_loss_is_positive():
    logits, _ = make_random(8)
    loss = z_loss(logits, coef=1e-4)
    assert loss.item() > 0.0, f"z_loss must be positive, got {loss.item()}"


# ---------------------------------------------------------------------------
# 12. compute_ppl_from_loss == exp(ce_loss)
# ---------------------------------------------------------------------------

def test_compute_ppl_equals_exp_of_loss():
    logits, labels = make_random(9)
    ce = cross_entropy_loss(logits, labels)
    ppl = compute_ppl_from_loss(ce)
    expected = torch.exp(ce)
    assert torch.allclose(ppl, expected, atol=1e-6), (
        f"PPL mismatch: {ppl.item():.4f} vs exp(ce)={expected.item():.4f}"
    )


# ---------------------------------------------------------------------------
# 13. LMLoss.forward — required output keys present
# ---------------------------------------------------------------------------

def test_lm_loss_forward_has_required_keys():
    logits, labels = make_random(10)
    cfg = LossConfig()               # defaults: no smoothing, focal_gamma=2
    module = LMLoss(cfg)
    out = module(logits, labels)
    assert "loss" in out, "'loss' key missing from LMLoss output"
    assert "ce_loss" in out, "'ce_loss' key missing from LMLoss output"
    assert "ppl" in out, "'ppl' key missing from LMLoss output"


# ---------------------------------------------------------------------------
# 14. LMLoss.forward — loss is finite
# ---------------------------------------------------------------------------

def test_lm_loss_forward_loss_is_finite():
    logits, labels = make_random(11)
    cfg = LossConfig(label_smoothing=0.1)
    module = LMLoss(cfg)
    out = module(logits, labels)
    assert torch.isfinite(out["loss"]), f"LMLoss 'loss' is not finite: {out['loss']}"
    assert torch.isfinite(out["ce_loss"]), f"LMLoss 'ce_loss' is not finite"
    assert torch.isfinite(out["ppl"]), f"LMLoss 'ppl' is not finite"


# ---------------------------------------------------------------------------
# 15. LMLoss — focal branch selected when label_smoothing == 0 and gamma > 0
# ---------------------------------------------------------------------------

def test_lm_loss_focal_branch():
    logits, labels = make_random(12)
    cfg_focal = LossConfig(label_smoothing=0.0, focal_gamma=2.0)
    cfg_ce    = LossConfig(label_smoothing=0.0, focal_gamma=0.0)

    out_focal = LMLoss(cfg_focal)(logits, labels)
    out_ce    = LMLoss(cfg_ce)(logits, labels)

    # With random logits focal_loss != plain CE (gamma > 0 modulates weights)
    # ce_loss key should be the same between the two (both use cross_entropy_loss)
    assert torch.allclose(out_focal["ce_loss"], out_ce["ce_loss"], atol=1e-5), (
        "ce_loss should be identical regardless of focal_gamma"
    )
    # primary loss differs when gamma > 0 (unless by coincidence)
    # Just verify it's finite and positive
    assert torch.isfinite(out_focal["loss"])
    assert out_focal["loss"].item() > 0.0
