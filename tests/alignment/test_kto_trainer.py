"""Tests for src/alignment/kto_trainer.py — KTO (Ethayarajh et al. 2024).

Pure PyTorch; no transformers, scipy, sklearn, trl, etc.

Tests: 14 unit tests + 1 integration test = 15 total.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.alignment.kto_trainer import KTOBatch, KTOConfig, KTOTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 6,
    T: int = 12,
    n_desirable: int | None = None,
    seed: int = 42,
    lp_offset: float = 0.0,
    ref_offset: float = 0.0,
) -> KTOBatch:
    """Return a synthetic KTOBatch with controllable desirability split."""
    g = torch.Generator()
    g.manual_seed(seed)
    log_probs = torch.randn(B, T, generator=g) - 3.0 + lp_offset
    ref_log_probs = torch.randn(B, T, generator=g) - 3.0 + ref_offset
    # All tokens real (no padding) for simplicity
    attention_mask = torch.ones(B, T)

    if n_desirable is None:
        n_desirable = B // 2
    desirable = torch.zeros(B, dtype=torch.bool)
    desirable[:n_desirable] = True

    return KTOBatch(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        attention_mask=attention_mask,
        desirable=desirable,
    )


# ---------------------------------------------------------------------------
# Test 1: test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = KTOConfig()
    assert cfg.beta == 0.1
    assert cfg.lambda_u == 1.0
    assert cfg.kl_num_samples == 8
    assert cfg.eps > 0.0


# ---------------------------------------------------------------------------
# Test 2: test_sequence_log_ratio_shape
# ---------------------------------------------------------------------------


def test_sequence_log_ratio_shape():
    trainer = KTOTrainer(KTOConfig())
    B, T = 5, 10
    lp = torch.randn(B, T)
    ref_lp = torch.randn(B, T)
    mask = torch.ones(B, T)
    out = trainer.sequence_log_ratio(lp, ref_lp, mask)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: test_sequence_log_ratio_zero — identical probs → all zeros
# ---------------------------------------------------------------------------


def test_sequence_log_ratio_zero():
    trainer = KTOTrainer(KTOConfig())
    B, T = 4, 8
    lp = torch.randn(B, T)
    mask = torch.ones(B, T)
    out = trainer.sequence_log_ratio(lp, lp, mask)
    assert torch.allclose(out, torch.zeros(B), atol=1e-6), (
        f"Expected all zeros when log_probs == ref_log_probs, got {out}"
    )


# ---------------------------------------------------------------------------
# Test 4: test_estimate_kl_non_negative
# ---------------------------------------------------------------------------


def test_estimate_kl_non_negative():
    """z_ref must always be >= 0 (clamped from below)."""
    trainer = KTOTrainer(KTOConfig())
    B, T = 8, 16
    # Deliberately force negative mean ratio
    lp = torch.full((B, T), -5.0)
    ref_lp = torch.full((B, T), 0.0)
    mask = torch.ones(B, T)
    z_ref = trainer.estimate_kl(lp, ref_lp, mask)
    assert z_ref.item() >= 0.0, f"z_ref={z_ref.item()} should be >= 0"


# ---------------------------------------------------------------------------
# Test 5: test_desirable_loss_bounds — output in (0, 1)
# ---------------------------------------------------------------------------


def test_desirable_loss_bounds():
    trainer = KTOTrainer(KTOConfig(beta=0.1))
    log_ratio = torch.randn(8)
    z_ref = torch.tensor(0.05)
    loss = trainer.desirable_loss(log_ratio, z_ref)
    assert 0.0 < loss.item() < 1.0, (
        f"desirable_loss={loss.item()} expected in (0, 1)"
    )


# ---------------------------------------------------------------------------
# Test 6: test_undesirable_loss_bounds — output in (0, 1)
# ---------------------------------------------------------------------------


def test_undesirable_loss_bounds():
    trainer = KTOTrainer(KTOConfig(beta=0.1))
    log_ratio = torch.randn(8)
    z_ref = torch.tensor(0.05)
    loss = trainer.undesirable_loss(log_ratio, z_ref)
    assert 0.0 < loss.item() < 1.0, (
        f"undesirable_loss={loss.item()} expected in (0, 1)"
    )


# ---------------------------------------------------------------------------
# Test 7: test_desirable_better_preferred — high log_ratio → lower d_loss
# ---------------------------------------------------------------------------


def test_desirable_better_preferred():
    """Higher log_ratio (policy >> ref) should reduce desirable loss."""
    trainer = KTOTrainer(KTOConfig(beta=0.5))
    z_ref = torch.tensor(0.0)
    low_ratio = torch.full((8,), -2.0)
    high_ratio = torch.full((8,), 2.0)
    loss_low = trainer.desirable_loss(low_ratio, z_ref)
    loss_high = trainer.desirable_loss(high_ratio, z_ref)
    assert loss_high.item() < loss_low.item(), (
        f"Expected d_loss(high={loss_high.item():.4f}) < d_loss(low={loss_low.item():.4f})"
    )


# ---------------------------------------------------------------------------
# Test 8: test_undesirable_low_preferred — low log_ratio → lower u_loss
# ---------------------------------------------------------------------------


def test_undesirable_low_preferred():
    """Lower log_ratio (policy << ref) should reduce undesirable loss."""
    trainer = KTOTrainer(KTOConfig(beta=0.5))
    z_ref = torch.tensor(0.0)
    low_ratio = torch.full((8,), -2.0)
    high_ratio = torch.full((8,), 2.0)
    loss_low = trainer.undesirable_loss(low_ratio, z_ref)
    loss_high = trainer.undesirable_loss(high_ratio, z_ref)
    assert loss_low.item() < loss_high.item(), (
        f"Expected u_loss(low={loss_low.item():.4f}) < u_loss(high={loss_high.item():.4f})"
    )


# ---------------------------------------------------------------------------
# Test 9: test_total_loss_keys
# ---------------------------------------------------------------------------


def test_total_loss_keys():
    trainer = KTOTrainer(KTOConfig())
    batch = _make_batch(B=6, T=10)
    result = trainer.total_loss(batch)
    required = {"loss", "desirable_loss", "undesirable_loss", "z_ref"}
    assert required == set(result.keys()), (
        f"Missing or extra keys. Got {set(result.keys())}, expected {required}"
    )


# ---------------------------------------------------------------------------
# Test 10: test_total_loss_scalar
# ---------------------------------------------------------------------------


def test_total_loss_scalar():
    trainer = KTOTrainer(KTOConfig())
    batch = _make_batch(B=6, T=10)
    result = trainer.total_loss(batch)
    for key, val in result.items():
        assert val.shape == (), f"Expected scalar for '{key}', got shape {val.shape}"
    assert torch.isfinite(result["loss"]), "Total loss must be finite"


# ---------------------------------------------------------------------------
# Test 11: test_all_desirable_batch
# ---------------------------------------------------------------------------


def test_all_desirable_batch():
    """Batch with only desirable samples: undesirable_loss == 0, loss > 0."""
    trainer = KTOTrainer(KTOConfig())
    batch = _make_batch(B=4, T=8, n_desirable=4)
    result = trainer.total_loss(batch)
    assert result["undesirable_loss"].item() == 0.0, (
        f"undesirable_loss should be 0 for all-desirable batch, "
        f"got {result['undesirable_loss'].item()}"
    )
    assert torch.isfinite(result["loss"])
    assert result["loss"].item() > 0.0


# ---------------------------------------------------------------------------
# Test 12: test_all_undesirable_batch
# ---------------------------------------------------------------------------


def test_all_undesirable_batch():
    """Batch with only undesirable samples: desirable_loss == 0, loss > 0."""
    trainer = KTOTrainer(KTOConfig())
    batch = _make_batch(B=4, T=8, n_desirable=0)
    result = trainer.total_loss(batch)
    assert result["desirable_loss"].item() == 0.0, (
        f"desirable_loss should be 0 for all-undesirable batch, "
        f"got {result['desirable_loss'].item()}"
    )
    assert torch.isfinite(result["loss"])
    assert result["loss"].item() > 0.0


# ---------------------------------------------------------------------------
# Test 13: test_lambda_u_scaling
# ---------------------------------------------------------------------------


def test_lambda_u_scaling():
    """Doubling lambda_u should roughly double the undesirable loss contribution."""
    batch = _make_batch(B=8, T=10, n_desirable=4, seed=99)

    cfg1 = KTOConfig(lambda_u=1.0)
    cfg2 = KTOConfig(lambda_u=2.0)

    result1 = KTOTrainer(cfg1).total_loss(batch)
    result2 = KTOTrainer(cfg2).total_loss(batch)

    # The undesirable_loss component itself is independent of lambda_u
    assert torch.allclose(result1["undesirable_loss"], result2["undesirable_loss"], atol=1e-6), (
        "undesirable_loss component should not change with lambda_u"
    )

    # But the total loss should be higher with lambda_u=2.0
    assert result2["loss"].item() > result1["loss"].item(), (
        "Higher lambda_u must produce higher total loss"
    )

    # Quantitative check: total_loss2 - total_loss1 ≈ 1.0 * u_loss
    delta = result2["loss"].item() - result1["loss"].item()
    expected_delta = result1["undesirable_loss"].item()
    assert abs(delta - expected_delta) < 1e-5, (
        f"Expected Δloss = u_loss = {expected_delta:.6f}, got {delta:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 14: test_gradient_flows
# ---------------------------------------------------------------------------


def test_gradient_flows():
    """Gradients should reach log_probs and be finite."""
    trainer = KTOTrainer(KTOConfig(beta=0.1))
    B, T = 6, 10
    log_probs = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.randn(B, T)
    attention_mask = torch.ones(B, T)
    desirable = torch.zeros(B, dtype=torch.bool)
    desirable[:3] = True

    batch = KTOBatch(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        attention_mask=attention_mask,
        desirable=desirable,
    )

    result = trainer.total_loss(batch)
    result["loss"].backward()

    assert log_probs.grad is not None, "Gradient should exist for log_probs"
    assert torch.isfinite(log_probs.grad).all(), (
        "Gradients must be finite, found NaN/Inf"
    )


# ---------------------------------------------------------------------------
# Test 15: Integration — B=8, T=16, 4 desirable + 4 undesirable
# ---------------------------------------------------------------------------


def test_integration_full_batch():
    """End-to-end integration: forward + backward on a realistic batch."""
    B, T = 8, 16
    torch.manual_seed(2024)

    log_probs = torch.randn(B, T, requires_grad=True)
    ref_log_probs = torch.randn(B, T).detach()
    # Slight padding at the end to exercise masking
    attention_mask = torch.ones(B, T)
    attention_mask[:, -2:] = 0  # last 2 tokens are padding

    desirable = torch.zeros(B, dtype=torch.bool)
    desirable[:4] = True  # first 4 desirable, last 4 undesirable

    batch = KTOBatch(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        attention_mask=attention_mask,
        desirable=desirable,
    )

    cfg = KTOConfig(beta=0.1, lambda_u=1.0, kl_num_samples=4)
    trainer = KTOTrainer(cfg)

    result = trainer.total_loss(batch)

    # Structural checks
    assert set(result.keys()) == {"loss", "desirable_loss", "undesirable_loss", "z_ref"}
    for key, val in result.items():
        assert val.shape == (), f"'{key}' must be scalar"
        assert torch.isfinite(val), f"'{key}'={val.item()} is not finite"

    # Both components active
    assert result["desirable_loss"].item() > 0.0
    assert result["undesirable_loss"].item() > 0.0

    # Backward pass
    result["loss"].backward()
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all(), "Gradients must be finite after backward"

    # Statistics helper
    stats = trainer.statistics(batch)
    for k, v in stats.items():
        assert math.isfinite(v), f"statistic '{k}'={v} is not finite"
    assert "mean_log_ratio" in stats
    assert "z_ref" in stats
    assert "desirable_mean_ratio" in stats
    assert "undesirable_mean_ratio" in stats

    # Registry
    from src.alignment import ALIGNMENT_REGISTRY
    assert "kto" in ALIGNMENT_REGISTRY
    assert ALIGNMENT_REGISTRY["kto"] is KTOTrainer
