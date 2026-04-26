"""Unit + integration tests for src/alignment/cpo_trainer.py — 15 tests."""

from __future__ import annotations

import pytest
import torch

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.cpo_trainer import CPOBatch, CPOConfig, CPOTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 4,
    T_w: int = 8,
    T_l: int = 6,
    chosen_scale: float = -1.0,
    rejected_scale: float = -3.0,
    requires_grad: bool = False,
) -> CPOBatch:
    """Create a synthetic CPOBatch with log-probs in (−∞, 0).

    chosen_scale / rejected_scale control how good each side looks;
    higher (less negative) means better.
    """
    chosen_lp = torch.full((B, T_w), chosen_scale).requires_grad_(requires_grad)
    rejected_lp = torch.full((B, T_l), rejected_scale)
    chosen_mask = torch.ones(B, T_w)
    rejected_mask = torch.ones(B, T_l)
    return CPOBatch(
        chosen_log_probs=chosen_lp,
        rejected_log_probs=rejected_lp,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )


def _make_padded_batch(B: int = 2, T: int = 8) -> CPOBatch:
    """Batch where the second sequence has padding in the last 2 positions."""
    lp = torch.full((B, T), -1.0)
    mask = torch.ones(B, T)
    mask[1, -2:] = 0.0  # pad last 2 tokens of second sequence
    return CPOBatch(
        chosen_log_probs=lp,
        rejected_log_probs=lp.clone(),
        chosen_mask=mask,
        rejected_mask=mask.clone(),
    )


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CPOConfig()
    assert cfg.beta == pytest.approx(0.1)
    assert cfg.sft_weight == pytest.approx(1.0)
    assert cfg.label_smoothing == pytest.approx(0.0)
    assert cfg.eps > 0


# ---------------------------------------------------------------------------
# 2. sequence_log_prob — shape
# ---------------------------------------------------------------------------


def test_sequence_log_prob_shape():
    trainer = CPOTrainer()
    B, T = 4, 10
    lp = torch.randn(B, T).clamp(max=0)
    mask = torch.ones(B, T)
    out = trainer.sequence_log_prob(lp, mask)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. sequence_log_prob — masked tokens excluded
# ---------------------------------------------------------------------------


def test_sequence_log_prob_masked():
    """Tokens zeroed by the mask must not contribute to the mean."""
    trainer = CPOTrainer()
    B, T = 2, 6
    # All -1.0 for valid tokens, but set a "poison" value in the padded slot
    lp = torch.full((B, T), -1.0)
    lp[:, -1] = -999.0  # will be masked out
    mask = torch.ones(B, T)
    mask[:, -1] = 0.0  # pad the last position

    out = trainer.sequence_log_prob(lp, mask)
    # Mean over 5 valid tokens, each = -1  →  -1.0
    assert torch.allclose(out, torch.full((B,), -1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# 4. sft_loss — negative log-probs produce positive loss
# ---------------------------------------------------------------------------


def test_sft_loss_positive():
    trainer = CPOTrainer()
    B, T = 4, 8
    lp = torch.full((B, T), -2.0)  # strictly negative log-probs
    mask = torch.ones(B, T)
    loss = trainer.sft_loss(lp, mask)
    assert loss.item() > 0, "SFT loss must be positive for negative log-probs"


# ---------------------------------------------------------------------------
# 5. sft_loss — longer (fuller) mask gives lower per-token loss
# ---------------------------------------------------------------------------


def test_sft_loss_mask_respected():
    """With uniform log-probs the loss value should be the same regardless
    of how many valid tokens are present (mean is invariant to count).
    But a *partial* mask that omits bad (very negative) tokens should
    yield a better (lower) loss than including them."""
    trainer = CPOTrainer()
    B, T = 2, 8
    lp = torch.full((B, T), -1.0)
    lp[:, -2:] = -10.0  # bad tokens at the end

    full_mask = torch.ones(B, T)
    short_mask = torch.ones(B, T)
    short_mask[:, -2:] = 0.0

    loss_full = trainer.sft_loss(lp, full_mask)
    loss_short = trainer.sft_loss(lp, short_mask)

    # Excluding the bad tokens should give a lower (less negative) NLL
    assert loss_short.item() < loss_full.item(), (
        "Masking out very-negative tokens should lower the SFT loss"
    )


# ---------------------------------------------------------------------------
# 6. label_smoothing changes sft_loss
# ---------------------------------------------------------------------------


def test_label_smoothing_changes_loss():
    B, T = 4, 8
    lp = torch.full((B, T), -2.0)
    mask = torch.ones(B, T)

    trainer_no_smooth = CPOTrainer(CPOConfig(label_smoothing=0.0))
    trainer_smooth = CPOTrainer(CPOConfig(label_smoothing=0.1))

    loss_plain = trainer_no_smooth.sft_loss(lp, mask)
    loss_smooth = trainer_smooth.sft_loss(lp, mask)

    assert not torch.isclose(loss_plain, loss_smooth), (
        "Label smoothing should change the SFT loss value"
    )


# ---------------------------------------------------------------------------
# 7. preference_loss — scalar
# ---------------------------------------------------------------------------


def test_preference_loss_scalar():
    trainer = CPOTrainer()
    batch = _make_batch()
    loss = trainer.preference_loss(batch)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() > 0  # -logsigmoid(positive) is positive


# ---------------------------------------------------------------------------
# 8. preference_loss direction — chosen >> rejected lowers loss
# ---------------------------------------------------------------------------


def test_preference_loss_direction():
    """When chosen is clearly better than rejected the preference loss
    should be lower than when they are equal or reversed."""
    trainer = CPOTrainer(CPOConfig(beta=1.0))

    # Chosen clearly better
    good_batch = _make_batch(chosen_scale=-0.5, rejected_scale=-5.0)
    # Chosen clearly worse (reversed)
    bad_batch = _make_batch(chosen_scale=-5.0, rejected_scale=-0.5)

    loss_good = trainer.preference_loss(good_batch)
    loss_bad = trainer.preference_loss(bad_batch)

    assert loss_good.item() < loss_bad.item(), (
        "Preference loss should be lower when chosen is clearly better"
    )


# ---------------------------------------------------------------------------
# 9. total_loss — required keys present
# ---------------------------------------------------------------------------


def test_total_loss_keys():
    trainer = CPOTrainer()
    batch = _make_batch()
    out = trainer.total_loss(batch)

    required = {"loss", "sft_loss", "pref_loss", "log_ratio_mean", "reward_accuracy"}
    assert required <= out.keys(), f"Missing keys: {required - out.keys()}"


# ---------------------------------------------------------------------------
# 10. total_loss — all values finite
# ---------------------------------------------------------------------------


def test_total_loss_finite():
    trainer = CPOTrainer()
    batch = _make_batch()
    out = trainer.total_loss(batch)

    for key, val in out.items():
        assert torch.isfinite(val), f"total_loss['{key}'] is not finite: {val}"


# ---------------------------------------------------------------------------
# 11. sft_weight=0 → loss equals preference loss only
# ---------------------------------------------------------------------------


def test_sft_weight_zero():
    """When sft_weight is zero the total loss must equal the preference loss."""
    trainer = CPOTrainer(CPOConfig(sft_weight=0.0))
    batch = _make_batch()
    out = trainer.total_loss(batch)

    assert torch.isclose(out["loss"], out["pref_loss"], atol=1e-6), (
        "With sft_weight=0, total loss must equal preference loss"
    )


# ---------------------------------------------------------------------------
# 12. reward_accuracy in [0, 1]
# ---------------------------------------------------------------------------


def test_reward_accuracy_range():
    trainer = CPOTrainer()
    batch = _make_batch()
    out = trainer.total_loss(batch)
    acc = out["reward_accuracy"].item()
    assert 0.0 <= acc <= 1.0, f"reward_accuracy {acc} outside [0, 1]"


# ---------------------------------------------------------------------------
# 13. reward_accuracy ≈ 1 when chosen is clearly better
# ---------------------------------------------------------------------------


def test_reward_accuracy_perfect():
    """When chosen log-probs are much higher than rejected, accuracy should be 1."""
    trainer = CPOTrainer()
    batch = _make_batch(chosen_scale=-0.1, rejected_scale=-10.0)
    out = trainer.total_loss(batch)
    assert out["reward_accuracy"].item() == pytest.approx(1.0), (
        "reward_accuracy should be 1.0 when chosen is clearly better"
    )


# ---------------------------------------------------------------------------
# 14. gradient flows through total_loss
# ---------------------------------------------------------------------------


def test_gradient_flows():
    """Backward pass must produce finite, non-zero gradients."""
    trainer = CPOTrainer()
    B, T_w, T_l = 2, 6, 5
    chosen_lp = (torch.randn(B, T_w) - 1.0).requires_grad_(True)
    rejected_lp = (torch.randn(B, T_l) - 1.0).requires_grad_(True)
    mask_w = torch.ones(B, T_w)
    mask_l = torch.ones(B, T_l)

    batch = CPOBatch(
        chosen_log_probs=chosen_lp,
        rejected_log_probs=rejected_lp,
        chosen_mask=mask_w,
        rejected_mask=mask_l,
    )

    out = trainer.total_loss(batch)
    out["loss"].backward()

    assert chosen_lp.grad is not None, "No gradient on chosen_log_probs"
    assert rejected_lp.grad is not None, "No gradient on rejected_log_probs"
    assert torch.isfinite(chosen_lp.grad).all(), "Non-finite gradient on chosen"
    assert torch.isfinite(rejected_lp.grad).all(), "Non-finite gradient on rejected"
    assert chosen_lp.grad.abs().sum() > 0, "Zero gradient on chosen_log_probs"
    assert rejected_lp.grad.abs().sum() > 0, "Zero gradient on rejected_log_probs"


# ---------------------------------------------------------------------------
# 15. ALIGNMENT_REGISTRY contains "cpo" → CPOTrainer
# ---------------------------------------------------------------------------


def test_registry_entry():
    assert "cpo" in ALIGNMENT_REGISTRY, "'cpo' not found in ALIGNMENT_REGISTRY"
    assert ALIGNMENT_REGISTRY["cpo"] is CPOTrainer, (
        "ALIGNMENT_REGISTRY['cpo'] does not point to CPOTrainer"
    )


# ---------------------------------------------------------------------------
# Integration test — B=4, T_w=12, T_l=10, full forward + backward
# ---------------------------------------------------------------------------


def test_integration_forward_backward():
    """Integration: realistic batch dimensions, full forward + backward pass."""
    B, T_w, T_l = 4, 12, 10
    cfg = CPOConfig(beta=0.1, sft_weight=1.0, label_smoothing=0.05)
    trainer = CPOTrainer(cfg)

    torch.manual_seed(42)
    chosen_lp = (torch.randn(B, T_w) - 1.5).requires_grad_(True)
    rejected_lp = (torch.randn(B, T_l) - 2.5).requires_grad_(True)

    # Realistic masks: last 1-2 tokens of some sequences are padding
    chosen_mask = torch.ones(B, T_w)
    chosen_mask[1, -2:] = 0.0
    chosen_mask[3, -1:] = 0.0
    rejected_mask = torch.ones(B, T_l)
    rejected_mask[0, -1:] = 0.0

    batch = CPOBatch(
        chosen_log_probs=chosen_lp,
        rejected_log_probs=rejected_lp,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )

    out = trainer.total_loss(batch)

    # All output values must be finite scalars
    for key, val in out.items():
        assert torch.isfinite(val), f"[integration] out['{key}'] not finite: {val}"

    # Backward pass
    out["loss"].backward()

    assert chosen_lp.grad is not None
    assert rejected_lp.grad is not None
    assert torch.isfinite(chosen_lp.grad).all(), "[integration] chosen grad not finite"
    assert torch.isfinite(rejected_lp.grad).all(), "[integration] rejected grad not finite"

    # Statistics (no-grad)
    stats = trainer.statistics(batch)
    for k, v in stats.items():
        assert isinstance(v, float), f"statistics['{k}'] should be float"
        assert torch.isfinite(torch.tensor(v)), f"statistics['{k}'] not finite"

    assert 0.0 <= stats["reward_accuracy"] <= 1.0
