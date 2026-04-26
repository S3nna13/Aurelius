"""Unit and integration tests for src/training/ipo_trainer.py.

Coverage:
    1.  test_config_defaults              — IPOConfig defaults: tau=0.1, eps=1e-8
    2.  test_sequence_mean_log_prob_shape — returns shape [B]
    3.  test_sequence_mean_log_prob_masked — masked tokens excluded from mean
    4.  test_compute_h_shape              — returns shape [B]
    5.  test_compute_h_zero               — policy == reference → h = 0
    6.  test_compute_h_positive           — chosen better than ref → positive h
    7.  test_total_loss_keys              — dict has required keys
    8.  test_total_loss_scalar            — loss tensor is 0-dimensional
    9.  test_loss_at_target               — h == 1/(2*tau) everywhere → loss = 0
    10. test_loss_symmetric               — squared deviation, not directional
    11. test_reward_accuracy_range        — reward_accuracy in [0, 1]
    12. test_tau_effect                   — smaller tau → smaller target
    13. test_gradient_flows               — backward() produces non-None grads
    14. test_statistics_keys              — statistics() returns all expected keys
    15. test_statistics_types             — all values are plain Python floats
    16. test_registry_entry              — TRAINING_REGISTRY["ipo"] is IPOTrainer
    Integration:
        test_integration_forward_backward — B=4, T_w=12, T_l=10, finite loss + backward
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training import TRAINING_REGISTRY
from src.training.ipo_trainer import IPOBatch, IPOConfig, IPOTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 2,
    T_w: int = 8,
    T_l: int = 6,
    *,
    chosen_offset: float = 0.0,
    rejected_offset: float = 0.0,
    requires_grad: bool = False,
) -> IPOBatch:
    """Build a synthetic IPOBatch with uniform log-probs and full masks."""
    chosen_lp = torch.full((B, T_w), -1.0 + chosen_offset, requires_grad=requires_grad)
    rejected_lp = torch.full((B, T_l), -2.0 + rejected_offset, requires_grad=requires_grad)
    chosen_ref_lp = torch.full((B, T_w), -1.0)
    rejected_ref_lp = torch.full((B, T_l), -2.0)
    chosen_mask = torch.ones(B, T_w)
    rejected_mask = torch.ones(B, T_l)
    return IPOBatch(
        chosen_log_probs=chosen_lp,
        rejected_log_probs=rejected_lp,
        chosen_ref_log_probs=chosen_ref_lp,
        rejected_ref_log_probs=rejected_ref_lp,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = IPOConfig()
    assert cfg.tau == pytest.approx(0.1)
    assert cfg.eps == pytest.approx(1e-8)


# ---------------------------------------------------------------------------
# 2. test_sequence_mean_log_prob_shape
# ---------------------------------------------------------------------------


def test_sequence_mean_log_prob_shape():
    trainer = IPOTrainer()
    B, T = 4, 10
    lp = torch.randn(B, T)
    mask = torch.ones(B, T)
    out = trainer.sequence_mean_log_prob(lp, mask)
    assert out.shape == (B,), f"expected shape ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_sequence_mean_log_prob_masked
# ---------------------------------------------------------------------------


def test_sequence_mean_log_prob_masked():
    """Only unmasked tokens should contribute to the mean."""
    trainer = IPOTrainer()
    B, T = 2, 6
    # First half of tokens are valid, second half are masked out.
    lp = torch.ones(B, T)  # all 1.0
    lp[:, T // 2 :] = 999.0  # large value — should be masked away
    mask = torch.zeros(B, T)
    mask[:, : T // 2] = 1.0

    out = trainer.sequence_mean_log_prob(lp, mask)
    # Mean of the 3 valid tokens (all 1.0) should be 1.0 for every sequence.
    assert torch.allclose(out, torch.ones(B)), f"got {out}"


# ---------------------------------------------------------------------------
# 4. test_compute_h_shape
# ---------------------------------------------------------------------------


def test_compute_h_shape():
    trainer = IPOTrainer()
    batch = _make_batch(B=3)
    h = trainer.compute_h(batch)
    assert h.shape == (3,), f"expected shape (3,), got {h.shape}"


# ---------------------------------------------------------------------------
# 5. test_compute_h_zero — policy identical to reference → h = 0
# ---------------------------------------------------------------------------


def test_compute_h_zero():
    trainer = IPOTrainer()
    B, T_w, T_l = 4, 8, 6
    # Policy log-probs == reference log-probs for both chosen and rejected.
    lp_chosen = torch.randn(B, T_w)
    lp_rejected = torch.randn(B, T_l)
    mask_w = torch.ones(B, T_w)
    mask_l = torch.ones(B, T_l)
    batch = IPOBatch(
        chosen_log_probs=lp_chosen,
        rejected_log_probs=lp_rejected,
        chosen_ref_log_probs=lp_chosen.clone(),  # identical to policy
        rejected_ref_log_probs=lp_rejected.clone(),
        chosen_mask=mask_w,
        rejected_mask=mask_l,
    )
    h = trainer.compute_h(batch)
    assert torch.allclose(h, torch.zeros(B), atol=1e-6), f"expected zeros, got {h}"


# ---------------------------------------------------------------------------
# 6. test_compute_h_positive — chosen improved more than rejected → h > 0
# ---------------------------------------------------------------------------


def test_compute_h_positive():
    trainer = IPOTrainer()
    B, T_w, T_l = 2, 5, 5
    # Policy chosen = ref + 1.0  (improved chosen)
    # Policy rejected = ref       (no change in rejected)
    # → h = +1.0 per sequence
    ref_chosen = torch.full((B, T_w), -1.0)
    ref_rejected = torch.full((B, T_l), -2.0)
    batch = IPOBatch(
        chosen_log_probs=ref_chosen + 1.0,
        rejected_log_probs=ref_rejected.clone(),
        chosen_ref_log_probs=ref_chosen,
        rejected_ref_log_probs=ref_rejected,
        chosen_mask=torch.ones(B, T_w),
        rejected_mask=torch.ones(B, T_l),
    )
    h = trainer.compute_h(batch)
    assert (h > 0).all(), f"expected all positive h, got {h}"


# ---------------------------------------------------------------------------
# 7. test_total_loss_keys
# ---------------------------------------------------------------------------


def test_total_loss_keys():
    trainer = IPOTrainer()
    out = trainer.total_loss(_make_batch())
    required = {"loss", "h_mean", "target", "reward_accuracy"}
    assert required <= out.keys(), f"missing keys: {required - out.keys()}"


# ---------------------------------------------------------------------------
# 8. test_total_loss_scalar
# ---------------------------------------------------------------------------


def test_total_loss_scalar():
    trainer = IPOTrainer()
    out = trainer.total_loss(_make_batch())
    assert out["loss"].dim() == 0, f"loss should be scalar, got shape {out['loss'].shape}"


# ---------------------------------------------------------------------------
# 9. test_loss_at_target — h exactly equals 1/(2*tau) → loss = 0
# ---------------------------------------------------------------------------


def test_loss_at_target():
    cfg = IPOConfig(tau=0.1)
    trainer = IPOTrainer(cfg)
    target_val = 1.0 / (2.0 * cfg.tau)  # = 5.0

    B, T_w, T_l = 3, 6, 6
    # Construct a batch where:
    #   (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected) = target_val
    # Simple setup: chosen delta = target_val, rejected delta = 0.
    ref_chosen = torch.zeros(B, T_w)
    ref_rejected = torch.zeros(B, T_l)
    policy_chosen = ref_chosen + target_val  # per-token shift → mean also = target_val
    policy_rejected = ref_rejected.clone()

    batch = IPOBatch(
        chosen_log_probs=policy_chosen,
        rejected_log_probs=policy_rejected,
        chosen_ref_log_probs=ref_chosen,
        rejected_ref_log_probs=ref_rejected,
        chosen_mask=torch.ones(B, T_w),
        rejected_mask=torch.ones(B, T_l),
    )
    out = trainer.total_loss(batch)
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-5), (
        f"expected loss ~0 at target, got {out['loss'].item()}"
    )


# ---------------------------------------------------------------------------
# 10. test_loss_symmetric — symmetric deviation has same loss regardless of sign
# ---------------------------------------------------------------------------


def test_loss_symmetric():
    """(h - target)^2 = (target - h)^2: loss is the same for +δ and -δ offsets."""
    cfg = IPOConfig(tau=0.2)
    trainer = IPOTrainer(cfg)
    target_val = 1.0 / (2.0 * cfg.tau)

    B, T = 2, 5
    delta = 0.7

    def _batch_with_gap(gap: float) -> IPOBatch:
        ref = torch.zeros(B, T)
        pol = ref + gap
        return IPOBatch(
            chosen_log_probs=pol,
            rejected_log_probs=ref.clone(),
            chosen_ref_log_probs=ref.clone(),
            rejected_ref_log_probs=ref.clone(),
            chosen_mask=torch.ones(B, T),
            rejected_mask=torch.ones(B, T),
        )

    loss_above = trainer.total_loss(_batch_with_gap(target_val + delta))["loss"].item()
    loss_below = trainer.total_loss(_batch_with_gap(target_val - delta))["loss"].item()
    assert loss_above == pytest.approx(loss_below, rel=1e-5), (
        f"expected symmetric loss, got above={loss_above}, below={loss_below}"
    )


# ---------------------------------------------------------------------------
# 11. test_reward_accuracy_range
# ---------------------------------------------------------------------------


def test_reward_accuracy_range():
    trainer = IPOTrainer()
    batch = _make_batch(B=8)
    out = trainer.total_loss(batch)
    acc = out["reward_accuracy"].item()
    assert 0.0 <= acc <= 1.0, f"reward_accuracy {acc} out of [0, 1]"


# ---------------------------------------------------------------------------
# 12. test_tau_effect — smaller tau → smaller target gap
# ---------------------------------------------------------------------------


def test_tau_effect():
    small_tau = IPOTrainer(IPOConfig(tau=0.05))
    large_tau = IPOTrainer(IPOConfig(tau=0.5))

    batch = _make_batch()
    target_small = small_tau.total_loss(batch)["target"].item()
    target_large = large_tau.total_loss(batch)["target"].item()

    # 1/(2*0.05) = 10  vs  1/(2*0.5) = 1
    assert target_small > target_large, (
        f"smaller tau should give larger target gap; got small={target_small}, large={target_large}"
    )


# ---------------------------------------------------------------------------
# 13. test_gradient_flows
# ---------------------------------------------------------------------------


def test_gradient_flows():
    trainer = IPOTrainer()
    batch = _make_batch(requires_grad=True)
    out = trainer.total_loss(batch)
    out["loss"].backward()

    assert batch.chosen_log_probs.grad is not None, "grad missing for chosen_log_probs"
    assert batch.rejected_log_probs.grad is not None, "grad missing for rejected_log_probs"
    assert not torch.isnan(batch.chosen_log_probs.grad).any(), "NaN in chosen grad"
    assert not torch.isnan(batch.rejected_log_probs.grad).any(), "NaN in rejected grad"


# ---------------------------------------------------------------------------
# 14. test_statistics_keys
# ---------------------------------------------------------------------------


def test_statistics_keys():
    trainer = IPOTrainer()
    stats = trainer.statistics(_make_batch())
    required = {"h_mean", "h_std", "reward_accuracy", "chosen_logp_mean", "rejected_logp_mean"}
    assert required <= stats.keys(), f"missing keys: {required - stats.keys()}"


# ---------------------------------------------------------------------------
# 15. test_statistics_types
# ---------------------------------------------------------------------------


def test_statistics_types():
    trainer = IPOTrainer()
    stats = trainer.statistics(_make_batch())
    for key, val in stats.items():
        assert isinstance(val, float), f"stats['{key}'] should be float, got {type(val)}"


# ---------------------------------------------------------------------------
# 16. test_registry_entry
# ---------------------------------------------------------------------------


def test_registry_entry():
    assert "ipo" in TRAINING_REGISTRY, "'ipo' not found in TRAINING_REGISTRY"
    assert TRAINING_REGISTRY["ipo"] is IPOTrainer, (
        f"TRAINING_REGISTRY['ipo'] is {TRAINING_REGISTRY['ipo']}, expected IPOTrainer"
    )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_integration_forward_backward():
    """Full forward + backward pass with realistic tensor shapes."""
    B, T_w, T_l = 4, 12, 10
    cfg = IPOConfig(tau=0.1)
    trainer = IPOTrainer(cfg)

    torch.manual_seed(42)
    chosen_lp = torch.randn(B, T_w, requires_grad=True)
    rejected_lp = torch.randn(B, T_l, requires_grad=True)
    chosen_ref_lp = torch.randn(B, T_w)
    rejected_ref_lp = torch.randn(B, T_l)

    # Realistic masks: last 2 tokens in each sequence are padding
    chosen_mask = torch.ones(B, T_w)
    chosen_mask[:, -2:] = 0.0
    rejected_mask = torch.ones(B, T_l)
    rejected_mask[:, -2:] = 0.0

    batch = IPOBatch(
        chosen_log_probs=chosen_lp,
        rejected_log_probs=rejected_lp,
        chosen_ref_log_probs=chosen_ref_lp,
        rejected_ref_log_probs=rejected_ref_lp,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )

    out = trainer.total_loss(batch)

    # Loss must be a finite scalar.
    loss_val = out["loss"].item()
    assert math.isfinite(loss_val), f"loss is not finite: {loss_val}"
    assert out["loss"].dim() == 0

    # Target sanity-check: 1/(2*0.1) = 5.0
    assert out["target"].item() == pytest.approx(5.0, abs=1e-6)

    # Backward pass should complete without error.
    out["loss"].backward()

    # Gradients must be present and finite.
    assert chosen_lp.grad is not None
    assert rejected_lp.grad is not None
    assert torch.isfinite(chosen_lp.grad).all(), "non-finite grad in chosen_lp"
    assert torch.isfinite(rejected_lp.grad).all(), "non-finite grad in rejected_lp"

    # Statistics method returns sensible values.
    stats = trainer.statistics(batch)
    assert math.isfinite(stats["h_mean"])
    assert math.isfinite(stats["h_std"])
    assert 0.0 <= stats["reward_accuracy"] <= 1.0
