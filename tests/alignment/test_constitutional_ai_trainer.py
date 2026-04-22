"""Unit + integration tests for src/alignment/constitutional_ai_trainer.py.

17 tests total: 16 unit tests + 1 integration test.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.constitutional_ai_trainer import (
    CAIBatch,
    CAIConfig,
    CAIExample,
    CAIPrinciple,
    CAITrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 4,
    T_orig: int = 8,
    T_rev: int = 10,
    T_crit: int = 6,
    lp_orig: float = -1.5,
    lp_rev: float = -1.0,
    lp_crit: float = -2.0,
    requires_grad: bool = False,
) -> CAIBatch:
    """Build a synthetic CAIBatch with uniform log-probs and full masks."""
    orig = torch.full((B, T_orig), lp_orig).requires_grad_(requires_grad)
    rev = torch.full((B, T_rev), lp_rev).requires_grad_(requires_grad)
    crit = torch.full((B, T_crit), lp_crit).requires_grad_(requires_grad)
    return CAIBatch(
        original_log_probs=orig,
        revised_log_probs=rev,
        critique_log_probs=crit,
        original_mask=torch.ones(B, T_orig),
        revised_mask=torch.ones(B, T_rev),
        critique_mask=torch.ones(B, T_crit),
    )


def _make_examples(n: int = 4, with_violation: bool = True) -> list[CAIExample]:
    """Generate a list of synthetic CAIExample objects."""
    examples = []
    for i in range(n):
        examples.append(
            CAIExample(
                original_response="This is a response " * 5,
                critiques=[
                    "This text may be harmful and violates the harmless principle.",
                    "Still needs revision for honesty.",
                ],
                revised_responses=[
                    "A revised response that is safer.",
                    "A further revised response that is both safe and honest.",
                ],
                principle_violated="harmless" if (with_violation and i % 2 == 0) else None,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CAIConfig()
    assert cfg.n_revisions == 2
    assert cfg.revision_weight == pytest.approx(1.0)
    assert cfg.critique_weight == pytest.approx(0.5)
    assert cfg.max_critique_tokens == 256
    # Default principles should include all three CAIPrinciple values
    assert set(cfg.principles) == {p.value for p in CAIPrinciple}


# ---------------------------------------------------------------------------
# 2. test_sft_loss_positive — negative log-probs → positive loss
# ---------------------------------------------------------------------------


def test_sft_loss_positive():
    trainer = CAITrainer()
    B, T = 4, 8
    lp = torch.full((B, T), -2.0)
    mask = torch.ones(B, T)
    loss = trainer.sft_loss(lp, mask)
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 3. test_sft_loss_masked — masked tokens excluded from loss
# ---------------------------------------------------------------------------


def test_sft_loss_masked():
    """Tokens zeroed by the mask must not contribute to the NLL."""
    trainer = CAITrainer()
    B, T = 2, 8
    lp = torch.full((B, T), -1.0)
    lp[:, -2:] = -999.0  # poisoned values that must be ignored

    full_mask = torch.ones(B, T)
    short_mask = torch.ones(B, T)
    short_mask[:, -2:] = 0.0

    loss_short = trainer.sft_loss(lp, short_mask)
    loss_full = trainer.sft_loss(lp, full_mask)

    # Masking out the very-negative tokens should lower the loss
    assert loss_short.item() < loss_full.item(), (
        "Masking out highly-negative tokens must reduce the NLL"
    )
    # The short-mask loss should be exactly 1.0 (mean of -(-1.0) = 1.0)
    assert loss_short.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. test_sft_loss_scalar — loss is a 0-d tensor
# ---------------------------------------------------------------------------


def test_sft_loss_scalar():
    trainer = CAITrainer()
    lp = torch.full((4, 8), -1.5)
    mask = torch.ones(4, 8)
    loss = trainer.sft_loss(lp, mask)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 5. test_revision_quality_score_keys
# ---------------------------------------------------------------------------


def test_revision_quality_score_keys():
    trainer = CAITrainer()
    examples = _make_examples()
    scores = trainer.revision_quality_score(examples)
    assert "length_ratio" in scores
    assert "revision_count" in scores
    assert "critique_rate" in scores


# ---------------------------------------------------------------------------
# 6. test_revision_quality_score_ranges
# ---------------------------------------------------------------------------


def test_revision_quality_score_ranges():
    trainer = CAITrainer()
    examples = _make_examples(n=8, with_violation=True)
    scores = trainer.revision_quality_score(examples)

    assert scores["length_ratio"] > 0.0, "length_ratio must be positive"
    assert scores["revision_count"] >= 0.0, "revision_count must be non-negative"
    assert 0.0 <= scores["critique_rate"] <= 1.0, (
        f"critique_rate {scores['critique_rate']} outside [0, 1]"
    )


# ---------------------------------------------------------------------------
# 7. test_total_loss_keys
# ---------------------------------------------------------------------------


def test_total_loss_keys():
    trainer = CAITrainer()
    batch = _make_batch()
    out = trainer.total_loss(batch)
    required = {"loss", "revised_loss", "critique_loss"}
    assert required <= out.keys(), f"Missing keys: {required - out.keys()}"


# ---------------------------------------------------------------------------
# 8. test_total_loss_scalar — all returned tensors are 0-d
# ---------------------------------------------------------------------------


def test_total_loss_scalar():
    trainer = CAITrainer()
    batch = _make_batch()
    out = trainer.total_loss(batch)
    for key, val in out.items():
        assert val.shape == (), f"out['{key}'] is not scalar: shape={val.shape}"


# ---------------------------------------------------------------------------
# 9. test_total_loss_weights — doubling revision_weight doubles revised contrib
# ---------------------------------------------------------------------------


def test_total_loss_weights():
    """Verify that revision_weight linearly scales the revised loss contribution."""
    batch = _make_batch()

    trainer1 = CAITrainer(CAIConfig(revision_weight=1.0, critique_weight=0.0))
    trainer2 = CAITrainer(CAIConfig(revision_weight=2.0, critique_weight=0.0))

    out1 = trainer1.total_loss(batch)
    out2 = trainer2.total_loss(batch)

    # With critique_weight=0, loss == revision_weight * revised_loss
    assert out2["loss"].item() == pytest.approx(2.0 * out1["loss"].item(), rel=1e-5), (
        "Doubling revision_weight must double total loss when critique_weight=0"
    )


# ---------------------------------------------------------------------------
# 10. test_apply_principles_filter_hit
# ---------------------------------------------------------------------------


def test_apply_principles_filter_hit():
    """A critique that mentions 'harmless' should return True."""
    trainer = CAITrainer()
    example = CAIExample(
        original_response="Here is my answer.",
        critiques=["This response is not harmless and may cause harm."],
        revised_responses=["Here is a safer answer."],
        principle_violated="harmless",
    )
    result = trainer.apply_principles_filter(example, ["harmless", "honest"])
    assert result is True, "Filter should return True when a principle is mentioned"


# ---------------------------------------------------------------------------
# 11. test_apply_principles_filter_miss
# ---------------------------------------------------------------------------


def test_apply_principles_filter_miss():
    """Critiques that mention nothing from the principles list → False."""
    trainer = CAITrainer()
    example = CAIExample(
        original_response="Answer.",
        critiques=["This response is somewhat verbose."],
        revised_responses=["A shorter answer."],
        principle_violated=None,
    )
    result = trainer.apply_principles_filter(example, ["harmless", "honest", "helpful"])
    assert result is False, (
        "Filter should return False when no principle appears in critiques"
    )


# ---------------------------------------------------------------------------
# 12. test_statistics_keys
# ---------------------------------------------------------------------------


def test_statistics_keys():
    trainer = CAITrainer()
    batch = _make_batch()
    stats = trainer.statistics(batch)
    required = {"revised_nll", "critique_nll", "loss"}
    assert required <= stats.keys(), f"Missing stats keys: {required - stats.keys()}"


# ---------------------------------------------------------------------------
# 13. test_statistics_finite
# ---------------------------------------------------------------------------


def test_statistics_finite():
    trainer = CAITrainer()
    batch = _make_batch()
    stats = trainer.statistics(batch)
    for key, val in stats.items():
        assert isinstance(val, float), f"stats['{key}'] should be float, got {type(val)}"
        assert math.isfinite(val), f"stats['{key}'] = {val} is not finite"


# ---------------------------------------------------------------------------
# 14. test_gradient_flows — backward on total_loss
# ---------------------------------------------------------------------------


def test_gradient_flows():
    """Backward pass must produce finite, non-zero gradients on log-prob inputs."""
    trainer = CAITrainer()
    B, T_orig, T_rev, T_crit = 2, 8, 10, 6

    torch.manual_seed(0)
    orig = (torch.randn(B, T_orig) - 1.0).requires_grad_(True)
    rev = (torch.randn(B, T_rev) - 1.0).requires_grad_(True)
    crit = (torch.randn(B, T_crit) - 1.0).requires_grad_(True)

    batch = CAIBatch(
        original_log_probs=orig,
        revised_log_probs=rev,
        critique_log_probs=crit,
        original_mask=torch.ones(B, T_orig),
        revised_mask=torch.ones(B, T_rev),
        critique_mask=torch.ones(B, T_crit),
    )

    out = trainer.total_loss(batch)
    out["loss"].backward()

    # revised and critique get gradients (they are in the loss);
    # original only gets gradients when kl_coeff > 0
    assert rev.grad is not None, "No gradient on revised_log_probs"
    assert crit.grad is not None, "No gradient on critique_log_probs"
    assert torch.isfinite(rev.grad).all(), "Non-finite gradient on revised_log_probs"
    assert torch.isfinite(crit.grad).all(), "Non-finite gradient on critique_log_probs"
    assert rev.grad.abs().sum() > 0, "Zero gradient on revised_log_probs"
    assert crit.grad.abs().sum() > 0, "Zero gradient on critique_log_probs"


# ---------------------------------------------------------------------------
# 15. test_kl_coeff_increases_loss — kl_coeff > 0 adds to total loss
# ---------------------------------------------------------------------------


def test_kl_coeff_increases_loss():
    """Enabling the KL regulariser should increase the total loss."""
    trainer = CAITrainer()
    batch = _make_batch()

    out_no_kl = trainer.total_loss(batch, kl_coeff=0.0)
    out_with_kl = trainer.total_loss(batch, kl_coeff=1.0)

    assert out_with_kl["loss"].item() > out_no_kl["loss"].item(), (
        "Adding KL regularisation (kl_coeff=1.0) must increase the total loss"
    )


# ---------------------------------------------------------------------------
# 16. test_registry_entry — ALIGNMENT_REGISTRY["constitutional_ai"] == CAITrainer
# ---------------------------------------------------------------------------


def test_registry_entry():
    assert "constitutional_ai" in ALIGNMENT_REGISTRY, (
        "'constitutional_ai' not found in ALIGNMENT_REGISTRY"
    )
    assert ALIGNMENT_REGISTRY["constitutional_ai"] is CAITrainer, (
        "ALIGNMENT_REGISTRY['constitutional_ai'] does not point to CAITrainer"
    )


# ---------------------------------------------------------------------------
# Integration test — B=4, T=16, full forward + backward, verify finite loss
# ---------------------------------------------------------------------------


def test_integration_forward_backward():
    """Integration: realistic batch dimensions, full forward + backward pass."""
    B, T_orig, T_rev, T_crit = 4, 16, 16, 12

    cfg = CAIConfig(
        n_revisions=2,
        revision_weight=1.0,
        critique_weight=0.5,
        principles=["harmless", "honest", "helpful"],
        max_critique_tokens=256,
    )
    trainer = CAITrainer(cfg)

    torch.manual_seed(42)
    orig = (torch.randn(B, T_orig) - 1.5)
    rev = (torch.randn(B, T_rev) - 1.0).requires_grad_(True)
    crit = (torch.randn(B, T_crit) - 2.0).requires_grad_(True)

    # Realistic masks: some sequences have trailing padding
    orig_mask = torch.ones(B, T_orig)
    rev_mask = torch.ones(B, T_rev)
    rev_mask[1, -3:] = 0.0
    rev_mask[3, -1:] = 0.0
    crit_mask = torch.ones(B, T_crit)
    crit_mask[0, -2:] = 0.0

    batch = CAIBatch(
        original_log_probs=orig,
        revised_log_probs=rev,
        critique_log_probs=crit,
        original_mask=orig_mask,
        revised_mask=rev_mask,
        critique_mask=crit_mask,
    )

    # --- Forward pass ---
    out = trainer.total_loss(batch)

    for key, val in out.items():
        assert val.shape == (), f"[integration] out['{key}'] is not scalar"
        assert torch.isfinite(val), f"[integration] out['{key}'] not finite: {val}"

    assert out["loss"].item() > 0, "[integration] loss must be positive"

    # --- Backward pass ---
    out["loss"].backward()

    assert rev.grad is not None, "[integration] No gradient on revised_log_probs"
    assert crit.grad is not None, "[integration] No gradient on critique_log_probs"
    assert torch.isfinite(rev.grad).all(), "[integration] Non-finite grad on rev"
    assert torch.isfinite(crit.grad).all(), "[integration] Non-finite grad on crit"
    assert rev.grad.abs().sum() > 0, "[integration] Zero gradient on revised"
    assert crit.grad.abs().sum() > 0, "[integration] Zero gradient on critique"

    # --- Statistics (no-grad) ---
    stats = trainer.statistics(batch)
    assert set(stats.keys()) == {"revised_nll", "critique_nll", "loss"}
    for k, v in stats.items():
        assert isinstance(v, float), f"[integration] stats['{k}'] should be float"
        assert math.isfinite(v), f"[integration] stats['{k}'] = {v} not finite"
    assert stats["revised_nll"] > 0
    assert stats["critique_nll"] > 0
    assert stats["loss"] > 0

    # --- Revision quality scoring ---
    examples = _make_examples(n=B, with_violation=True)
    scores = trainer.revision_quality_score(examples)
    assert 0.0 <= scores["critique_rate"] <= 1.0
    assert scores["revision_count"] >= 0.0
    assert scores["length_ratio"] > 0.0

    # --- Principle filter ---
    hit_ex = examples[0]  # has "harmless" in critiques
    assert trainer.apply_principles_filter(hit_ex, cfg.principles) is True
