"""Tests for src/alignment/orpo_trainer.py — ORPO Trainer.

Covers:
    1.  test_config_defaults
    2.  test_sequence_log_prob_masked
    3.  test_sequence_log_prob_shape
    4.  test_log_odds_range
    5.  test_log_odds_monotone
    6.  test_sft_loss_positive
    7.  test_or_loss_positive
    8.  test_or_loss_preferred_chosen
    9.  test_or_loss_scalar
    10. test_total_loss_keys
    11. test_total_loss_finite
    12. test_statistics_reward_accuracy_range
    13. test_statistics_chosen_higher
    14. test_gradient_flows
    Integration: test_integration_forward_backward
"""

from __future__ import annotations

import torch

from src.alignment.orpo_trainer import ORPOBatch, ORPOConfig, ORPOTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    B: int = 2,
    T_w: int = 8,
    T_l: int = 6,
    seed: int = 42,
    requires_grad: bool = False,
) -> ORPOBatch:
    """Build a random ORPOBatch with valid log-probs in (-5, 0)."""
    torch.manual_seed(seed)
    # Log-probs must be < 0 for log_odds to be valid (they represent log p, p in (0,1))
    chosen_lp = -torch.rand(B, T_w) * 4.0 - 0.1  # in (-4.1, -0.1)
    rejected_lp = -torch.rand(B, T_l) * 4.0 - 0.1
    if requires_grad:
        chosen_lp = chosen_lp.requires_grad_(True)
        rejected_lp = rejected_lp.requires_grad_(True)
    chosen_mask = torch.ones(B, T_w)
    rejected_mask = torch.ones(B, T_l)
    return ORPOBatch(
        chosen_log_probs=chosen_lp,
        rejected_log_probs=rejected_lp,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """ORPOConfig defaults: lambda_or=0.1, eps=1e-8."""
    cfg = ORPOConfig()
    assert cfg.lambda_or == 0.1
    assert cfg.eps == 1e-8


# ---------------------------------------------------------------------------
# 2. test_sequence_log_prob_masked
# ---------------------------------------------------------------------------


def test_sequence_log_prob_masked():
    """Padding tokens (mask=0) must not affect sequence_log_prob output."""
    trainer = ORPOTrainer()
    torch.manual_seed(0)
    B, T = 3, 10
    log_probs = -torch.rand(B, T) * 3.0 - 0.1

    # Full mask — all tokens valid
    full_mask = torch.ones(B, T)
    trainer.sequence_log_prob(log_probs, full_mask)

    # Partial mask — zero out the last 3 positions AND set those log_probs
    # to something wildly different so that if they were counted the result
    # would change noticeably.
    partial_mask = torch.ones(B, T)
    partial_mask[:, -3:] = 0.0
    lp_modified = log_probs.clone()
    lp_modified[:, -3:] = -100.0  # extreme values that would shift the mean

    result_partial = trainer.sequence_log_prob(lp_modified, partial_mask)

    # result_partial should equal sequence_log_prob over the first 7 tokens of log_probs
    expected = log_probs[:, :7].mean(dim=-1)
    assert torch.allclose(result_partial, expected, atol=1e-5), (
        f"Masked tokens polluted the result: {result_partial} vs {expected}"
    )


# ---------------------------------------------------------------------------
# 3. test_sequence_log_prob_shape
# ---------------------------------------------------------------------------


def test_sequence_log_prob_shape():
    """sequence_log_prob must return a 1-D tensor of shape [B]."""
    trainer = ORPOTrainer()
    B, T = 4, 12
    log_probs = -torch.rand(B, T) - 0.1
    mask = torch.ones(B, T)
    out = trainer.sequence_log_prob(log_probs, mask)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. test_log_odds_range
# ---------------------------------------------------------------------------


def test_log_odds_range():
    """log_odds must return finite values for log-probs strictly in (-inf, 0)."""
    trainer = ORPOTrainer()
    # Sweep across a broad range of valid log-probs
    seq_lp = torch.linspace(-10.0, -0.01, steps=100)
    lo = trainer.log_odds(seq_lp)
    assert torch.isfinite(lo).all(), f"Non-finite log_odds: {lo[~torch.isfinite(lo)]}"


# ---------------------------------------------------------------------------
# 5. test_log_odds_monotone
# ---------------------------------------------------------------------------


def test_log_odds_monotone():
    """Higher log-prob => higher log_odds (monotone increasing)."""
    trainer = ORPOTrainer()
    # Strictly increasing log-probs (all < 0)
    seq_lp = torch.tensor([-5.0, -3.0, -1.0, -0.5, -0.1])
    lo = trainer.log_odds(seq_lp)
    diffs = lo[1:] - lo[:-1]
    assert (diffs > 0).all(), f"log_odds is not monotonically increasing: {lo}"


# ---------------------------------------------------------------------------
# 6. test_sft_loss_positive
# ---------------------------------------------------------------------------


def test_sft_loss_positive():
    """sft_loss must be > 0 because it is the negative of negative log-probs."""
    trainer = ORPOTrainer()
    batch = _make_batch()
    loss = trainer.sft_loss(batch.chosen_log_probs, batch.chosen_mask)
    assert loss.item() > 0.0, f"Expected sft_loss > 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 7. test_or_loss_positive
# ---------------------------------------------------------------------------


def test_or_loss_positive():
    """odds_ratio_loss must be > 0 for arbitrary random inputs."""
    trainer = ORPOTrainer()
    batch = _make_batch(seed=7)
    or_loss = trainer.odds_ratio_loss(batch)
    assert or_loss.item() > 0.0, f"Expected or_loss > 0, got {or_loss.item()}"


# ---------------------------------------------------------------------------
# 8. test_or_loss_preferred_chosen
# ---------------------------------------------------------------------------


def test_or_loss_preferred_chosen():
    """When chosen_log_probs >> rejected_log_probs, or_loss is near 0."""
    trainer = ORPOTrainer()
    B, T = 4, 8
    # Chosen has log-probs close to 0 (high probability)
    chosen_lp = torch.full((B, T), -0.05)
    # Rejected has log-probs very negative (low probability)
    rejected_lp = torch.full((B, T), -8.0)
    mask_w = torch.ones(B, T)
    mask_l = torch.ones(B, T)

    batch_good = ORPOBatch(chosen_lp, rejected_lp, mask_w, mask_l)

    # And the reverse: chosen worse than rejected
    batch_bad = ORPOBatch(rejected_lp, chosen_lp, mask_l, mask_w)

    loss_good = trainer.odds_ratio_loss(batch_good).item()
    loss_bad = trainer.odds_ratio_loss(batch_bad).item()

    assert loss_good < loss_bad, (
        f"Expected smaller loss when chosen dominates: {loss_good} vs {loss_bad}"
    )


# ---------------------------------------------------------------------------
# 9. test_or_loss_scalar
# ---------------------------------------------------------------------------


def test_or_loss_scalar():
    """odds_ratio_loss must return a 0-dimensional (scalar) tensor."""
    trainer = ORPOTrainer()
    batch = _make_batch()
    or_loss = trainer.odds_ratio_loss(batch)
    assert or_loss.dim() == 0, f"Expected scalar, got shape {or_loss.shape}"


# ---------------------------------------------------------------------------
# 10. test_total_loss_keys
# ---------------------------------------------------------------------------


def test_total_loss_keys():
    """total_loss must return a dict with exactly the required keys."""
    trainer = ORPOTrainer()
    batch = _make_batch()
    out = trainer.total_loss(batch)
    required = {"loss", "sft_loss", "or_loss", "log_odds_ratio"}
    assert required <= set(out.keys()), f"Missing keys: {required - set(out.keys())}"


# ---------------------------------------------------------------------------
# 11. test_total_loss_finite
# ---------------------------------------------------------------------------


def test_total_loss_finite():
    """All tensors in total_loss output must be finite."""
    trainer = ORPOTrainer()
    batch = _make_batch()
    out = trainer.total_loss(batch)
    for key, val in out.items():
        assert torch.isfinite(val).all(), f"Non-finite value for key '{key}': {val}"


# ---------------------------------------------------------------------------
# 12. test_statistics_reward_accuracy_range
# ---------------------------------------------------------------------------


def test_statistics_reward_accuracy_range():
    """reward_accuracy in statistics() must be in [0, 1]."""
    trainer = ORPOTrainer()
    batch = _make_batch(B=8)
    stats = trainer.statistics(batch)
    acc = stats["reward_accuracy"]
    assert 0.0 <= acc <= 1.0, f"reward_accuracy={acc} out of [0, 1]"


# ---------------------------------------------------------------------------
# 13. test_statistics_chosen_higher
# ---------------------------------------------------------------------------


def test_statistics_chosen_higher():
    """When chosen log-probs dominate, reward_accuracy should equal 1.0."""
    trainer = ORPOTrainer()
    B, T = 6, 8
    # Chosen: high probability (log-prob close to 0)
    chosen_lp = torch.full((B, T), -0.05)
    # Rejected: low probability (very negative log-prob)
    rejected_lp = torch.full((B, T), -8.0)
    batch = ORPOBatch(
        chosen_log_probs=chosen_lp,
        rejected_log_probs=rejected_lp,
        chosen_mask=torch.ones(B, T),
        rejected_mask=torch.ones(B, T),
    )
    stats = trainer.statistics(batch)
    assert stats["reward_accuracy"] == 1.0, (
        f"Expected reward_accuracy=1.0, got {stats['reward_accuracy']}"
    )


# ---------------------------------------------------------------------------
# 14. test_gradient_flows
# ---------------------------------------------------------------------------


def test_gradient_flows():
    """backward() on total_loss must produce finite gradients on log-probs."""
    trainer = ORPOTrainer()
    batch = _make_batch(requires_grad=True)
    out = trainer.total_loss(batch)
    out["loss"].backward()

    assert batch.chosen_log_probs.grad is not None, "No gradient on chosen_log_probs"
    assert batch.rejected_log_probs.grad is not None, "No gradient on rejected_log_probs"
    assert torch.isfinite(batch.chosen_log_probs.grad).all(), "Non-finite grad on chosen_log_probs"
    assert torch.isfinite(batch.rejected_log_probs.grad).all(), (
        "Non-finite grad on rejected_log_probs"
    )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_integration_forward_backward():
    """Integration: B=4, T_w=12, T_l=10 — run total_loss and backward."""
    torch.manual_seed(99)
    B, T_w, T_l = 4, 12, 10

    config = ORPOConfig(lambda_or=0.1)
    trainer = ORPOTrainer(config)

    chosen_lp = (-torch.rand(B, T_w) * 3.0 - 0.1).requires_grad_(True)
    rejected_lp = (-torch.rand(B, T_l) * 3.0 - 0.1).requires_grad_(True)
    chosen_mask = torch.ones(B, T_w)
    rejected_mask = torch.ones(B, T_l)
    # Simulate a shorter rejected response with some padding
    rejected_mask[:, -2:] = 0.0

    batch = ORPOBatch(chosen_lp, rejected_lp, chosen_mask, rejected_mask)

    # Forward
    out = trainer.total_loss(batch)

    # Keys present
    assert "loss" in out
    assert "sft_loss" in out
    assert "or_loss" in out
    assert "log_odds_ratio" in out

    # All finite
    for key, val in out.items():
        assert torch.isfinite(val).all(), f"Non-finite value in integration test for '{key}': {val}"

    # Combined loss = sft_loss + lambda_or * or_loss
    expected_total = out["sft_loss"] + config.lambda_or * out["or_loss"]
    assert torch.isclose(out["loss"], expected_total, atol=1e-5), (
        f"total loss mismatch: {out['loss'].item()} vs {expected_total.item()}"
    )

    # Backward pass
    out["loss"].backward()

    assert chosen_lp.grad is not None, "No gradient on chosen_log_probs"
    assert rejected_lp.grad is not None, "No gradient on rejected_log_probs"
    assert torch.isfinite(chosen_lp.grad).all(), "Non-finite grad on chosen"
    assert torch.isfinite(rejected_lp.grad).all(), "Non-finite grad on rejected"

    # Statistics
    stats = trainer.statistics(batch)
    assert "chosen_mean_logp" in stats
    assert "rejected_mean_logp" in stats
    assert "log_odds_ratio_mean" in stats
    assert "reward_accuracy" in stats
    assert 0.0 <= stats["reward_accuracy"] <= 1.0
