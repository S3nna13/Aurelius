"""Tests for src/alignment/online_rft.py — Online Rejection Fine-Tuning."""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.online_rft import (
    OnlineRFTConfig,
    OnlineRFTTrainer,
    RFTSample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    is_correct: bool = True,
    reward: float = 1.0,
    prompt_len: int = 3,
    response_len: int = 5,
) -> RFTSample:
    return RFTSample(
        prompt_tokens=list(range(prompt_len)),
        response_tokens=list(range(response_len)),
        is_correct=is_correct,
        reward=reward,
    )


def _make_candidates(
    n: int = 8,
    n_correct: int = 5,
) -> list[RFTSample]:
    """Return n candidates, the first n_correct of which are marked correct."""
    samples = []
    for i in range(n):
        correct = i < n_correct
        # Vary reward slightly so sorting is deterministic
        reward = float(n - i) / n
        samples.append(_make_sample(is_correct=correct, reward=reward))
    return samples


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Default config has n_candidates=8 and filter_strategy='correct_only'."""
    cfg = OnlineRFTConfig()
    assert cfg.n_candidates == 8
    assert cfg.filter_strategy == "correct_only"
    assert cfg.min_keep_ratio == 0.125
    assert cfg.max_keep_ratio == 0.5
    assert cfg.temperature == 0.8
    assert cfg.sft_loss_weight == 1.0
    assert cfg.kl_penalty_weight == 0.1
    assert cfg.top_k_ratio == 0.5


# ---------------------------------------------------------------------------
# 2. test_filter_correct_only
# ---------------------------------------------------------------------------


def test_filter_correct_only():
    """correct_only strategy keeps only samples with is_correct=True."""
    cfg = OnlineRFTConfig(n_candidates=8, filter_strategy="correct_only")
    trainer = OnlineRFTTrainer(cfg)

    candidates = _make_candidates(n=8, n_correct=5)
    kept = trainer.filter_candidates(candidates)

    assert all(s.is_correct for s in kept), "All kept samples must be correct"
    assert len(kept) == 5


# ---------------------------------------------------------------------------
# 3. test_filter_min_keep
# ---------------------------------------------------------------------------


def test_filter_min_keep():
    """If fewer correct than min_keep, top-up by reward to reach min_keep."""
    # n_candidates=8, min_keep_ratio=0.125 → min_keep = ceil(8*0.125) = 1
    cfg = OnlineRFTConfig(n_candidates=8, filter_strategy="correct_only", min_keep_ratio=0.25)
    trainer = OnlineRFTTrainer(cfg)

    # min_keep = ceil(8 * 0.25) = 2; only 1 correct
    candidates = _make_candidates(n=8, n_correct=1)
    kept = trainer.filter_candidates(candidates)

    min_keep = math.ceil(8 * 0.25)
    assert len(kept) >= min_keep, (
        f"Expected at least {min_keep} kept, got {len(kept)}"
    )


# ---------------------------------------------------------------------------
# 4. test_filter_top_k
# ---------------------------------------------------------------------------


def test_filter_top_k():
    """top_k strategy keeps ceil(n_candidates * top_k_ratio) by reward."""
    cfg = OnlineRFTConfig(n_candidates=8, filter_strategy="top_k", top_k_ratio=0.5)
    trainer = OnlineRFTTrainer(cfg)

    candidates = _make_candidates(n=8, n_correct=3)
    kept = trainer.filter_candidates(candidates)

    expected_k = math.ceil(8 * 0.5)
    assert len(kept) == expected_k, (
        f"Expected {expected_k} kept, got {len(kept)}"
    )


# ---------------------------------------------------------------------------
# 5. test_filter_empty_correct
# ---------------------------------------------------------------------------


def test_filter_empty_correct():
    """When all candidates are wrong, min_keep is still enforced."""
    cfg = OnlineRFTConfig(n_candidates=8, filter_strategy="correct_only", min_keep_ratio=0.125)
    trainer = OnlineRFTTrainer(cfg)

    # None correct
    candidates = [_make_sample(is_correct=False, reward=float(i)) for i in range(8)]
    kept = trainer.filter_candidates(candidates)

    min_keep = math.ceil(8 * 0.125)
    assert len(kept) >= min_keep, (
        f"Expected at least {min_keep} when none correct, got {len(kept)}"
    )


# ---------------------------------------------------------------------------
# 6. test_sft_loss_scalar
# ---------------------------------------------------------------------------


def test_sft_loss_scalar():
    """compute_sft_loss returns a scalar (0-dim) tensor."""
    trainer = OnlineRFTTrainer()
    B, T, V = 2, 6, 50
    logits = torch.randn(B, T, V)
    labels = torch.randint(1, V, (B, T))  # no pad

    loss = trainer.compute_sft_loss(logits, labels)
    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 7. test_sft_loss_ignores_pad
# ---------------------------------------------------------------------------


def test_sft_loss_ignores_pad():
    """Pad tokens (label=0) are excluded from SFT loss computation."""
    trainer = OnlineRFTTrainer()
    B, T, V = 1, 8, 32
    torch.manual_seed(42)
    logits = torch.randn(B, T, V)

    # All non-pad labels
    labels_no_pad = torch.randint(1, V, (B, T))
    # Half-pad labels — first 4 positions are 0 (pad)
    labels_with_pad = labels_no_pad.clone()
    labels_with_pad[:, :4] = 0

    loss_no_pad = trainer.compute_sft_loss(logits, labels_no_pad)
    loss_with_pad = trainer.compute_sft_loss(logits, labels_with_pad)

    # Losses should differ because pad positions are excluded
    assert not torch.isclose(loss_no_pad, loss_with_pad, atol=1e-5), (
        "Expected different losses when half the labels are padded"
    )


# ---------------------------------------------------------------------------
# 8. test_sft_loss_gradient
# ---------------------------------------------------------------------------


def test_sft_loss_gradient():
    """Backward pass through compute_sft_loss works without error."""
    trainer = OnlineRFTTrainer()
    B, T, V = 2, 5, 20
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(1, V, (B, T))

    loss = trainer.compute_sft_loss(logits, labels)
    loss.backward()

    assert logits.grad is not None, "Expected gradients on logits"
    assert logits.grad.shape == logits.shape


# ---------------------------------------------------------------------------
# 9. test_kl_penalty_zero_same
# ---------------------------------------------------------------------------


def test_kl_penalty_zero_same():
    """Identical log_probs produce KL penalty ≈ 0."""
    trainer = OnlineRFTTrainer()
    log_probs = torch.randn(2, 8)
    kl = trainer.compute_kl_penalty(log_probs, log_probs)
    assert abs(kl.item()) < 1e-6, f"Expected ~0, got {kl.item()}"


# ---------------------------------------------------------------------------
# 10. test_kl_penalty_positive
# ---------------------------------------------------------------------------


def test_kl_penalty_positive():
    """Different log_probs produce non-zero KL penalty."""
    trainer = OnlineRFTTrainer()
    # policy_lp mean = -0.5, ref_lp mean = -2.0 → KL = mean(policy - ref) = 1.5 > 0
    policy_lp = torch.tensor([[0.0, -1.0]])
    ref_lp = torch.tensor([[-2.0, -2.0]])
    kl = trainer.compute_kl_penalty(policy_lp, ref_lp)
    assert kl.item() > 0.0, f"Expected positive KL, got {kl.item()}"


# ---------------------------------------------------------------------------
# 11. test_total_loss_keys
# ---------------------------------------------------------------------------


def test_total_loss_keys():
    """total_loss returns a dict with 'sft', 'kl', 'total' keys."""
    trainer = OnlineRFTTrainer()
    B, T, V = 1, 4, 16
    logits = torch.randn(B, T, V)
    labels = torch.randint(1, V, (B, T))
    policy_lp = torch.randn(B, T)
    ref_lp = torch.randn(B, T)

    _, metrics = trainer.total_loss(logits, labels, policy_lp, ref_lp)
    assert "sft" in metrics
    assert "kl" in metrics
    assert "total" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


# ---------------------------------------------------------------------------
# 12. test_total_loss_weights
# ---------------------------------------------------------------------------


def test_total_loss_weights():
    """When kl_penalty_weight=0, total loss equals sft_loss."""
    cfg = OnlineRFTConfig(sft_loss_weight=1.0, kl_penalty_weight=0.0)
    trainer = OnlineRFTTrainer(cfg)

    B, T, V = 2, 6, 24
    torch.manual_seed(7)
    logits = torch.randn(B, T, V)
    labels = torch.randint(1, V, (B, T))
    policy_lp = torch.randn(B, T)
    ref_lp = torch.randn(B, T)

    total, metrics = trainer.total_loss(logits, labels, policy_lp, ref_lp)
    assert abs(metrics["total"] - metrics["sft"]) < 1e-5, (
        f"total={metrics['total']}, sft={metrics['sft']}"
    )


# ---------------------------------------------------------------------------
# 13. test_statistics_keys
# ---------------------------------------------------------------------------


def test_statistics_keys():
    """statistics() returns dict with all required keys."""
    trainer = OnlineRFTTrainer()
    candidates = _make_candidates(n=8, n_correct=4)
    kept = trainer.filter_candidates(candidates)
    stats = trainer.statistics(candidates, kept)

    required = {"n_candidates", "n_kept", "keep_rate", "n_correct", "mean_reward"}
    assert required <= set(stats.keys()), (
        f"Missing keys: {required - set(stats.keys())}"
    )


# ---------------------------------------------------------------------------
# 14. test_statistics_keep_rate
# ---------------------------------------------------------------------------


def test_statistics_keep_rate():
    """keep_rate == n_kept / n_candidates."""
    trainer = OnlineRFTTrainer()
    candidates = _make_candidates(n=8, n_correct=4)
    kept = trainer.filter_candidates(candidates)
    stats = trainer.statistics(candidates, kept)

    expected_rate = stats["n_kept"] / stats["n_candidates"]
    assert abs(stats["keep_rate"] - expected_rate) < 1e-9


# ---------------------------------------------------------------------------
# 15. test_statistics_n_correct
# ---------------------------------------------------------------------------


def test_statistics_n_correct():
    """n_correct counts samples with is_correct=True in the candidates list."""
    trainer = OnlineRFTTrainer()
    candidates = _make_candidates(n=8, n_correct=3)
    kept = trainer.filter_candidates(candidates)
    stats = trainer.statistics(candidates, kept)

    assert stats["n_correct"] == 3, (
        f"Expected n_correct=3, got {stats['n_correct']}"
    )
    assert stats["n_candidates"] == 8
