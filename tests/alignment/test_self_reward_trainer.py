"""Tests for src/alignment/self_reward_trainer.py — Self-Reward Trainer.

Covers:
    1.  test_config_defaults
    2.  test_create_pairs_sufficient_gap
    3.  test_create_pairs_insufficient_gap
    4.  test_create_pairs_ordering
    5.  test_create_pairs_multiple
    6.  test_create_pairs_all_same
    7.  test_dpo_loss_scalar
    8.  test_dpo_loss_positive
    9.  test_dpo_loss_correct_preferred
    10. test_compute_loss_keys
    11. test_compute_loss_no_pairs
    12. test_n_pairs_counted
    13. test_score_statistics_keys
    14. test_gradient_flows
    Integration: test_integration_forward_backward
"""

from __future__ import annotations

import torch
import pytest

from src.alignment.self_reward_trainer import (
    SelfRewardConfig,
    ScoredCandidate,
    SelfRewardBatch,
    SelfRewardTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    score: float,
    T: int = 8,
    seed: int | None = None,
    requires_grad: bool = False,
) -> ScoredCandidate:
    """Create a ScoredCandidate with random log-probs."""
    if seed is not None:
        torch.manual_seed(seed)
    log_probs = -torch.rand(T) * 3.0 - 0.1  # in (-3.1, -0.1)
    if requires_grad:
        log_probs = log_probs.requires_grad_(True)
    mask = torch.ones(T)
    token_ids = list(range(T))
    return ScoredCandidate(token_ids=token_ids, log_probs=log_probs, mask=mask, score=score)


def _make_ref_lps(candidates: list[ScoredCandidate], seed: int = 0) -> list[torch.Tensor]:
    """Create detached reference log-probs matching each candidate's length."""
    torch.manual_seed(seed)
    refs = []
    for c in candidates:
        T = c.log_probs.shape[0]
        refs.append(-torch.rand(T) * 3.0 - 0.1)
    return refs


def _make_batch(
    scores_per_prompt: list[list[float]],
    T: int = 8,
    requires_grad: bool = False,
    seed: int = 42,
) -> SelfRewardBatch:
    """Build a SelfRewardBatch from score lists per prompt."""
    all_candidates: list[list[ScoredCandidate]] = []
    all_refs: list[torch.Tensor] = []
    for p_idx, scores in enumerate(scores_per_prompt):
        cands = [
            _make_candidate(s, T=T, seed=seed + p_idx * 10 + c_idx,
                            requires_grad=requires_grad)
            for c_idx, s in enumerate(scores)
        ]
        all_candidates.append(cands)
        flat = [c for c in cands]
        all_refs.extend(_make_ref_lps(flat, seed=seed + p_idx))
    return SelfRewardBatch(candidates=all_candidates, ref_log_probs=all_refs)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """SelfRewardConfig defaults match the paper's recommended values."""
    cfg = SelfRewardConfig()
    assert cfg.n_candidates == 4
    assert cfg.min_score_gap == 2.0
    assert cfg.max_score == 5.0
    assert cfg.beta == 0.1
    assert cfg.eps == 1e-8


# ---------------------------------------------------------------------------
# 2. test_create_pairs_sufficient_gap
# ---------------------------------------------------------------------------


def test_create_pairs_sufficient_gap():
    """A gap exactly equal to min_score_gap forms one valid pair."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=2.0))
    cands = [_make_candidate(4.0, seed=0), _make_candidate(2.0, seed=1)]
    pairs = trainer.create_preference_pairs(cands)
    assert len(pairs) == 1
    chosen, rejected = pairs[0]
    assert chosen.score == 4.0
    assert rejected.score == 2.0


# ---------------------------------------------------------------------------
# 3. test_create_pairs_insufficient_gap
# ---------------------------------------------------------------------------


def test_create_pairs_insufficient_gap():
    """A gap strictly below min_score_gap yields no pairs."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=2.0))
    cands = [_make_candidate(3.0, seed=0), _make_candidate(1.5, seed=1)]
    # gap == 1.5 < 2.0
    pairs = trainer.create_preference_pairs(cands)
    assert pairs == []


# ---------------------------------------------------------------------------
# 4. test_create_pairs_ordering
# ---------------------------------------------------------------------------


def test_create_pairs_ordering():
    """chosen always has a strictly higher score than rejected in every pair."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=1.5))
    cands = [
        _make_candidate(5.0, seed=0),
        _make_candidate(3.0, seed=1),
        _make_candidate(0.5, seed=2),
    ]
    pairs = trainer.create_preference_pairs(cands)
    assert len(pairs) > 0
    for chosen, rejected in pairs:
        assert chosen.score > rejected.score, (
            f"Pair ordering violated: chosen={chosen.score}, rejected={rejected.score}"
        )


# ---------------------------------------------------------------------------
# 5. test_create_pairs_multiple
# ---------------------------------------------------------------------------


def test_create_pairs_multiple():
    """Four candidates with varied scores produce multiple valid pairs."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=2.0))
    cands = [
        _make_candidate(5.0, seed=0),
        _make_candidate(4.0, seed=1),
        _make_candidate(2.0, seed=2),
        _make_candidate(0.0, seed=3),
    ]
    pairs = trainer.create_preference_pairs(cands)
    # (5,2),(5,0),(4,0),(4,2) are valid — at least 3 should exist
    assert len(pairs) >= 3, f"Expected >=3 pairs, got {len(pairs)}"
    # All chosen scores must be higher than their rejected counterpart
    for c, r in pairs:
        assert c.score - r.score >= 2.0


# ---------------------------------------------------------------------------
# 6. test_create_pairs_all_same
# ---------------------------------------------------------------------------


def test_create_pairs_all_same():
    """All candidates with identical scores yield no preference pairs."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=2.0))
    cands = [_make_candidate(3.0, seed=i) for i in range(4)]
    pairs = trainer.create_preference_pairs(cands)
    assert pairs == [], f"Expected 0 pairs for equal scores, got {len(pairs)}"


# ---------------------------------------------------------------------------
# 7. test_dpo_loss_scalar
# ---------------------------------------------------------------------------


def test_dpo_loss_scalar():
    """dpo_loss must return a 0-dimensional (scalar) tensor."""
    trainer = SelfRewardTrainer()
    torch.manual_seed(0)
    B = 4
    c_lp = -torch.rand(B) - 0.1
    c_ref = -torch.rand(B) - 0.1
    r_lp = -torch.rand(B) - 0.1
    r_ref = -torch.rand(B) - 0.1
    loss = trainer.dpo_loss(c_lp, c_ref, r_lp, r_ref)
    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 8. test_dpo_loss_positive
# ---------------------------------------------------------------------------


def test_dpo_loss_positive():
    """dpo_loss is > 0 for random, arbitrary inputs."""
    trainer = SelfRewardTrainer()
    torch.manual_seed(1)
    B = 6
    c_lp = -torch.rand(B) - 0.1
    c_ref = -torch.rand(B) - 0.1
    r_lp = -torch.rand(B) - 0.1
    r_ref = -torch.rand(B) - 0.1
    loss = trainer.dpo_loss(c_lp, c_ref, r_lp, r_ref)
    assert loss.item() > 0.0, f"Expected dpo_loss > 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 9. test_dpo_loss_correct_preferred
# ---------------------------------------------------------------------------


def test_dpo_loss_correct_preferred():
    """When chosen log-ratios >> rejected log-ratios, loss is near zero."""
    trainer = SelfRewardTrainer(SelfRewardConfig(beta=0.1))
    B = 4
    # Chosen: policy >> reference (large positive log-ratio)
    c_lp = torch.full((B,), -0.05)
    c_ref = torch.full((B,), -5.0)
    # Rejected: policy << reference (large negative log-ratio)
    r_lp = torch.full((B,), -5.0)
    r_ref = torch.full((B,), -0.05)

    loss_good = trainer.dpo_loss(c_lp, c_ref, r_lp, r_ref).item()

    # Reverse: rejected preferred
    loss_bad = trainer.dpo_loss(r_lp, r_ref, c_lp, c_ref).item()

    assert loss_good < loss_bad, (
        f"Expected smaller loss when chosen dominates: {loss_good:.4f} vs {loss_bad:.4f}"
    )


# ---------------------------------------------------------------------------
# 10. test_compute_loss_keys
# ---------------------------------------------------------------------------


def test_compute_loss_keys():
    """compute_loss must return a dict with exactly the required keys."""
    trainer = SelfRewardTrainer()
    batch = _make_batch([[5.0, 2.0, 0.0, 4.0], [3.0, 1.0, 5.0, 0.0]])
    out = trainer.compute_loss(batch)
    required = {"loss", "n_pairs", "mean_score_gap", "reward_accuracy"}
    assert required <= set(out.keys()), (
        f"Missing keys: {required - set(out.keys())}"
    )


# ---------------------------------------------------------------------------
# 11. test_compute_loss_no_pairs
# ---------------------------------------------------------------------------


def test_compute_loss_no_pairs():
    """When no valid pairs exist, compute_loss returns loss == 0."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=2.0))
    # All scores within gap < 2.0 of each other
    batch = _make_batch([[3.0, 3.5, 2.8, 3.2]])
    out = trainer.compute_loss(batch)
    assert out["n_pairs"].item() == 0.0
    assert out["loss"].item() == 0.0


# ---------------------------------------------------------------------------
# 12. test_n_pairs_counted
# ---------------------------------------------------------------------------


def test_n_pairs_counted():
    """n_pairs in output matches the actual number of valid preference pairs."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=2.0))
    scores_prompt0 = [5.0, 4.0, 2.0, 0.0]  # known valid pairs
    scores_prompt1 = [3.0, 3.0, 3.0, 3.0]  # no pairs

    batch = _make_batch([scores_prompt0, scores_prompt1])

    # Count manually
    cands0 = [_make_candidate(s, seed=i) for i, s in enumerate(scores_prompt0)]
    expected_pairs = len(trainer.create_preference_pairs(cands0))

    out = trainer.compute_loss(batch)
    assert out["n_pairs"].item() == float(expected_pairs), (
        f"Expected {expected_pairs} pairs, got {out['n_pairs'].item()}"
    )


# ---------------------------------------------------------------------------
# 13. test_score_statistics_keys
# ---------------------------------------------------------------------------


def test_score_statistics_keys():
    """score_statistics must return a dict with all required keys."""
    trainer = SelfRewardTrainer()
    candidates_per_prompt = [
        [_make_candidate(s, seed=i + j * 5) for j, s in enumerate([5.0, 2.0, 4.0, 0.0])]
        for i in range(3)
    ]
    stats = trainer.score_statistics(candidates_per_prompt)
    required = {
        "mean_score",
        "std_score",
        "max_score_observed",
        "min_score_observed",
        "pairs_created",
    }
    assert required <= set(stats.keys()), (
        f"Missing keys: {required - set(stats.keys())}"
    )
    assert 0.0 <= stats["min_score_observed"] <= stats["max_score_observed"]
    assert stats["std_score"] >= 0.0
    assert stats["pairs_created"] >= 0.0


# ---------------------------------------------------------------------------
# 14. test_gradient_flows
# ---------------------------------------------------------------------------


def test_gradient_flows():
    """backward() on compute_loss produces finite gradients on policy log-probs."""
    trainer = SelfRewardTrainer(SelfRewardConfig(min_score_gap=2.0))
    batch = _make_batch([[5.0, 0.0, 3.0, 1.0]], requires_grad=True)

    out = trainer.compute_loss(batch)
    assert out["n_pairs"].item() > 0, "No pairs — gradient test needs pairs"
    out["loss"].backward()

    for cands in batch.candidates:
        for c in cands:
            if c.log_probs.requires_grad:
                assert c.log_probs.grad is not None, (
                    f"No gradient on candidate with score={c.score}"
                )
                assert torch.isfinite(c.log_probs.grad).all(), (
                    f"Non-finite gradient for score={c.score}"
                )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_integration_forward_backward():
    """Integration: B=2 prompts, M=4 candidates each, varied scores — full
    forward + backward pass through compute_loss.
    """
    torch.manual_seed(99)

    config = SelfRewardConfig(n_candidates=4, min_score_gap=2.0, beta=0.1)
    trainer = SelfRewardTrainer(config)

    # Build two prompts with deliberate score distributions
    scores_p0 = [5.0, 4.0, 1.5, 0.0]   # several valid pairs
    scores_p1 = [4.5, 2.0, 3.5, 0.5]   # several valid pairs

    batch = _make_batch([scores_p0, scores_p1], requires_grad=True, seed=77)

    # Forward
    out = trainer.compute_loss(batch)

    # Keys present
    for key in ("loss", "n_pairs", "mean_score_gap", "reward_accuracy"):
        assert key in out, f"Missing key: {key}"

    # Loss is finite
    assert torch.isfinite(out["loss"]), f"Non-finite loss: {out['loss']}"

    # Some valid pairs must have been found
    assert out["n_pairs"].item() > 0, "Expected at least one valid preference pair"

    # mean_score_gap >= min_score_gap
    assert out["mean_score_gap"].item() >= config.min_score_gap - 1e-6, (
        f"mean_score_gap={out['mean_score_gap'].item()} < min_score_gap={config.min_score_gap}"
    )

    # reward_accuracy in [0, 1]
    acc = out["reward_accuracy"].item()
    assert 0.0 <= acc <= 1.0, f"reward_accuracy={acc} out of [0,1]"

    # Backward
    out["loss"].backward()

    # Gradients flow to at least some candidates
    grads_found = 0
    for cands in batch.candidates:
        for c in cands:
            if c.log_probs.requires_grad and c.log_probs.grad is not None:
                assert torch.isfinite(c.log_probs.grad).all(), (
                    f"Non-finite gradient for candidate with score={c.score}"
                )
                grads_found += 1

    assert grads_found > 0, "No gradients flowed to any candidate log_probs"

    # Score statistics
    stats = trainer.score_statistics(batch.candidates)
    assert stats["max_score_observed"] == 5.0
    assert stats["min_score_observed"] == 0.0
    assert stats["pairs_created"] > 0
    assert stats["mean_score"] > 0.0
