"""Tests for src/training/curriculum_learning.py."""

from __future__ import annotations

import math
import pytest
import torch

from src.training.curriculum_learning import (
    CurriculumConfig,
    DifficultyScore,
    linear_curriculum_weight,
    exponential_curriculum_weight,
    step_curriculum_weight,
    DifficultyRanker,
    CurriculumSampler,
    compute_sample_difficulty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(n: int = 10) -> list[DifficultyScore]:
    """Create n DifficultyScore objects with scores 0.0, 1.0, ..., n-1."""
    return [DifficultyScore(sample_id=i, score=float(i)) for i in range(n)]


def _make_ranker(n: int = 10) -> DifficultyRanker:
    return DifficultyRanker(_make_scores(n))


# ---------------------------------------------------------------------------
# CurriculumConfig defaults
# ---------------------------------------------------------------------------

def test_curriculum_config_defaults():
    cfg = CurriculumConfig()
    assert cfg.strategy == "linear"
    assert cfg.n_stages == 5
    assert cfg.warmup_steps == 100
    assert cfg.total_steps == 1000
    assert cfg.difficulty_metric == "loss"


def test_curriculum_config_custom():
    cfg = CurriculumConfig(strategy="step", n_stages=3, total_steps=500, difficulty_metric="perplexity")
    assert cfg.strategy == "step"
    assert cfg.n_stages == 3
    assert cfg.total_steps == 500
    assert cfg.difficulty_metric == "perplexity"


def test_difficulty_score_defaults():
    ds = DifficultyScore(sample_id=7, score=0.42)
    assert ds.sample_id == 7
    assert ds.score == pytest.approx(0.42)
    assert ds.metadata == {}


# ---------------------------------------------------------------------------
# linear_curriculum_weight
# ---------------------------------------------------------------------------

def test_linear_weight_at_zero():
    assert linear_curriculum_weight(0, 100) == pytest.approx(0.0)


def test_linear_weight_at_total():
    assert linear_curriculum_weight(100, 100) == pytest.approx(1.0)


def test_linear_weight_midpoint():
    assert linear_curriculum_weight(50, 100) == pytest.approx(0.5)


def test_linear_weight_clamped_above():
    assert linear_curriculum_weight(200, 100) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# exponential_curriculum_weight
# ---------------------------------------------------------------------------

def test_exponential_weight_at_zero():
    assert exponential_curriculum_weight(0, 100) == pytest.approx(0.0, abs=1e-9)


def test_exponential_weight_at_total():
    assert exponential_curriculum_weight(100, 100) == pytest.approx(1.0, abs=1e-9)


def test_exponential_weight_monotone():
    vals = [exponential_curriculum_weight(s, 100) for s in range(0, 110, 10)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a


def test_exponential_weight_clamped():
    val = exponential_curriculum_weight(200, 100)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# step_curriculum_weight
# ---------------------------------------------------------------------------

def test_step_weight_staircase():
    """Values should be staircase-shaped (non-decreasing with discrete jumps)."""
    n_stages = 5
    total = 100
    prev = -1.0
    for step in range(0, total + 1, 10):
        val = step_curriculum_weight(step, n_stages, total)
        assert val >= prev - 1e-9
        prev = val


def test_step_weight_at_zero():
    assert step_curriculum_weight(0, 5, 100) == pytest.approx(0.0)


def test_step_weight_discrete_values():
    """Only n_stages distinct values (0/n, 1/n, ..., (n-1)/n)."""
    n_stages = 4
    total = 100
    seen = set()
    for step in range(0, total + 1):
        seen.add(step_curriculum_weight(step, n_stages, total))
    assert len(seen) == n_stages  # 0/4, 1/4, 2/4, 3/4


# ---------------------------------------------------------------------------
# DifficultyRanker.rank
# ---------------------------------------------------------------------------

def test_rank_ascending():
    ranker = _make_ranker(10)
    ranked = ranker.rank(ascending=True)
    scores = [ds.score for ds in ranked]
    assert scores == sorted(scores)


def test_rank_descending():
    ranker = _make_ranker(10)
    ranked = ranker.rank(ascending=False)
    scores = [ds.score for ds in ranked]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# DifficultyRanker.get_percentile
# ---------------------------------------------------------------------------

def test_get_percentile_at_0():
    ranker = _make_ranker(10)
    assert ranker.get_percentile(0) == pytest.approx(min(float(i) for i in range(10)))


def test_get_percentile_at_100():
    ranker = _make_ranker(10)
    assert ranker.get_percentile(100) == pytest.approx(max(float(i) for i in range(10)))


def test_get_percentile_at_50():
    ranker = _make_ranker(11)  # 0..10, median is 5
    p50 = ranker.get_percentile(50)
    assert 4.0 <= p50 <= 6.0


# ---------------------------------------------------------------------------
# DifficultyRanker.get_easy_samples / get_hard_samples
# ---------------------------------------------------------------------------

def test_get_easy_samples_fraction():
    ranker = _make_ranker(20)
    easy = ranker.get_easy_samples(0.25)
    # ceil(20 * 0.25) = 5
    assert len(easy) == 5
    # All should be among the lowest scores
    easy_scores = [ds.score for ds in easy]
    assert max(easy_scores) < 10.0  # scores 0-4 are the easiest of 20


def test_get_hard_samples_fraction():
    ranker = _make_ranker(20)
    hard = ranker.get_hard_samples(0.25)
    assert len(hard) == 5
    hard_scores = [ds.score for ds in hard]
    assert min(hard_scores) >= 10.0  # scores 15-19 are hardest


def test_get_easy_returns_lowest():
    ranker = _make_ranker(10)
    easy = ranker.get_easy_samples(0.3)  # ceil(10*0.3)=3
    easy_scores = sorted(ds.score for ds in easy)
    assert easy_scores == [0.0, 1.0, 2.0]


def test_get_hard_returns_highest():
    ranker = _make_ranker(10)
    hard = ranker.get_hard_samples(0.3)  # ceil(10*0.3)=3
    hard_scores = sorted((ds.score for ds in hard), reverse=True)
    assert hard_scores == [9.0, 8.0, 7.0]


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------

def _make_sampler(strategy: str = "linear", n: int = 10) -> CurriculumSampler:
    cfg = CurriculumConfig(strategy=strategy, total_steps=100, n_stages=5)
    ranker = _make_ranker(n)
    return CurriculumSampler(ranker, cfg)


def test_curriculum_fraction_grows_with_step():
    sampler = _make_sampler("linear")
    fracs = [sampler.get_curriculum_fraction(s) for s in [0, 25, 50, 75, 100]]
    for a, b in zip(fracs, fracs[1:]):
        assert b >= a - 1e-9


def test_sample_indices_length():
    sampler = _make_sampler()
    indices = sampler.sample_indices(step=50, n_samples=7)
    assert len(indices) == 7


def test_sample_indices_are_valid_ids():
    sampler = _make_sampler(n=10)
    valid_ids = set(range(10))
    indices = sampler.sample_indices(step=80, n_samples=15)
    for idx in indices:
        assert idx in valid_ids


def test_sample_indices_early_step_easier():
    """At step 0 (easy only) vs step 100 (full dataset), earlier steps should yield lower ids."""
    sampler = _make_sampler(n=20)
    early = sampler.sample_indices(step=1, n_samples=5)
    late = sampler.sample_indices(step=100, n_samples=5)
    # At least the max id seen early should be <= max id seen late (since hard samples added later)
    assert max(early) <= max(late)


def test_update_scores_changes_ranking():
    sampler = _make_sampler(n=10)
    # Flip: make sample_id=0 the hardest by giving it a huge score
    updated = [DifficultyScore(sample_id=0, score=999.0)]
    sampler.update_scores(updated)
    ranked = sampler._ranker.rank(ascending=True)
    # sample_id=0 should now be last
    assert ranked[-1].sample_id == 0


def test_update_scores_new_sample():
    """Adding a brand-new sample_id should appear in the ranker."""
    sampler = _make_sampler(n=5)
    sampler.update_scores([DifficultyScore(sample_id=99, score=0.1)])
    all_ids = {ds.sample_id for ds in sampler._ranker._scores}
    assert 99 in all_ids


# ---------------------------------------------------------------------------
# compute_sample_difficulty
# ---------------------------------------------------------------------------

def test_compute_sample_difficulty_length():
    losses = torch.tensor([0.1, 0.5, 1.2, 0.3])
    result = compute_sample_difficulty(losses)
    assert len(result) == 4


def test_compute_sample_difficulty_type():
    losses = torch.tensor([0.1, 0.5, 1.2])
    result = compute_sample_difficulty(losses)
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_compute_sample_difficulty_values():
    losses = torch.tensor([0.1, 0.5, 1.2])
    result = compute_sample_difficulty(losses)
    assert result[0] == pytest.approx(0.1, abs=1e-5)
    assert result[1] == pytest.approx(0.5, abs=1e-5)
    assert result[2] == pytest.approx(1.2, abs=1e-5)


def test_compute_sample_difficulty_no_grad():
    """Should work with gradient-tracked tensors (detaches internally)."""
    losses = torch.tensor([0.2, 0.8], requires_grad=True)
    result = compute_sample_difficulty(losses)
    assert len(result) == 2
    assert isinstance(result[0], float)
