"""Tests for curriculum sampling utilities."""

import pytest

from src.training.curriculum_sampling import (
    DifficultyBucket,
    allocate_curriculum_budget,
    curriculum_progress,
    curriculum_weights,
    sample_bucket_order,
)


def make_buckets():
    return [
        DifficultyBucket("easy", difficulty=0.1, weight=1.0),
        DifficultyBucket("medium", difficulty=0.5, weight=1.0),
        DifficultyBucket("hard", difficulty=1.0, weight=1.0),
    ]


def test_curriculum_progress_clamps_to_unit_interval():
    assert curriculum_progress(-1, 10) == pytest.approx(0.0)
    assert curriculum_progress(15, 10) == pytest.approx(1.0)


def test_curriculum_weights_shift_toward_harder_buckets():
    early = curriculum_weights(make_buckets(), step=0, total_steps=10)
    late = curriculum_weights(make_buckets(), step=10, total_steps=10)
    assert late["hard"] > early["hard"]


def test_sample_bucket_order_returns_most_likely_first():
    order = sample_bucket_order(make_buckets(), step=10, total_steps=10)
    assert order[0] == "hard"


def test_allocate_curriculum_budget_respects_total():
    allocation = allocate_curriculum_budget(
        make_buckets(), step=5, total_steps=10, total_examples=100
    )
    assert sum(allocation.values()) == 100


def test_curriculum_weights_handles_empty_buckets():
    assert curriculum_weights([], step=0, total_steps=10) == {}


def test_curriculum_progress_rejects_bad_total_steps():
    with pytest.raises(ValueError):
        curriculum_progress(1, 0)


def test_allocate_curriculum_budget_rejects_negative_total():
    with pytest.raises(ValueError):
        allocate_curriculum_budget(make_buckets(), step=1, total_steps=10, total_examples=-1)
