"""Tests for curriculum transition helpers."""

import pytest

from src.training.curriculum_transition import (
    blended_weight,
    cosine_transition,
    linear_transition,
    stage_from_transition,
)


def test_linear_transition_clamps_to_edges():
    assert linear_transition(0, 10, 20) == pytest.approx(0.0)
    assert linear_transition(25, 10, 20) == pytest.approx(1.0)


def test_linear_transition_midpoint():
    assert linear_transition(15, 10, 20) == pytest.approx(0.5)


def test_cosine_transition_midpoint():
    assert cosine_transition(15, 10, 20) == pytest.approx(0.5)


def test_blended_weight_interpolates_between_weights():
    weight = blended_weight(15, 10, 20, 1.0, 3.0)
    assert weight == pytest.approx(2.0)


def test_stage_from_transition_counts_boundaries_crossed():
    assert stage_from_transition(25, [10, 20, 30]) == 2


def test_linear_transition_rejects_bad_range():
    with pytest.raises(ValueError):
        linear_transition(0, 10, 10)


def test_stage_from_transition_rejects_unsorted_boundaries():
    with pytest.raises(ValueError):
        stage_from_transition(5, [10, 5])
