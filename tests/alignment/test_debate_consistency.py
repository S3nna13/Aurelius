"""Tests for debate consistency helpers."""

import pytest

from src.alignment.debate_consistency import (
    argument_overlap,
    debate_run_consistency,
    winner_consistency,
)


def test_winner_consistency_detects_majority():
    assert winner_consistency(["a", "a", "b"]) == pytest.approx(2.0 / 3.0)


def test_winner_consistency_handles_empty_input():
    assert winner_consistency([]) == pytest.approx(0.0)


def test_argument_overlap_one_for_identical_sets():
    assert argument_overlap(["x", "y"], ["x", "y"]) == pytest.approx(1.0)


def test_argument_overlap_zero_for_disjoint_sets():
    assert argument_overlap(["x"], ["y"]) == pytest.approx(0.0)


def test_debate_run_consistency_averages_pairwise_overlap():
    score = debate_run_consistency([["a", "b"], ["a", "b"], ["c"]])
    assert 0.0 < score < 1.0


def test_debate_run_consistency_handles_single_run():
    assert debate_run_consistency([["a"]]) == pytest.approx(1.0)


def test_argument_overlap_handles_empty_sets():
    assert argument_overlap([], []) == pytest.approx(1.0)
