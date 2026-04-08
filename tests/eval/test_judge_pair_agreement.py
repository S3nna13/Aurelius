"""Tests for judge pair agreement helpers."""

import pytest

from src.eval.judge_pair_agreement import majority_pair_label, pair_agreement, pair_label_diversity


def test_pair_agreement_counts_matches():
    assert pair_agreement(["a", "b"], ["a", "c"]) == pytest.approx(0.5)


def test_pair_agreement_handles_empty_input():
    assert pair_agreement([], []) == pytest.approx(0.0)


def test_majority_pair_label_returns_mode():
    assert majority_pair_label(["win", "win", "lose"]) == "win"


def test_pair_label_diversity_fraction_unique():
    assert pair_label_diversity(["a", "b", "a"]) == pytest.approx(2.0 / 3.0)


def test_pair_agreement_rejects_length_mismatch():
    with pytest.raises(ValueError):
        pair_agreement(["a"], ["a", "b"])


def test_majority_pair_label_rejects_empty():
    with pytest.raises(ValueError):
        majority_pair_label([])


def test_pair_label_diversity_handles_empty():
    assert pair_label_diversity([]) == pytest.approx(0.0)

