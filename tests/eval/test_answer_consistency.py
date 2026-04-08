"""Tests for answer consistency metrics."""

import pytest

from src.eval.answer_consistency import exact_match_rate, lexical_overlap, mean_pairwise_consistency, normalize_answer


def test_normalize_answer_tokenizes_lowercase():
    assert normalize_answer("Hello, WORLD!") == ("hello", "world")


def test_exact_match_rate_detects_majority_answer():
    rate = exact_match_rate(["Yes", "yes", "No"])
    assert rate == pytest.approx(2.0 / 3.0)


def test_lexical_overlap_is_one_for_identical_answers():
    assert lexical_overlap("the cat", "the cat") == pytest.approx(1.0)


def test_lexical_overlap_is_zero_for_disjoint_answers():
    assert lexical_overlap("cat", "dog") == pytest.approx(0.0)


def test_mean_pairwise_consistency_averages_over_pairs():
    score = mean_pairwise_consistency(["the cat", "the cat", "a dog"])
    assert 0.0 < score < 1.0


def test_exact_match_rate_handles_empty_input():
    assert exact_match_rate([]) == pytest.approx(0.0)


def test_mean_pairwise_consistency_handles_single_answer():
    assert mean_pairwise_consistency(["only one"]) == pytest.approx(1.0)

