"""Tests for judge diversity metrics."""

import pytest

from src.eval.judge_diversity import lexical_diversity, pairwise_verdict_disagreement, unique_verdict_fraction


def test_unique_verdict_fraction_counts_distinct_labels():
    assert unique_verdict_fraction(["a", "b", "a"]) == pytest.approx(2.0 / 3.0)


def test_unique_verdict_fraction_handles_empty_input():
    assert unique_verdict_fraction([]) == pytest.approx(0.0)


def test_lexical_diversity_positive_for_nonempty_texts():
    score = lexical_diversity(["red blue", "red green"])
    assert 0.0 < score <= 1.0


def test_lexical_diversity_zero_for_empty_texts():
    assert lexical_diversity([]) == pytest.approx(0.0)


def test_pairwise_verdict_disagreement_counts_mismatches():
    score = pairwise_verdict_disagreement(["a", "b"], ["a", "c"])
    assert score == pytest.approx(0.5)


def test_pairwise_verdict_disagreement_rejects_length_mismatch():
    with pytest.raises(ValueError):
        pairwise_verdict_disagreement(["a"], ["a", "b"])


def test_lexical_diversity_handles_repeated_tokens():
    assert lexical_diversity(["x x x"]) == pytest.approx(1.0 / 3.0)
