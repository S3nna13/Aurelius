"""Tests for response faithfulness helpers."""

import pytest

from src.eval.response_faithfulness import contradiction_rate, faithfulness_score, support_coverage


def test_support_coverage_one_for_supported_response():
    assert support_coverage("the cat sat", "the cat sat on mat") == pytest.approx(1.0)


def test_support_coverage_partial_for_partial_overlap():
    score = support_coverage("the cat jumped", "the cat sat")
    assert 0.0 < score < 1.0


def test_contradiction_rate_zero_when_negation_matches():
    assert contradiction_rate("do not go", "we should not go") == pytest.approx(0.0)


def test_contradiction_rate_positive_for_negation_mismatch():
    assert contradiction_rate("go now", "do not go") > 0.0


def test_faithfulness_score_combines_signals():
    score = faithfulness_score("the cat sat", "the cat sat")
    assert score == pytest.approx(1.0)


def test_faithfulness_score_clamps_at_zero():
    score = faithfulness_score("go now", "do not go")
    assert score >= 0.0


def test_support_coverage_handles_empty_response():
    assert support_coverage("", "anything") == pytest.approx(1.0)
