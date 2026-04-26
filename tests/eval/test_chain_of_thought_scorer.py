"""Tests for chain-of-thought scorer."""

from __future__ import annotations

import pytest

from src.eval.chain_of_thought_scorer import (
    CHAIN_OF_THOUGHT_SCORER_REGISTRY,
    DEFAULT_CHAIN_OF_THOUGHT_SCORER,
    ChainOfThoughtScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def scorer() -> ChainOfThoughtScorer:
    return ChainOfThoughtScorer()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_score_with_matching_answer(scorer):
    reasoning = (
        "Step 1: Identify the problem statement very carefully.\n"
        "Step 2: Apply the formula because it is known to work.\n"
        "Step 3: Compute the final result step by step.\n"
        "Step 4: 42"
    )
    result = scorer.score(reasoning, expected_answer="42")
    assert result["step_count"] == 4
    assert result["logical_flow"] == 1.0
    assert result["conclusion_match"] == 1.0
    # Step 4 is only 1 word so clarity is 0.5; average = (1+1+1+0.5)/4 = 0.875
    assert result["step_clarity"] == pytest.approx(0.875)
    expected_overall = 0.2 * 0.875 + 0.3 * 1.0 + 0.5 * 1.0
    assert result["overall"] == pytest.approx(expected_overall)


def test_score_with_non_matching_answer(scorer):
    reasoning = (
        "Step 1: Identify the problem statement very carefully.\n"
        "Step 2: Apply the formula because it is known to work.\n"
        "Step 3: Compute the final result step by step.\n"
        "Step 4: 99"
    )
    result = scorer.score(reasoning, expected_answer="42")
    assert result["conclusion_match"] == 0.0
    assert result["logical_flow"] == 1.0
    # Step 4 is only 1 word so clarity is 0.5; average = (1+1+1+0.5)/4 = 0.875
    assert result["step_clarity"] == pytest.approx(0.875)
    expected_overall = 0.2 * 0.875 + 0.3 * 1.0 + 0.5 * 0.0
    assert result["overall"] == pytest.approx(expected_overall)


def test_score_without_expected_answer(scorer):
    reasoning = (
        "1. First we analyze the input data carefully.\n"
        "2. Then we apply the rule so we get an intermediate value.\n"
        "3. Finally we output the answer right here."
    )
    result = scorer.score(reasoning)
    assert result["step_count"] == 3
    assert result["step_clarity"] == 1.0
    assert result["logical_flow"] == 1.0
    assert result["conclusion_match"] == 0.0
    expected_overall = 0.4 * 1.0 + 0.6 * 1.0
    assert result["overall"] == pytest.approx(expected_overall)


def test_score_single_short_step_low_clarity(scorer):
    reasoning = "Step 1: short."
    result = scorer.score(reasoning)
    assert result["step_count"] == 1
    assert result["step_clarity"] == 0.5


def test_score_without_logical_keywords(scorer):
    reasoning = (
        "Step 1: Identify the problem statement very carefully.\n"
        "Step 2: Apply the formula to the given values.\n"
        "Step 3: Compute the final result step by step."
    )
    result = scorer.score(reasoning)
    assert result["step_count"] == 3
    assert result["step_clarity"] == 1.0
    assert result["logical_flow"] == 0.7
    expected_overall = 0.4 * 1.0 + 0.6 * 0.7
    assert result["overall"] == pytest.approx(expected_overall)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_empty_reasoning_raises_value_error(scorer):
    with pytest.raises(ValueError, match="reasoning must not be empty"):
        scorer.score("")


def test_oversized_reasoning_raises_value_error(scorer):
    oversized = "a" * 50_001
    with pytest.raises(ValueError, match="exceeds maximum"):
        scorer.score(oversized)


def test_non_string_reasoning_raises_value_error(scorer):
    with pytest.raises(ValueError, match="reasoning must be a string"):
        scorer.score(123)  # type: ignore[arg-type]


def test_non_string_expected_answer_raises_value_error(scorer):
    with pytest.raises(ValueError, match="expected_answer must be a string or None"):
        scorer.score("Step 1: test.", expected_answer=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_score_with_no_numbered_markers(scorer):
    reasoning = "This is a continuous paragraph without any explicit steps at all."
    result = scorer.score(reasoning)
    assert result["step_count"] == 1
    assert result["step_clarity"] == 1.0


def test_score_with_mixed_clarity(scorer):
    reasoning = "Step 1: This step has many words for clarity.\n" "Step 2: Short."
    result = scorer.score(reasoning)
    assert result["step_count"] == 2
    assert result["step_clarity"] == pytest.approx(0.75)


def test_score_case_insensitive_conclusion_match(scorer):
    reasoning = "Step 1: Reasoning.\nStep 2: HELLO"
    result = scorer.score(reasoning, expected_answer="hello")
    assert result["conclusion_match"] == 1.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_contains_default():
    assert "default" in CHAIN_OF_THOUGHT_SCORER_REGISTRY
    assert (
        CHAIN_OF_THOUGHT_SCORER_REGISTRY["default"] is DEFAULT_CHAIN_OF_THOUGHT_SCORER
    )
