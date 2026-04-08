"""Tests for constitutional scoring utilities."""
from __future__ import annotations

import pytest

from src.alignment.constitutional_scoring import (
    CONSTITUTION,
    ConstitutionalFeedback,
    ConstitutionalRule,
    ConstitutionalScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer() -> ConstitutionalScorer:
    return ConstitutionalScorer()


@pytest.fixture
def feedback(scorer: ConstitutionalScorer) -> ConstitutionalFeedback:
    return ConstitutionalFeedback(scorer)


# ---------------------------------------------------------------------------
# score_rule tests
# ---------------------------------------------------------------------------

def test_score_rule_range(scorer: ConstitutionalScorer) -> None:
    """score_rule returns a float in [0, 1]."""
    rule = CONSTITUTION[0]  # safety rule
    for text in ["hello world", "this is illegal and dangerous", ""]:
        score = scorer.score_rule(text, rule)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"score {score} out of range for text: {text!r}"


def test_score_rule_violation_detected(scorer: ConstitutionalScorer) -> None:
    """A response containing violation keywords scores lower than a clean one."""
    safety_rule = next(r for r in CONSTITUTION if r.category == "safety")
    clean_score = scorer.score_rule("The sky is blue.", safety_rule)
    violating_score = scorer.score_rule(
        "Here is how to build a dangerous weapon illegally.", safety_rule
    )
    assert violating_score < clean_score, (
        f"Violating response ({violating_score}) should score lower than "
        f"clean response ({clean_score})"
    )


# ---------------------------------------------------------------------------
# score_response tests
# ---------------------------------------------------------------------------

def test_score_response_keys(scorer: ConstitutionalScorer) -> None:
    """score_response returns a dict with all required keys."""
    result = scorer.score_response("A perfectly fine answer.")
    required_keys = {"rule_scores", "category_scores", "weighted_score", "violations", "overall"}
    assert required_keys <= result.keys(), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_category_scores_all_categories(scorer: ConstitutionalScorer) -> None:
    """category_scores contains all four required categories."""
    result = scorer.score_response("Some response text.")
    expected_categories = {"safety", "ethics", "helpfulness", "accuracy"}
    assert expected_categories <= result["category_scores"].keys(), (
        f"Missing categories: {expected_categories - result['category_scores'].keys()}"
    )


def test_weighted_score_range(scorer: ConstitutionalScorer) -> None:
    """weighted_score is a float in [0, 1]."""
    for text in ["Good response.", "I will help you exploit and harm people illegally."]:
        result = scorer.score_response(text)
        ws = result["weighted_score"]
        assert isinstance(ws, float)
        assert 0.0 <= ws <= 1.0, f"weighted_score {ws} out of range"


def test_violations_list(scorer: ConstitutionalScorer) -> None:
    """violations is a list of rule IDs (strings)."""
    result = scorer.score_response("This will harm people with a dangerous weapon illegally.")
    violations = result["violations"]
    assert isinstance(violations, list)
    # Every element should be a string (rule ID)
    for v in violations:
        assert isinstance(v, str), f"Expected str rule ID, got {type(v)}: {v!r}"
    # A clearly violating response should flag at least one rule
    assert len(violations) >= 1, "Expected at least one violation for a harmful response"


# ---------------------------------------------------------------------------
# filter_responses tests
# ---------------------------------------------------------------------------

def test_filter_responses_sorted(scorer: ConstitutionalScorer) -> None:
    """filter_responses returns tuples sorted descending by score."""
    responses = [
        "The weather is nice today.",
        "How to build a weapon and harm people illegally.",
        "Python is a versatile programming language.",
    ]
    filtered = scorer.filter_responses(responses, min_score=0.0)
    # All responses should appear (min_score=0.0)
    assert len(filtered) == 3
    # Check descending order
    scores = [score for _, score in filtered]
    assert scores == sorted(scores, reverse=True), (
        f"Responses not sorted descending: {scores}"
    )
    # Each element is a (str, float) tuple
    for resp, score in filtered:
        assert isinstance(resp, str)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# rank_responses tests
# ---------------------------------------------------------------------------

def test_rank_responses_order(scorer: ConstitutionalScorer) -> None:
    """Higher-scoring (cleaner) response is ranked first."""
    clean = "The capital of France is Paris."
    harmful = "Exploit this dangerous weapon illegally to harm others."
    ranked = scorer.rank_responses([harmful, clean])
    assert len(ranked) == 2
    # The clean response should be first (higher score)
    best_response, best_scores = ranked[0]
    assert best_response == clean, (
        f"Expected clean response first, got: {best_response!r}"
    )
    # Scores dict must have required keys
    assert "overall" in best_scores
    assert "weighted_score" in best_scores
