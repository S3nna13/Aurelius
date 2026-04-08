"""Tests for constitutional scoring utilities."""

import pytest

from src.eval.constitutional_scoring import (
    ConstitutionalPrinciple,
    constitutional_score,
    passes_constitution,
    principle_breakdown,
    principle_score,
)


def make_principles() -> list[ConstitutionalPrinciple]:
    return [
        ConstitutionalPrinciple("helpfulness", 1.0, ("help", "support"), ("refuse",)),
        ConstitutionalPrinciple("safety", 2.0, ("safe",), ("harm", "unsafe")),
    ]


def test_principle_score_rewards_preferred_terms():
    principle = ConstitutionalPrinciple("help", 1.0, ("help",), ())
    assert principle_score("I can help with that", principle) > 0


def test_principle_score_penalizes_disallowed_terms():
    principle = ConstitutionalPrinciple("safe", 1.0, (), ("harm",))
    assert principle_score("This could cause harm", principle) < 0


def test_constitutional_score_aggregates_principles():
    score = constitutional_score("I can help safely", make_principles())
    assert score > 0


def test_principle_breakdown_returns_named_scores():
    breakdown = principle_breakdown("I can help safely", make_principles())
    assert set(breakdown) == {"helpfulness", "safety"}


def test_passes_constitution_true_for_positive_text():
    assert passes_constitution("I can help safely", make_principles())


def test_passes_constitution_false_for_negative_text():
    assert not passes_constitution("I refuse and may cause harm", make_principles(), threshold=0.0)


def test_constitutional_score_zero_for_empty_principles():
    assert constitutional_score("anything", []) == pytest.approx(0.0)
