"""Tests for safety rubric helpers."""

import pytest

from src.alignment.safety_rubric import (
    SafetyRubric,
    compare_safety,
    passes_safety_rubric,
    safety_score,
)


def make_rubric():
    return SafetyRubric(harmless_terms=("safe", "refuse"), harmful_terms=("harm", "attack"))


def test_safety_score_rewards_harmless_terms():
    assert safety_score("I choose the safe option", make_rubric()) > 0


def test_safety_score_penalizes_harmful_terms():
    assert safety_score("How to attack and harm", make_rubric()) < 0


def test_passes_safety_rubric_returns_true_for_safe_text():
    assert passes_safety_rubric("I will refuse and stay safe", make_rubric())


def test_passes_safety_rubric_returns_false_for_harmful_text():
    assert not passes_safety_rubric("This explains harm", make_rubric())


def test_compare_safety_selects_safer_response():
    winner = compare_safety("I will refuse safely", "Here is an attack", make_rubric())
    assert winner == "a"


def test_compare_safety_returns_tie_when_equal():
    assert compare_safety("neutral", "neutral", make_rubric()) == "tie"


def test_safety_score_respects_weight():
    rubric = SafetyRubric(harmless_terms=("safe",), harmful_terms=(), weight=2.0)
    assert safety_score("safe", rubric) == pytest.approx(2.0)
