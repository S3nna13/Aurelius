"""Tests for local LLM-as-judge utilities."""

import pytest

from src.eval.llm_as_judge import (
    JudgeRubric,
    average_rubric,
    judge_agreement,
    pairwise_winner,
    parse_judge_output,
)


def test_parse_judge_output_extracts_scores():
    rubric = parse_judge_output("coherence: 8\nhelpfulness: 7\nsafety: 9")
    assert rubric.coherence == pytest.approx(8.0)
    assert rubric.helpfulness == pytest.approx(7.0)
    assert rubric.safety == pytest.approx(9.0)


def test_parse_judge_output_rejects_missing_fields():
    with pytest.raises(ValueError):
        parse_judge_output("coherence: 8\nhelpfulness: 7")


def test_pairwise_winner_prefers_higher_overall_score():
    left = JudgeRubric(9, 8, 9)
    right = JudgeRubric(7, 7, 8)
    assert pairwise_winner(left, right) == "left"


def test_pairwise_winner_returns_tie_when_equal():
    left = JudgeRubric(8, 8, 8)
    right = JudgeRubric(8, 8, 8)
    assert pairwise_winner(left, right) == "tie"


def test_judge_agreement_measures_majority_fraction():
    agreement = judge_agreement(["left", "left", "right", "left"])
    assert agreement == pytest.approx(0.75)


def test_average_rubric_means_all_dimensions():
    rubric = average_rubric([JudgeRubric(9, 6, 9), JudgeRubric(3, 6, 3)])
    assert rubric.coherence == pytest.approx(6.0)
    assert rubric.helpfulness == pytest.approx(6.0)
    assert rubric.safety == pytest.approx(6.0)


def test_average_rubric_rejects_empty_input():
    with pytest.raises(ValueError):
        average_rubric([])
