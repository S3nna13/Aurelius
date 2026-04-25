"""Tests for TruthfulQA scorer."""

from __future__ import annotations

import pytest

from src.eval.truthfulqa_scorer import (
    TRUTHFULQA_REGISTRY,
    TruthfulQAQuestion,
    TruthfulQAResult,
    TruthfulQAScorer,
)


@pytest.fixture
def scorer() -> TruthfulQAScorer:
    return TruthfulQAScorer()


@pytest.fixture
def question() -> TruthfulQAQuestion:
    return TruthfulQAQuestion(
        question_id="q1",
        question="Is the sky blue?",
        correct_answers=["Yes, the sky appears blue", "The sky is blue"],
        incorrect_answers=["No, the sky is green", "The sky is purple"],
        category="science",
    )


def test_exact_match_correct(scorer, question):
    result = scorer.score_answer(question, "The sky is blue")
    assert result.is_truthful is True
    assert result.match_type == "exact"
    assert result.question_id == "q1"


def test_exact_match_case_insensitive(scorer, question):
    result = scorer.score_answer(question, "THE SKY IS BLUE")
    assert result.is_truthful is True
    assert result.match_type == "exact"


def test_exact_match_incorrect(scorer, question):
    result = scorer.score_answer(question, "No, the sky is green")
    assert result.is_truthful is False
    assert result.match_type == "exact"


def test_substring_match_correct(scorer, question):
    result = scorer.score_answer(question, "Yes, the sky appears blue on a clear day")
    assert result.is_truthful is True
    assert result.match_type == "substring"


def test_substring_match_incorrect(scorer, question):
    result = scorer.score_answer(question, "Actually the sky is purple at times")
    assert result.is_truthful is False
    assert result.match_type == "substring"


def test_no_match(scorer, question):
    result = scorer.score_answer(question, "I have no idea whatsoever")
    assert result.is_truthful is False
    assert result.match_type == "none"


def test_result_is_frozen(scorer, question):
    result = scorer.score_answer(question, "The sky is blue")
    with pytest.raises((AttributeError, TypeError)):
        result.is_truthful = False


def test_score_batch_accuracy(scorer):
    questions = [
        TruthfulQAQuestion(
            question_id=f"q{i}",
            question="Q?",
            correct_answers=["correct"],
            incorrect_answers=["wrong"],
            category="test",
        )
        for i in range(4)
    ]
    answers = ["correct", "correct", "wrong", "no match"]
    result = scorer.score_batch(questions, answers)
    assert result["n_total"] == 4
    assert result["n_truthful"] == 2
    assert abs(result["accuracy"] - 0.5) < 1e-9


def test_score_batch_by_category(scorer):
    questions = [
        TruthfulQAQuestion("q0", "Q?", ["yes"], ["no"], category="cat_a"),
        TruthfulQAQuestion("q1", "Q?", ["yes"], ["no"], category="cat_a"),
        TruthfulQAQuestion("q2", "Q?", ["yes"], ["no"], category="cat_b"),
    ]
    answers = ["yes", "no", "yes"]
    result = scorer.score_batch(questions, answers)
    by_cat = result["by_category"]
    assert abs(by_cat["cat_a"] - 0.5) < 1e-9
    assert abs(by_cat["cat_b"] - 1.0) < 1e-9


def test_load_sample_questions_returns_five(scorer):
    samples = scorer.load_sample_questions()
    assert len(samples) == 5


def test_load_sample_questions_are_valid(scorer):
    for q in scorer.load_sample_questions():
        assert isinstance(q, TruthfulQAQuestion)
        assert len(q.correct_answers) >= 1
        assert len(q.incorrect_answers) >= 1


def test_registry_key(scorer):
    assert "default" in TRUTHFULQA_REGISTRY
    assert TRUTHFULQA_REGISTRY["default"] is TruthfulQAScorer
