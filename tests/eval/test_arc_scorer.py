"""Tests for ARC scorer."""

from __future__ import annotations

import pytest

from src.eval.arc_scorer import (
    ARC_REGISTRY,
    ARCQuestion,
    ARCScorer,
)


@pytest.fixture
def scorer() -> ARCScorer:
    return ARCScorer()


@pytest.fixture
def question() -> ARCQuestion:
    return ARCQuestion(
        question_id="arc_test_0",
        question="What color is the sky?",
        choices={"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
        correct_key="B",
        difficulty="Easy",
    )


def test_parse_answer_parenthesis_format(scorer, question):
    assert scorer.parse_answer("(B)", question.choices) == "B"


def test_parse_answer_letter_paren_format(scorer, question):
    assert scorer.parse_answer("B)", question.choices) == "B"


def test_parse_answer_standalone_letter(scorer, question):
    assert scorer.parse_answer("B", question.choices) == "B"


def test_parse_answer_lowercase(scorer, question):
    assert scorer.parse_answer("(b)", question.choices) == "B"


def test_parse_answer_text_match(scorer, question):
    assert scorer.parse_answer("The answer is Blue", question.choices) == "B"


def test_parse_answer_unresolvable(scorer, question):
    assert scorer.parse_answer("I don't know", question.choices) == "?"


def test_score_answer_correct(scorer, question):
    result = scorer.score_answer(question, "(B)")
    assert result.is_correct is True
    assert result.predicted_key == "B"
    assert result.question_id == "arc_test_0"


def test_score_answer_wrong(scorer, question):
    result = scorer.score_answer(question, "(A)")
    assert result.is_correct is False
    assert result.predicted_key == "A"


def test_result_is_frozen(scorer, question):
    result = scorer.score_answer(question, "(B)")
    with pytest.raises((AttributeError, TypeError)):
        result.is_correct = False


def test_score_batch_easy_challenge_split(scorer):
    questions = [
        ARCQuestion("e0", "Q?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A", "Easy"),
        ARCQuestion("e1", "Q?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A", "Easy"),
        ARCQuestion("c0", "Q?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "B", "Challenge"),
        ARCQuestion("c1", "Q?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "B", "Challenge"),
    ]
    answers = ["(A)", "(B)", "(B)", "(A)"]
    result = scorer.score_batch(questions, answers)
    assert result["n_total"] == 4
    assert result["n_correct"] == 2
    assert abs(result["accuracy"] - 0.5) < 1e-9
    assert abs(result["easy_accuracy"] - 0.5) < 1e-9
    assert abs(result["challenge_accuracy"] - 0.5) < 1e-9


def test_score_batch_all_easy(scorer):
    questions = [
        ARCQuestion("e0", "Q?", {"A": "correct answer", "B": "b", "C": "c", "D": "d"}, "A", "Easy"),
    ]
    answers = ["(A)"]
    result = scorer.score_batch(questions, answers)
    assert abs(result["easy_accuracy"] - 1.0) < 1e-9
    assert abs(result["challenge_accuracy"] - 0.0) < 1e-9


def test_load_sample_questions_returns_five(scorer):
    samples = scorer.load_sample_questions()
    assert len(samples) == 5


def test_load_sample_questions_valid(scorer):
    for q in scorer.load_sample_questions():
        assert isinstance(q, ARCQuestion)
        assert q.correct_key in ("A", "B", "C", "D")
        assert q.difficulty in ("Easy", "Challenge")


def test_registry_key():
    assert "default" in ARC_REGISTRY
    assert ARC_REGISTRY["default"] is ARCScorer
