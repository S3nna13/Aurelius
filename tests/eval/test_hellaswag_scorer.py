"""Tests for HellaSwag scorer."""

from __future__ import annotations

import pytest

from src.eval.hellaswag_scorer import (
    HELLASWAG_REGISTRY,
    HellaSwagExample,
    HellaSwagResult,
    HellaSwagScorer,
)


@pytest.fixture
def scorer() -> HellaSwagScorer:
    return HellaSwagScorer()


@pytest.fixture
def example() -> HellaSwagExample:
    return HellaSwagExample(
        example_id="ex0",
        context="A chef is preparing a meal in the kitchen.",
        activity_label="Cooking",
        endings=[
            "She chops vegetables quickly.",
            "She reads a book about cooking.",
            "She leaves the kitchen.",
            "She orders takeout instead.",
        ],
        correct_idx=0,
    )


def test_parse_digit_answer(scorer, example):
    result = scorer.score_answer(example, "0")
    assert result.predicted_idx == 0
    assert result.is_correct is True


def test_parse_digit_answer_wrong(scorer, example):
    result = scorer.score_answer(example, "2")
    assert result.predicted_idx == 2
    assert result.is_correct is False


def test_parse_letter_answer_a(scorer, example):
    result = scorer.score_answer(example, "(A)")
    assert result.predicted_idx == 0
    assert result.is_correct is True


def test_parse_letter_answer_b(scorer, example):
    result = scorer.score_answer(example, "(B)")
    assert result.predicted_idx == 1
    assert result.is_correct is False


def test_parse_letter_answer_lowercase(scorer, example):
    result = scorer.score_answer(example, "(a)")
    assert result.predicted_idx == 0


def test_parse_text_match(scorer, example):
    result = scorer.score_answer(example, "She chops vegetables quickly.")
    assert result.predicted_idx == 0
    assert result.is_correct is True


def test_parse_text_match_picks_longest(scorer):
    ex = HellaSwagExample(
        example_id="ex1",
        context="Context.",
        activity_label="Test",
        endings=["run fast", "run", "walk slowly", "stand still"],
        correct_idx=0,
    )
    sc = HellaSwagScorer()
    result = sc.score_answer(ex, "They decided to run fast today")
    assert result.predicted_idx == 0


def test_default_to_zero_on_no_match(scorer, example):
    result = scorer.score_answer(example, "completely unrelated text xyz")
    assert result.predicted_idx == 0


def test_result_is_frozen(scorer, example):
    result = scorer.score_answer(example, "0")
    with pytest.raises((AttributeError, TypeError)):
        result.is_correct = False


def test_score_logprobs_argmax(scorer, example):
    logprobs = [-1.0, -0.5, -2.0, -3.0]
    result = scorer.score_logprobs(example, logprobs)
    assert result.predicted_idx == 1
    assert result.is_correct is False


def test_score_logprobs_correct(scorer, example):
    logprobs = [-0.1, -1.0, -2.0, -3.0]
    result = scorer.score_logprobs(example, logprobs)
    assert result.predicted_idx == 0
    assert result.is_correct is True


def test_score_batch_accuracy(scorer):
    examples = [
        HellaSwagExample("e0", "ctx", "label", ["a", "b", "c", "d"], 0),
        HellaSwagExample("e1", "ctx", "label", ["a", "b", "c", "d"], 1),
        HellaSwagExample("e2", "ctx", "label", ["a", "b", "c", "d"], 2),
        HellaSwagExample("e3", "ctx", "label", ["a", "b", "c", "d"], 3),
    ]
    answers = ["0", "1", "0", "3"]
    result = scorer.score_batch(examples, answers)
    assert result["n_total"] == 4
    assert result["n_correct"] == 3
    assert abs(result["accuracy"] - 0.75) < 1e-9


def test_load_sample_examples_returns_five(scorer):
    samples = scorer.load_sample_examples()
    assert len(samples) == 5


def test_load_sample_examples_valid(scorer):
    for ex in scorer.load_sample_examples():
        assert isinstance(ex, HellaSwagExample)
        assert len(ex.endings) == 4
        assert 0 <= ex.correct_idx <= 3


def test_registry_key():
    assert "default" in HELLASWAG_REGISTRY
    assert HELLASWAG_REGISTRY["default"] is HellaSwagScorer
