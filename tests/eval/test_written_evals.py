"""Tests for the written_evals module (model-written evaluation framework)."""

from __future__ import annotations

import pytest
import torch

from src.eval.written_evals import (
    EvalQuestion,
    EvalResult,
    WrittenEvalGenerator,
    WrittenEvalRunner,
    difficulty_score,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def _encode(text: str) -> list[int]:
    """Simple byte-level tokenizer (clamped to vocab_size=256)."""
    return [b % 256 for b in text.encode("utf-8", errors="replace")][:50]


def _make_generator(model) -> WrittenEvalGenerator:
    return WrittenEvalGenerator(
        model=model,
        tokenizer_encode=_encode,
        tokenizer_decode=None,
        max_seq_len=64,
    )


def _make_runner(model) -> WrittenEvalRunner:
    return WrittenEvalRunner(
        model=model,
        tokenizer_encode=_encode,
        max_seq_len=64,
    )


def _make_question(
    question_id: str = "reasoning_0_abc12345",
    question: str = "What is 2 + 2?",
    choices: list[str] | None = None,
    correct_choice: int = 0,
    category: str = "reasoning",
    difficulty: float = 0.3,
) -> EvalQuestion:
    if choices is None:
        choices = ["A) 4", "B) 3", "C) 5", "D) 6"]
    return EvalQuestion(
        question_id=question_id,
        question=question,
        choices=choices,
        correct_choice=correct_choice,
        category=category,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# Test 1: _build_generation_prompt contains category
# ---------------------------------------------------------------------------


def test_build_generation_prompt_contains_category(small_model):
    gen = _make_generator(small_model)
    prompt = gen._build_generation_prompt("mathematics", 0)
    assert "mathematics" in prompt


# ---------------------------------------------------------------------------
# Test 2: _parse_question valid
# ---------------------------------------------------------------------------


def test_parse_question_valid(small_model):
    gen = _make_generator(small_model)
    text = "Q: What is the boiling point of water?\nA) 100°C\nB) 0°C\nC) 50°C\nD) 200°C\nAnswer: A"
    result = gen._parse_question(text, "science", 0)
    assert result is not None
    assert isinstance(result, EvalQuestion)
    assert result.question == "What is the boiling point of water?"
    assert result.correct_choice == 0
    assert result.category == "science"
    assert len(result.choices) == 4


# ---------------------------------------------------------------------------
# Test 3: _parse_question invalid returns None
# ---------------------------------------------------------------------------


def test_parse_question_invalid_returns_none(small_model):
    gen = _make_generator(small_model)
    # Missing the Answer: line
    malformed = "This is not a properly formatted question at all."
    result = gen._parse_question(malformed, "reasoning", 0)
    assert result is None


# ---------------------------------------------------------------------------
# Test 4: eval_runner run returns results
# ---------------------------------------------------------------------------


def test_eval_runner_run_returns_results(small_model):
    runner = _make_runner(small_model)
    questions = [
        _make_question(question_id="reasoning_0_aaa", category="reasoning"),
        _make_question(question_id="factual_1_bbb", category="factual"),
        _make_question(question_id="math_2_ccc", category="math"),
    ]
    results = runner.run(questions)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, EvalResult)


# ---------------------------------------------------------------------------
# Test 5: confidence values are in [0, 1]
# ---------------------------------------------------------------------------


def test_eval_runner_confidence_in_range(small_model):
    runner = _make_runner(small_model)
    questions = [
        _make_question(question_id="reasoning_0_aaa", category="reasoning"),
        _make_question(question_id="factual_1_bbb", category="factual"),
        _make_question(question_id="math_2_ccc", category="math"),
    ]
    results = runner.run(questions)
    for r in results:
        assert 0.0 <= r.confidence <= 1.0, (
            f"confidence {r.confidence} out of [0, 1] for question {r.question_id}"
        )


# ---------------------------------------------------------------------------
# Test 6: aggregate accuracy in [0, 1]
# ---------------------------------------------------------------------------


def test_aggregate_accuracy_in_range(small_model):
    runner = _make_runner(small_model)
    questions = [
        _make_question(question_id="reasoning_0_aaa", category="reasoning"),
        _make_question(question_id="reasoning_1_bbb", category="reasoning"),
    ]
    results = runner.run(questions)
    metrics = runner.aggregate(results)
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Test 7: per_category has one key per category
# ---------------------------------------------------------------------------


def test_aggregate_per_category(small_model):
    runner = _make_runner(small_model)
    questions = [
        _make_question(question_id="reasoning_0_aaa", category="reasoning"),
        _make_question(question_id="reasoning_1_bbb", category="reasoning"),
        _make_question(question_id="factual_0_ccc", category="factual"),
        _make_question(question_id="factual_1_ddd", category="factual"),
    ]
    results = runner.run(questions)
    metrics = runner.aggregate(results)
    per_cat = metrics["per_category"]
    assert len(per_cat) == 2
    assert "reasoning" in per_cat
    assert "factual" in per_cat


# ---------------------------------------------------------------------------
# Test 8: difficulty_score
# ---------------------------------------------------------------------------


def test_difficulty_score():
    # All correct -> accuracy=1.0, difficulty=0.0
    all_correct = [
        EvalResult(
            question_id="q1", predicted_choice=0, correct=True, confidence=0.9, raw_output=""
        ),
        EvalResult(
            question_id="q2", predicted_choice=1, correct=True, confidence=0.8, raw_output=""
        ),
    ]
    assert difficulty_score(all_correct) == pytest.approx(0.0)

    # None correct -> accuracy=0.0, difficulty=1.0
    none_correct = [
        EvalResult(
            question_id="q1", predicted_choice=1, correct=False, confidence=0.7, raw_output=""
        ),
        EvalResult(
            question_id="q2", predicted_choice=2, correct=False, confidence=0.6, raw_output=""
        ),
    ]
    assert difficulty_score(none_correct) == pytest.approx(1.0)
