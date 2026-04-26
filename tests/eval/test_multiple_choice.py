"""Tests for multiple-choice benchmark scoring."""

import math
from unittest.mock import MagicMock

import pytest
import torch

from src.eval.multiple_choice import (
    BenchmarkResult,
    EvalResult,
    MultipleChoiceItem,
    evaluate_item,
    run_benchmark,
    score_completion,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def _fake_tokenizer():
    tok = MagicMock()
    tok.encode = lambda text: [ord(c) % 256 for c in text[:20]]
    return tok


def test_score_completion_returns_float(small_model):
    """score_completion must return a finite float."""
    tok = _fake_tokenizer()
    score = score_completion(small_model, tok, "The capital of France is", "Paris")
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert score < 0  # log-probability is always negative


def test_score_completion_empty_choice(small_model):
    """Empty completion must return -inf."""
    tok = MagicMock()
    tok.encode = lambda text: [] if text == "" else [1, 2, 3]
    score = score_completion(small_model, tok, "question", "")
    assert score == float("-inf")


def test_evaluate_item_returns_result(small_model):
    """evaluate_item must return EvalResult with correct fields."""
    tok = _fake_tokenizer()
    item = MultipleChoiceItem(
        question="The sky is",
        choices=["blue", "green", "red"],
        correct_idx=0,
    )
    result = evaluate_item(small_model, tok, item)

    assert isinstance(result, EvalResult)
    assert result.predicted_idx in range(3)
    assert isinstance(result.correct, bool)
    assert len(result.scores) == 3
    assert all(math.isfinite(s) for s in result.scores)


def test_evaluate_item_picks_highest_score(small_model):
    """predicted_idx must correspond to the highest score."""
    tok = _fake_tokenizer()
    item = MultipleChoiceItem(question="test", choices=["a", "b", "c"], correct_idx=0)
    result = evaluate_item(small_model, tok, item)
    assert result.predicted_idx == result.scores.index(max(result.scores))


def test_run_benchmark_accuracy(small_model):
    """run_benchmark must return accuracy in [0, 1]."""
    tok = _fake_tokenizer()
    items = [
        MultipleChoiceItem("Q1", ["a", "b"], 0),
        MultipleChoiceItem("Q2", ["c", "d"], 1),
        MultipleChoiceItem("Q3", ["e", "f"], 0),
    ]
    result = run_benchmark(small_model, tok, items, name="test")

    assert isinstance(result, BenchmarkResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_total == 3
    assert result.n_correct == sum(r.correct for r in result.results)


def test_run_benchmark_repr(small_model):
    """BenchmarkResult repr must be readable."""
    tok = _fake_tokenizer()
    items = [MultipleChoiceItem("Q", ["a", "b"], 0)]
    result = run_benchmark(small_model, tok, items, name="mytest")
    assert "mytest" in repr(result)
    assert "%" in repr(result)
