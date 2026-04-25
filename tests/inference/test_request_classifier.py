"""Tests for src/inference/request_classifier.py"""

from __future__ import annotations

import pytest

from src.inference.request_classifier import (
    ClassificationResult,
    ComplexityTier,
    RequestClassifier,
    TaskType,
)


@pytest.fixture
def clf() -> RequestClassifier:
    return RequestClassifier()


def test_classify_returns_classification_result(clf):
    result = clf.classify("Hello, how are you?")
    assert isinstance(result, ClassificationResult)


def test_classify_code_prompt(clf):
    result = clf.classify("def add(a, b): return a + b")
    assert result.task_type == TaskType.CODE


def test_classify_math_prompt(clf):
    result = clf.classify("Please solve the equation x^2 + 5 = 0")
    assert result.task_type == TaskType.MATH


def test_classify_summarize_prompt(clf):
    result = clf.classify("Please summarize this article for me")
    assert result.task_type == TaskType.SUMMARIZE


def test_classify_translate_prompt(clf):
    result = clf.classify("Translate the following text in Spanish")
    assert result.task_type == TaskType.TRANSLATE


def test_classify_reasoning_prompt(clf):
    result = clf.classify("Why does water expand when it freezes? Explain the reason.")
    assert result.task_type == TaskType.REASONING


def test_classify_retrieval_prompt(clf):
    result = clf.classify("What is the capital of France?")
    assert result.task_type == TaskType.RETRIEVAL


def test_classify_unknown_returns_chat(clf):
    result = clf.classify("blorp zork flarb snorp")
    assert result.task_type == TaskType.CHAT


def test_complexity_low_short_prompt(clf):
    result = clf.classify("Hi there")
    assert result.complexity == ComplexityTier.LOW


def test_complexity_medium_mid_length(clf):
    words = " ".join(["word"] * 300)
    result = clf.classify(words)
    assert result.complexity == ComplexityTier.MEDIUM


def test_complexity_high_long_prompt(clf):
    words = " ".join(["word"] * 900)
    result = clf.classify(words)
    assert result.complexity == ComplexityTier.HIGH


def test_token_estimate_scales_with_word_count(clf):
    short = clf.classify("hello world")
    long = clf.classify(" ".join(["word"] * 100))
    assert long.token_estimate > short.token_estimate


def test_confidence_in_range(clf):
    result = clf.classify("def foo(): pass")
    assert 0.0 <= result.confidence <= 1.0


def test_batch_classify_returns_list(clf):
    prompts = ["def foo(): pass", "solve x + 1 = 0", "summarize this"]
    results = clf.batch_classify(prompts)
    assert len(results) == 3
    assert all(isinstance(r, ClassificationResult) for r in results)


def test_batch_classify_preserves_order(clf):
    prompts = ["def foo(): pass", "translate in Spanish"]
    results = clf.batch_classify(prompts)
    assert results[0].task_type == TaskType.CODE
    assert results[1].task_type == TaskType.TRANSLATE


def test_batch_classify_empty(clf):
    results = clf.batch_classify([])
    assert results == []


def test_classify_empty_string(clf):
    result = clf.classify("")
    assert result.task_type == TaskType.CHAT
    assert result.token_estimate == 0
    assert result.complexity == ComplexityTier.LOW
