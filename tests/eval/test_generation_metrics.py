"""Tests for src/eval/generation_metrics.py"""
from __future__ import annotations

import pytest

from src.eval.generation_metrics import (
    GenerationEvaluator,
    GenerationStats,
    average_token_entropy,
    compression_ratio,
    coverage,
    density,
    distinct_n,
    generation_statistics,
    repetition_rate,
    self_bleu,
    vocabulary_coverage,
)


# ---------------------------------------------------------------------------
# distinct_n
# ---------------------------------------------------------------------------

def test_distinct_1_all_unique():
    """All different words -> distinct_1 should be 1.0."""
    texts = ["apple banana cherry", "dog elephant fox"]
    result = distinct_n(texts, n=1)
    assert result == pytest.approx(1.0)


def test_distinct_1_all_same():
    """Single word repeated many times -> distinct_1 should be very low (near 0)."""
    texts = ["the the the the the"]
    result = distinct_n(texts, n=1)
    # Only one unique unigram out of 5 total
    assert result == pytest.approx(1 / 5)


# ---------------------------------------------------------------------------
# repetition_rate
# ---------------------------------------------------------------------------

def test_repetition_rate_looping_text():
    """Looping text should have high repetition rate for 4-grams."""
    text = "the cat sat the cat sat the cat sat the cat sat"
    result = repetition_rate(text, n=4)
    # Lots of repeated 4-grams -> high repetition
    assert result > 0.5


# ---------------------------------------------------------------------------
# self_bleu
# ---------------------------------------------------------------------------

def test_self_bleu_identical():
    """Two identical texts -> high self_bleu (> 0.5)."""
    text = "the quick brown fox jumps over the lazy dog"
    result = self_bleu([text, text], n_gram=4)
    assert result > 0.5


def test_self_bleu_single_text():
    """Single text in list -> returns 0.0."""
    result = self_bleu(["hello world"], n_gram=4)
    assert result == 0.0


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_full():
    """Summary identical to source -> coverage = 1.0."""
    text = "the quick brown fox"
    result = coverage(text, text)
    assert result == pytest.approx(1.0)


def test_coverage_no_overlap():
    """Completely disjoint words -> coverage = 0.0."""
    source = "apple banana cherry"
    summary = "dog elephant fox"
    result = coverage(source, summary)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------

def test_density_extractive():
    """Summary copied verbatim from source -> density > 1.0."""
    source = "the cat sat on the mat and the dog ran fast"
    summary = "the cat sat on the mat"
    result = density(source, summary)
    # All 6 tokens form a single fragment of length 6: 6^2 / 6 = 6.0
    assert result > 1.0


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------

def test_compression_ratio_positive():
    """Longer source, shorter summary -> ratio > 1.0."""
    source = "this is a very long source document with many words"
    summary = "long source"
    result = compression_ratio(source, summary)
    assert result > 1.0


# ---------------------------------------------------------------------------
# generation_statistics
# ---------------------------------------------------------------------------

def test_generation_statistics_fields():
    """GenerationStats dataclass has all required fields with correct types."""
    texts = [
        "the cat sat on the mat",
        "a quick brown fox jumped over the lazy dog",
        "hello world",
    ]
    stats = generation_statistics(texts)
    assert isinstance(stats, GenerationStats)
    assert hasattr(stats, "mean_length")
    assert hasattr(stats, "std_length")
    assert hasattr(stats, "min_length")
    assert hasattr(stats, "max_length")
    assert hasattr(stats, "distinct_1")
    assert hasattr(stats, "distinct_2")
    assert hasattr(stats, "repetition_4gram")
    assert hasattr(stats, "self_bleu_4")
    # Sanity checks
    assert stats.min_length <= stats.mean_length <= stats.max_length
    assert stats.std_length >= 0.0


# ---------------------------------------------------------------------------
# vocabulary_coverage
# ---------------------------------------------------------------------------

def test_vocabulary_coverage_keys():
    """vocabulary_coverage returns dict with all required keys."""
    generated = ["the quick brown fox", "jumps over the lazy dog"]
    ref_vocab = {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"}
    result = vocabulary_coverage(generated, ref_vocab)
    assert "coverage_rate" in result
    assert "unknown_word_rate" in result
    assert "generated_vocab_size" in result
    assert isinstance(result["coverage_rate"], float)
    assert isinstance(result["unknown_word_rate"], float)
    assert isinstance(result["generated_vocab_size"], int)
    # All generated words are in ref_vocab -> unknown_word_rate should be 0
    assert result["unknown_word_rate"] == pytest.approx(0.0)
    # All ref vocab words are covered
    assert result["coverage_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# average_token_entropy
# ---------------------------------------------------------------------------

def test_average_token_entropy_positive():
    """Diverse texts with varied vocabulary -> entropy > 0."""
    texts = [
        "apple banana cherry date elderberry",
        "fig grape honeydew kiwi lemon",
        "mango nectarine orange papaya quince",
    ]
    result = average_token_entropy(texts)
    assert result > 0.0
