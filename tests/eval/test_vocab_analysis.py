"""Tests for src/eval/vocab_analysis.py"""
from __future__ import annotations

import math

import pytest

from src.eval.vocab_analysis import (
    TokenizerAnalyzer,
    VocabStats,
    analyze_token_frequency,
    compare_tokenizers,
    compute_compression_ratio,
    compute_coverage,
    compute_fertility,
    compute_zipf_exponent,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Simple byte-level tokenizer: each UTF-8 byte becomes one token id.
_byte_tokenizer = lambda text: list(text.encode("utf-8"))  # noqa: E731

# A "perfect" word tokenizer: assigns one integer id per word.
def _word_tokenizer(text: str) -> list[int]:
    words = text.split()
    return list(range(len(words)))  # arbitrary ids, one per word


# A tokenizer that emits two tokens per character (highly fragmented).
def _double_tokenizer(text: str) -> list[int]:
    ids = []
    for i, byte in enumerate(text.encode("utf-8")):
        ids.append(byte)
        ids.append(byte + 256)  # duplicate each byte as a second token
    return ids


# ---------------------------------------------------------------------------
# 1. VocabStats defaults
# ---------------------------------------------------------------------------

def test_vocab_stats_defaults():
    """VocabStats fields should have correct default values."""
    vs = VocabStats(vocab_size=1000)
    assert vs.vocab_size == 1000
    assert vs.n_special_tokens == 0
    assert vs.coverage == pytest.approx(0.0)
    assert vs.fertility == pytest.approx(0.0)
    assert vs.oov_rate == pytest.approx(0.0)
    assert vs.compression_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. compute_fertility — single word, perfect tokenizer → 1.0
# ---------------------------------------------------------------------------

def test_compute_fertility_single_word():
    """A word-level tokenizer on single-word texts should give fertility=1.0."""
    texts = ["hello", "world", "foo"]
    fertility = compute_fertility(texts, _word_tokenizer)
    assert fertility == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. compute_fertility — word split into 2 tokens → 2.0
# ---------------------------------------------------------------------------

def test_compute_fertility_fragmented():
    """_double_tokenizer emits 2 tokens per byte; for ASCII 1-char words → 2.0."""
    # Each text is a single 1-character word; _double_tokenizer emits 2 tokens.
    texts = ["a", "b", "c"]
    fertility = compute_fertility(texts, _double_tokenizer)
    assert fertility == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. compute_coverage — all words known → 1.0
# ---------------------------------------------------------------------------

def test_compute_coverage_all_known():
    """When every word in the texts is in vocab, coverage should be 1.0."""
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    all_words = set(w for t in texts for w in t.split())
    coverage = compute_coverage(texts, all_words)
    assert coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. compute_coverage — no words known → 0.0
# ---------------------------------------------------------------------------

def test_compute_coverage_none_known():
    """When vocab is empty, coverage should be 0.0."""
    texts = ["the quick brown fox"]
    coverage = compute_coverage(texts, set())
    assert coverage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. compute_compression_ratio — must be positive for non-empty texts
# ---------------------------------------------------------------------------

def test_compute_compression_ratio_positive():
    """Compression ratio must be strictly positive for non-empty input."""
    texts = ["hello world", "this is a test"]
    ratio = compute_compression_ratio(texts, _byte_tokenizer)
    assert ratio > 0.0


# ---------------------------------------------------------------------------
# 7. analyze_token_frequency — keys present
# ---------------------------------------------------------------------------

def test_analyze_token_frequency_keys():
    """Result dict must contain entropy, top10_coverage, hapax_ratio."""
    token_ids = list(range(100)) * 3  # 300 tokens, 100 unique
    result = analyze_token_frequency(token_ids, vocab_size=256)
    assert "entropy" in result
    assert "top10_coverage" in result
    assert "hapax_ratio" in result


# ---------------------------------------------------------------------------
# 8. analyze_token_frequency — entropy is positive for diverse input
# ---------------------------------------------------------------------------

def test_analyze_token_frequency_entropy_positive():
    """Entropy should be > 0 when there is more than one distinct token."""
    token_ids = [0, 1, 2, 3, 4] * 20
    result = analyze_token_frequency(token_ids, vocab_size=256)
    assert result["entropy"] > 0.0


# ---------------------------------------------------------------------------
# 9. compute_zipf_exponent — natural-like distribution → exponent ~ 1.0
# ---------------------------------------------------------------------------

def test_compute_zipf_exponent_natural():
    """A Zipf-distributed frequency list should yield exponent close to 1.0."""
    # Construct exact Zipf frequencies: freq(rank) = C / rank
    ranks = list(range(1, 101))
    freqs = [1000 // r for r in ranks]
    exponent = compute_zipf_exponent(freqs)
    # Allow generous tolerance since we use integer frequencies
    assert 0.5 <= exponent <= 1.5


# ---------------------------------------------------------------------------
# 10. TokenizerAnalyzer.analyze_corpus — returns VocabStats
# ---------------------------------------------------------------------------

def test_tokenizer_analyzer_corpus_type():
    """analyze_corpus should return a VocabStats instance."""
    analyzer = TokenizerAnalyzer(_byte_tokenizer, vocab_size=256)
    texts = ["hello world", "foo bar baz"]
    result = analyzer.analyze_corpus(texts)
    assert isinstance(result, VocabStats)


# ---------------------------------------------------------------------------
# 11. TokenizerAnalyzer.sequence_length_stats — keys present
# ---------------------------------------------------------------------------

def test_tokenizer_analyzer_sequence_stats_keys():
    """sequence_length_stats must return dict with mean, std, p50, p95."""
    analyzer = TokenizerAnalyzer(_byte_tokenizer, vocab_size=256)
    texts = ["short", "a slightly longer sentence here", "tiny"]
    stats = analyzer.sequence_length_stats(texts)
    assert "mean" in stats
    assert "std" in stats
    assert "p50" in stats
    assert "p95" in stats


# ---------------------------------------------------------------------------
# 12. compare_tokenizers — result dict contains winner key
# ---------------------------------------------------------------------------

def test_compare_tokenizers_winner_key():
    """compare_tokenizers result must contain a 'winner' key with value 'a' or 'b'."""
    texts = ["the quick brown fox jumps over the lazy dog"]
    result = compare_tokenizers(_byte_tokenizer, _double_tokenizer, texts)
    assert "winner" in result
    assert result["winner"] in ("a", "b")
