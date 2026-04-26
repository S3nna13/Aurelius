"""Tests for src/eval/seq2seq_metrics.py — 12 tests."""

from __future__ import annotations

import pytest

from src.eval.seq2seq_metrics import (
    Seq2SeqMetrics,
    chrf_score,
    corpus_seq2seq_metrics,
    evaluate_seq2seq,
    meteor_score,
    rouge_l,
    rouge_n,
    rouge_w,
    ter_score,
)

# ---------------------------------------------------------------------------
# 1. ROUGE-N identical
# ---------------------------------------------------------------------------


def test_rouge_n_identical():
    text = "the quick brown fox"
    result = rouge_n(text, text, n=2)
    assert result["rouge_n_precision"] == pytest.approx(1.0)
    assert result["rouge_n_recall"] == pytest.approx(1.0)
    assert result["rouge_n_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. ROUGE-N no overlap
# ---------------------------------------------------------------------------


def test_rouge_n_no_overlap():
    result = rouge_n("cat sat mat", "dog ran far", n=1)
    assert result["rouge_n_precision"] == pytest.approx(0.0)
    assert result["rouge_n_recall"] == pytest.approx(0.0)
    assert result["rouge_n_f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. ROUGE-N partial overlap
# ---------------------------------------------------------------------------


def test_rouge_n_partial():
    # "the cat" vs "the dog" — 1 unigram overlap ("the") out of 2
    result = rouge_n("the cat", "the dog", n=1)
    assert 0.0 < result["rouge_n_f1"] < 1.0
    assert result["rouge_n_precision"] == pytest.approx(0.5)
    assert result["rouge_n_recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. ROUGE-L identical
# ---------------------------------------------------------------------------


def test_rouge_l_identical():
    text = "hello world"
    result = rouge_l(text, text)
    assert result["rouge_l_precision"] == pytest.approx(1.0)
    assert result["rouge_l_recall"] == pytest.approx(1.0)
    assert result["rouge_l_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. ROUGE-W returns float
# ---------------------------------------------------------------------------


def test_rouge_w_returns_float():
    score = rouge_w("the quick brown fox", "the quick brown fox")
    assert isinstance(score, float)
    assert score == pytest.approx(1.0, abs=1e-6)

    score_partial = rouge_w("the quick brown fox", "a slow green cat")
    assert isinstance(score_partial, float)
    assert 0.0 <= score_partial <= 1.0


# ---------------------------------------------------------------------------
# 6. METEOR identical
# ---------------------------------------------------------------------------


def test_meteor_identical():
    score = meteor_score("the quick brown fox", "the quick brown fox")
    # Identical: all tokens matched, 1 contiguous chunk => fragmentation = 1/4
    # penalty = 0.5 * (1/4)^3 = 0.5 * 0.015625 = 0.0078125; score very close to 1
    assert score == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# 7. METEOR no overlap
# ---------------------------------------------------------------------------


def test_meteor_no_overlap():
    score = meteor_score("cat sat mat", "dog ran far")
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. chrF identical
# ---------------------------------------------------------------------------


def test_chrf_identical():
    score = chrf_score("hello world", "hello world")
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. chrF in [0, 1]
# ---------------------------------------------------------------------------


def test_chrf_range():
    pairs = [
        ("", ""),
        ("abc", "xyz"),
        ("the quick brown fox", "the slow brown dog"),
        ("a b c d e f", "a b c d e f"),
    ]
    for hyp, ref in pairs:
        score = chrf_score(hyp, ref)
        assert 0.0 <= score <= 1.0, f"chrf out of range for ({hyp!r}, {ref!r}): {score}"


# ---------------------------------------------------------------------------
# 10. TER identical
# ---------------------------------------------------------------------------


def test_ter_identical():
    score = ter_score("the cat sat on the mat", "the cat sat on the mat")
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. evaluate_seq2seq — all fields present and in valid range
# ---------------------------------------------------------------------------


def test_evaluate_seq2seq_all_metrics():
    hyp = "the quick brown fox jumps over the lazy dog"
    ref = "the fast brown fox leaps over the lazy dog"
    metrics = evaluate_seq2seq(hyp, ref)

    assert isinstance(metrics, Seq2SeqMetrics)
    assert hasattr(metrics, "rouge_1")
    assert hasattr(metrics, "rouge_2")
    assert hasattr(metrics, "rouge_l")
    assert hasattr(metrics, "meteor")
    assert hasattr(metrics, "chrf")
    assert hasattr(metrics, "ter")

    assert 0.0 <= metrics.rouge_1 <= 1.0
    assert 0.0 <= metrics.rouge_2 <= 1.0
    assert 0.0 <= metrics.rouge_l <= 1.0
    assert 0.0 <= metrics.meteor <= 1.0
    assert 0.0 <= metrics.chrf <= 1.0
    assert metrics.ter >= 0.0

    # overall property
    expected_overall = (
        metrics.rouge_1 + metrics.rouge_2 + metrics.rouge_l + metrics.meteor + metrics.chrf
    ) / 5
    assert metrics.overall == pytest.approx(expected_overall)


# ---------------------------------------------------------------------------
# 12. corpus_seq2seq_metrics — dict has all 6 keys
# ---------------------------------------------------------------------------


def test_corpus_metrics_keys():
    hypotheses = [
        "the cat sat on the mat",
        "hello world",
        "the quick brown fox",
    ]
    references = [
        "the cat sat on the mat",
        "hi world",
        "the quick brown dog",
    ]
    result = corpus_seq2seq_metrics(hypotheses, references)

    expected_keys = {"rouge_1", "rouge_2", "rouge_l", "meteor", "chrf", "ter"}
    assert set(result.keys()) == expected_keys

    for key, value in result.items():
        assert isinstance(value, float), f"Value for {key!r} is not float: {value}"
        assert value >= 0.0, f"Value for {key!r} is negative: {value}"
