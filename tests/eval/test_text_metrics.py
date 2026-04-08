"""Tests for text generation evaluation metrics."""
from __future__ import annotations

import pytest
from src.eval.text_metrics import (
    normalize_text,
    exact_match,
    token_f1,
    lcs_length,
    rouge_l,
    ngrams,
    bleu,
    corpus_metrics,
)
from collections import Counter


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_lowercase():
    assert normalize_text("Hello World") == "hello world"


def test_normalize_text_removes_punctuation():
    assert normalize_text("hello, world!") == "hello world"


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

def test_exact_match_true():
    assert exact_match("the cat sat", "the cat sat") == 1.0


def test_exact_match_false():
    assert exact_match("the cat sat", "a dog ran") == 0.0


def test_exact_match_normalized():
    assert exact_match("Hello!", "hello") == 1.0


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

def test_token_f1_perfect():
    assert token_f1("a b c", "a b c") == pytest.approx(1.0)


def test_token_f1_no_overlap():
    assert token_f1("a b c", "d e f") == pytest.approx(0.0)


def test_token_f1_partial():
    # pred: a b c  ref: b c d
    # common = {b, c} = 2 tokens
    # precision = 2/3, recall = 2/3, F1 = 2/3
    result = token_f1("a b c", "b c d")
    assert result == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# lcs_length
# ---------------------------------------------------------------------------

def test_lcs_length_basic():
    assert lcs_length([1, 2, 3], [1, 3]) == 2


def test_lcs_length_empty():
    assert lcs_length([], [1, 2]) == 0


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------

def test_rouge_l_perfect():
    assert rouge_l("a b c d", "a b c d") == pytest.approx(1.0)


def test_rouge_l_no_overlap():
    assert rouge_l("a b c", "d e f") == pytest.approx(0.0)


def test_rouge_l_partial():
    # "a b c d" vs "b c": LCS = ["b", "c"] = 2
    # precision = 2/4 = 0.5, recall = 2/2 = 1.0, F1 = 2*0.5*1.0/1.5 = 2/3
    score = rouge_l("a b c d", "b c")
    assert score > 0.0


# ---------------------------------------------------------------------------
# ngrams
# ---------------------------------------------------------------------------

def test_ngrams_unigrams():
    result = ngrams(["a", "b", "a"], 1)
    expected = Counter({("a",): 2, ("b",): 1})
    assert result == expected


def test_ngrams_bigrams():
    result = ngrams(["a", "b", "c"], 2)
    expected = Counter({("a", "b"): 1, ("b", "c"): 1})
    assert result == expected


# ---------------------------------------------------------------------------
# bleu
# ---------------------------------------------------------------------------

def test_bleu_perfect():
    score = bleu("the cat sat on the mat", ["the cat sat on the mat"])
    assert score == pytest.approx(1.0, abs=1e-6)


def test_bleu_no_overlap():
    # No overlap at all; with smoothing, score should be > 0 but tiny
    score = bleu("a b c d", ["x y z w"], smooth=True)
    assert 0.0 < score < 0.01


def test_bleu_partial():
    score = bleu("the cat sat on the mat", ["the cat sat on a mat"])
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# corpus_metrics
# ---------------------------------------------------------------------------

def test_corpus_metrics_returns_all_keys():
    result = corpus_metrics(["hello world"], ["hello world"])
    assert set(result.keys()) == {"exact_match", "token_f1", "rouge_l", "bleu"}


def test_corpus_metrics_perfect():
    preds = ["the quick brown fox", "hello world"]
    refs = ["the quick brown fox", "hello world"]
    result = corpus_metrics(preds, refs)
    assert result["exact_match"] == pytest.approx(1.0)
    assert result["token_f1"] == pytest.approx(1.0)
    assert result["rouge_l"] == pytest.approx(1.0)
    assert result["bleu"] == pytest.approx(1.0)
