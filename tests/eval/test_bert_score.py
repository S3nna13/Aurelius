"""Tests for BERTScore-style token-level embedding similarity metric."""

from __future__ import annotations

import pytest
import torch

from src.eval.bert_score import (
    EmbeddingBERTScore,
    bert_f1,
    bert_precision,
    bert_recall,
    compute_idf_weights,
    cosine_similarity_matrix,
)

# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------


def test_cosine_similarity_matrix_shape():
    a = torch.randn(4, 8)
    b = torch.randn(6, 8)
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (4, 6)


def test_cosine_similarity_identical():
    v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sim = cosine_similarity_matrix(v, v)
    # Diagonal should be 1.0
    assert sim[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert sim[1, 1].item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# bert_precision / bert_recall
# ---------------------------------------------------------------------------


def test_bert_precision_range():
    torch.manual_seed(42)
    cand = torch.randn(5, 16)
    ref = torch.randn(7, 16)
    p = bert_precision(cand, ref)
    assert 0.0 <= p <= 1.0 + 1e-5


def test_bert_recall_range():
    torch.manual_seed(42)
    cand = torch.randn(5, 16)
    ref = torch.randn(7, 16)
    r = bert_recall(cand, ref)
    assert 0.0 <= r <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# bert_f1
# ---------------------------------------------------------------------------


def test_bert_f1_harmonic_mean():
    p, r = 0.8, 0.6
    expected = 2 * p * r / (p + r)
    assert bert_f1(p, r) == pytest.approx(expected, abs=1e-6)


def test_bert_f1_zero():
    assert bert_f1(0.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EmbeddingBERTScore
# ---------------------------------------------------------------------------


@pytest.fixture
def small_scorer():
    torch.manual_seed(0)
    vocab_size, d_model = 50, 16
    embeddings = torch.randn(vocab_size, d_model)
    return EmbeddingBERTScore(embeddings)


def test_embedding_bert_score_keys(small_scorer):
    result = small_scorer.score([1, 2, 3], [1, 2, 3])
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


def test_embedding_bert_score_identical(small_scorer):
    ids = [1, 2, 3, 4, 5]
    result = small_scorer.score(ids, ids)
    assert result["f1"] == pytest.approx(1.0, abs=1e-5)


def test_batch_score_length(small_scorer):
    candidates = [[1, 2], [3, 4], [5, 6]]
    references = [[1, 2], [3, 4], [5, 6]]
    results = small_scorer.batch_score(candidates, references)
    assert len(results) == len(candidates)


def test_corpus_score_keys(small_scorer):
    candidates = [[1, 2, 3], [4, 5, 6]]
    references = [[1, 2, 3], [4, 5, 6]]
    result = small_scorer.corpus_score(candidates, references)
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


# ---------------------------------------------------------------------------
# compute_idf_weights
# ---------------------------------------------------------------------------


def test_idf_weights_shape():
    corpus = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    vocab_size = 10
    idf = compute_idf_weights(corpus, vocab_size)
    assert idf.shape == (vocab_size,)


def test_idf_rare_word_higher():
    # Token 1 appears in all 3 docs; token 99 appears in only 1 doc
    corpus = [
        [1, 2, 3],
        [1, 4, 5],
        [1, 99],
    ]
    vocab_size = 100
    idf = compute_idf_weights(corpus, vocab_size)
    # Token 1 is in 3/3 docs → low IDF; token 99 is in 1/3 docs → higher IDF
    assert idf[99].item() > idf[1].item()
