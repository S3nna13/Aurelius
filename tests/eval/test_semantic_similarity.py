"""Tests for src/eval/semantic_similarity.py"""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.semantic_similarity import (
    SemanticSimConfig,
    SemanticSimilarityEvaluator,
    bertscore_f1,
    bertscore_precision,
    bertscore_recall,
    compute_wmd_approx,
    cosine_similarity_matrix,
    extract_embeddings,
    ngram_overlap,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

# Byte-level tokenizer: encode a string as UTF-8 bytes, truncated to 256.
ENCODE_FN = lambda s: list(s.encode("utf-8", errors="replace"))[:256]


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(42)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture(scope="module")
def default_cfg():
    return SemanticSimConfig()


@pytest.fixture(scope="module")
def evaluator(tiny_model, default_cfg):
    return SemanticSimilarityEvaluator(tiny_model, ENCODE_FN, default_cfg)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SemanticSimConfig()
    assert cfg.pooling == "mean"
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# 2. test_extract_embeddings_shape
# ---------------------------------------------------------------------------

def test_extract_embeddings_shape(tiny_model, default_cfg):
    texts = ["hello world", "foo bar", "the quick brown fox"]
    embs = extract_embeddings(tiny_model, ENCODE_FN, texts, default_cfg)
    assert embs.shape == (len(texts), TINY_CFG.d_model)


# ---------------------------------------------------------------------------
# 3. test_extract_embeddings_normalized
# ---------------------------------------------------------------------------

def test_extract_embeddings_normalized(tiny_model):
    cfg = SemanticSimConfig(normalize=True)
    texts = ["alpha", "beta", "gamma"]
    embs = extract_embeddings(tiny_model, ENCODE_FN, texts, cfg)
    norms = embs.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# 4. test_cosine_similarity_matrix_shape
# ---------------------------------------------------------------------------

def test_cosine_similarity_matrix_shape():
    M, N, D = 3, 5, 16
    a = torch.randn(M, D)
    b = torch.randn(N, D)
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (M, N)


# ---------------------------------------------------------------------------
# 5. test_cosine_similarity_matrix_self
# ---------------------------------------------------------------------------

def test_cosine_similarity_matrix_self():
    D = 16
    a = torch.randn(4, D)
    sim = cosine_similarity_matrix(a, a)
    diag = sim.diag()
    assert torch.allclose(diag, torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. test_bertscore_precision_self
# ---------------------------------------------------------------------------

def test_bertscore_precision_self():
    embs = torch.randn(3, 16)
    embs = torch.nn.functional.normalize(embs, dim=-1)
    p = bertscore_precision(embs, embs)
    assert abs(p - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 7. test_bertscore_recall_self
# ---------------------------------------------------------------------------

def test_bertscore_recall_self():
    embs = torch.randn(3, 16)
    embs = torch.nn.functional.normalize(embs, dim=-1)
    r = bertscore_recall(embs, embs)
    assert abs(r - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 8. test_bertscore_f1_zero
# ---------------------------------------------------------------------------

def test_bertscore_f1_zero():
    assert bertscore_f1(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 9. test_bertscore_f1_range
# ---------------------------------------------------------------------------

def test_bertscore_f1_range():
    for p, r in [(0.5, 0.5), (0.3, 0.7), (1.0, 1.0), (0.0, 0.5)]:
        f = bertscore_f1(p, r)
        assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# 10. test_compute_wmd_approx_identical
# ---------------------------------------------------------------------------

def test_compute_wmd_approx_identical():
    words = ["hello", "world"]
    embs = {w: torch.nn.functional.normalize(torch.randn(16), dim=0) for w in words}
    dist = compute_wmd_approx(words, words, embs)
    assert abs(dist) < 1e-5


# ---------------------------------------------------------------------------
# 11. test_compute_wmd_approx_range
# ---------------------------------------------------------------------------

def test_compute_wmd_approx_range():
    words_a = ["apple", "orange"]
    words_b = ["banana", "grape"]
    all_words = words_a + words_b
    embs = {w: torch.nn.functional.normalize(torch.randn(16), dim=0) for w in all_words}
    dist = compute_wmd_approx(words_a, words_b, embs)
    assert 0.0 <= dist <= 1.0


# ---------------------------------------------------------------------------
# 12. test_ngram_overlap_identical
# ---------------------------------------------------------------------------

def test_ngram_overlap_identical():
    text = "hello world"
    score = ngram_overlap(text, text, n=2)
    assert abs(score - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 13. test_ngram_overlap_disjoint
# ---------------------------------------------------------------------------

def test_ngram_overlap_disjoint():
    # Use texts long enough but with no shared bigrams.
    score = ngram_overlap("aaaa", "bbbb", n=2)
    assert score == 0.0


# ---------------------------------------------------------------------------
# 14. test_evaluator_embedding_similarity_range
# ---------------------------------------------------------------------------

def test_evaluator_embedding_similarity_range(evaluator):
    sim = evaluator.compute_embedding_similarity("hello", "world")
    assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# 15. test_evaluator_bertscore_keys
# ---------------------------------------------------------------------------

def test_evaluator_bertscore_keys(evaluator):
    result = evaluator.compute_bertscore(
        candidates=["the cat sat on the mat"],
        references=["a cat sat on a mat"],
    )
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


# ---------------------------------------------------------------------------
# 16. test_evaluator_generation_quality_keys
# ---------------------------------------------------------------------------

def test_evaluator_generation_quality_keys(evaluator):
    result = evaluator.evaluate_generation_quality(
        generated=["generated text here", "another output"],
        references=["reference text here", "another reference"],
    )
    assert "mean_cosine_sim" in result
    assert "bertscore_f1" in result
    assert "mean_ngram_overlap" in result
    assert "n_pairs" in result
