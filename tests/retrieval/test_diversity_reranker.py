"""Unit tests for MMR / Jaccard diversity rerankers."""

from __future__ import annotations

import math

import pytest
import torch

from src.retrieval.diversity_reranker import (
    JaccardDiversityReranker,
    MMRReranker,
    cosine_similarity,
    jaccard_similarity,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _orth_embeddings() -> dict[str, torch.Tensor]:
    """Four pairwise-orthogonal unit vectors."""
    return {
        "a": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "b": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "c": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "d": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }


# --------------------------------------------------------------------------- #
# Similarity primitives                                                       #
# --------------------------------------------------------------------------- #


def test_cosine_identical_is_one():
    v = torch.tensor([0.3, -1.2, 4.5])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_is_zero():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_jaccard_identical_is_one():
    s = {"foo", "bar", "baz"}
    assert jaccard_similarity(s, s.copy()) == 1.0


def test_jaccard_disjoint_is_zero():
    assert jaccard_similarity({"a"}, {"b"}) == 0.0


# --------------------------------------------------------------------------- #
# MMR behavior                                                                #
# --------------------------------------------------------------------------- #


def test_lambda_one_preserves_relevance_order():
    emb = _orth_embeddings()
    ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5), ("d", 0.1)]
    rr = MMRReranker(lambda_=1.0)
    out = rr.rerank(ranked, emb, k=4)
    assert [d for d, _ in out] == ["a", "b", "c", "d"]
    # Scores under lambda=1 are just the relevance values.
    for (_, s), (_, r) in zip(out, ranked):
        assert s == pytest.approx(r, abs=1e-6)


def test_lambda_zero_maximizes_diversity():
    # Two near-duplicates ('a', 'a2') and two orthogonal alternatives.
    emb = {
        "a": torch.tensor([1.0, 0.0, 0.0]),
        "a2": torch.tensor([0.999, 0.001, 0.0]),
        "b": torch.tensor([0.0, 1.0, 0.0]),
        "c": torch.tensor([0.0, 0.0, 1.0]),
    }
    ranked = [("a", 0.9), ("a2", 0.89), ("b", 0.5), ("c", 0.4)]
    rr = MMRReranker(lambda_=0.0)
    out = rr.rerank(ranked, emb, k=3)
    ids = [d for d, _ in out]
    assert ids[0] == "a"  # first pick: ties on 0 => highest relevance first
    # After 'a', 'a2' is near-duplicate => must NOT be the next pick.
    assert "a2" not in ids[1:2]


def test_mmr_picks_diverse_over_redundant_at_mid_lambda():
    emb = {
        "a": torch.tensor([1.0, 0.0]),
        "a_dup": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.0, 1.0]),
    }
    ranked = [("a", 1.0), ("a_dup", 0.95), ("b", 0.6)]
    rr = MMRReranker(lambda_=0.5)
    out = rr.rerank(ranked, emb, k=2)
    ids = [d for d, _ in out]
    assert ids[0] == "a"
    assert ids[1] == "b"  # 'a_dup' fully penalized


def test_k_greater_than_ranked_returns_all():
    emb = _orth_embeddings()
    ranked = [("a", 0.9), ("b", 0.5)]
    rr = MMRReranker(lambda_=0.7)
    out = rr.rerank(ranked, emb, k=10)
    assert len(out) == 2
    assert {d for d, _ in out} == {"a", "b"}


def test_empty_ranked_returns_empty():
    rr = MMRReranker(lambda_=0.5)
    assert rr.rerank([], {}, k=5) == []


def test_missing_embedding_raises():
    emb = {"a": torch.tensor([1.0, 0.0])}
    ranked = [("a", 0.9), ("missing", 0.5)]
    rr = MMRReranker(lambda_=0.5)
    with pytest.raises(KeyError):
        rr.rerank(ranked, emb, k=2)


def test_invalid_lambda_raises():
    with pytest.raises(ValueError):
        MMRReranker(lambda_=1.5)
    with pytest.raises(ValueError):
        MMRReranker(lambda_=-0.1)
    with pytest.raises(ValueError):
        MMRReranker(lambda_=float("nan"))


def test_unknown_similarity_raises():
    with pytest.raises(ValueError):
        MMRReranker(lambda_=0.5, similarity="dot")
    with pytest.raises(ValueError):
        MMRReranker(lambda_=0.5, similarity="euclidean")


def test_determinism():
    emb = _orth_embeddings()
    ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5), ("d", 0.3)]
    rr = MMRReranker(lambda_=0.6)
    out1 = rr.rerank(ranked, emb, k=4)
    out2 = rr.rerank(ranked, emb, k=4)
    assert out1 == out2


def test_hand_computed_three_doc_example():
    """Concrete 3-doc MMR trace, lambda=0.5.

    Embeddings:
        d1 = [1, 0]          rel=1.0
        d2 = [0.6, 0.8]      rel=0.8   (cos(d1,d2)=0.6)
        d3 = [0, 1]          rel=0.5   (cos(d1,d3)=0.0, cos(d2,d3)=0.8)

    Step 1 (S=empty): scores = lambda*rel = [0.5, 0.4, 0.25] -> pick d1, score 0.5
    Step 2 (S={d1}):
        d2: 0.5*0.8 - 0.5*0.6 = 0.4 - 0.3 = 0.10
        d3: 0.5*0.5 - 0.5*0.0 = 0.25 - 0.0 = 0.25
        -> pick d3, score 0.25
    Step 3 (S={d1,d3}):
        d2: 0.5*0.8 - 0.5*max(0.6, 0.8) = 0.4 - 0.4 = 0.0
        -> pick d2, score 0.0
    """
    emb = {
        "d1": torch.tensor([1.0, 0.0]),
        "d2": torch.tensor([0.6, 0.8]),
        "d3": torch.tensor([0.0, 1.0]),
    }
    ranked = [("d1", 1.0), ("d2", 0.8), ("d3", 0.5)]
    rr = MMRReranker(lambda_=0.5)
    out = rr.rerank(ranked, emb, k=3)
    assert [d for d, _ in out] == ["d1", "d3", "d2"]
    assert out[0][1] == pytest.approx(0.5, abs=1e-6)
    assert out[1][1] == pytest.approx(0.25, abs=1e-6)
    assert out[2][1] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Jaccard reranker                                                            #
# --------------------------------------------------------------------------- #


def test_jaccard_diversifies_identical_docs():
    docs = {
        "x1": "the quick brown fox",
        "x2": "the quick brown fox",  # lexical duplicate
        "y": "completely different words here",
    }
    ranked = [("x1", 1.0), ("x2", 0.95), ("y", 0.4)]
    rr = JaccardDiversityReranker(lambda_=0.5)
    out = rr.rerank(ranked, docs, k=2)
    ids = [d for d, _ in out]
    assert ids[0] == "x1"
    # Duplicate must lose to 'y' under diversity penalty.
    assert ids[1] == "y"


def test_jaccard_invalid_lambda_raises():
    with pytest.raises(ValueError):
        JaccardDiversityReranker(lambda_=2.0)


def test_jaccard_empty_ranked():
    rr = JaccardDiversityReranker(lambda_=0.5)
    assert rr.rerank([], {}, k=3) == []


def test_jaccard_missing_doc_raises():
    rr = JaccardDiversityReranker(lambda_=0.5)
    with pytest.raises(KeyError):
        rr.rerank([("a", 1.0)], {}, k=1)
