"""Tests for HybridRetriever (RRF + weighted fusion)."""

from __future__ import annotations

import math
import time

import pytest

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, _CosineDenseRetriever


# --------------------------------------------------------------------------- #
# Deterministic embed fns used throughout the tests.                          #
# --------------------------------------------------------------------------- #


def char_count_embed(text: str) -> list[float]:
    """Deterministic 3-D embedding: [#a, #b, #c] lowercased.

    Deterministic and non-lexical: a doc that mentions "apples" a lot will
    embed close to a query that mentions "bananas" if the a/b/c character
    counts happen to align. We exploit this to craft a corpus where dense
    beats sparse on a targeted query.
    """
    t = text.lower()
    return [float(t.count("a")), float(t.count("b")), float(t.count("c"))]


def ascii_sum_embed(text: str) -> list[float]:
    """A second simple 2-D embed: [len, sum_ord_mod_7]."""
    return [float(len(text)), float(sum(ord(c) for c in text) % 7)]


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_bm25(docs):
    r = BM25Retriever()
    return r  # caller will add via hybrid.add_documents


@pytest.fixture
def corpus():
    # Crafted: doc 2 ("a a a b") shares character-profile with query "aaab"
    # but has zero lexical overlap in BM25 sense with the query "apples".
    return [
        "apples oranges bananas",       # 0
        "the quick brown fox",          # 1
        "a a a b",                      # 2  (char-profile matches "aaab")
        "completely unrelated stuff",   # 3
        "banana apple apple",           # 4
    ]


# --------------------------------------------------------------------------- #
# 1. Hybrid matches-or-beats sparse on a crafted corpus.                      #
# --------------------------------------------------------------------------- #


def test_hybrid_beats_sparse_on_crafted_corpus(corpus):
    # Query "aaab" has no lexical match against doc 2 via BM25's word
    # tokenizer (doc 2 tokens: ["a","a","a","b"]; query tokens: ["aaab"]).
    # But char_count_embed makes doc 2 the nearest neighbor.
    sparse = BM25Retriever()
    hybrid = HybridRetriever(
        sparse_retriever=sparse, embed_fn=char_count_embed, fusion="rrf"
    )
    hybrid.add_documents(corpus)
    sparse_only = BM25Retriever()
    sparse_only.add_documents(corpus)

    q = "aaab"
    hybrid_hits = hybrid.query(q, k=3)
    sparse_hits = sparse_only.query(q, k=3)

    # Sparse alone returns nothing lexical for "aaab".
    assert sparse_hits == []
    # Hybrid surfaces doc 2 via the dense channel.
    assert hybrid_hits, "hybrid should return something when dense matches"
    assert hybrid_hits[0][0] == 2


# --------------------------------------------------------------------------- #
# 2. RRF formula verified on hand-computed example.                           #
# --------------------------------------------------------------------------- #


def test_rrf_formula_hand_computed():
    # Sparse ranking: [A, B, C] -> ranks 1,2,3
    # Dense  ranking: [B, D, A] -> ranks 1,2,3
    # With k=60:
    #   A: 1/61 + 1/63
    #   B: 1/62 + 1/61
    #   C: 1/63 + 0
    #   D: 0    + 1/62
    sparse_hits = [(10, 0.9), (20, 0.8), (30, 0.7)]  # A=10 B=20 C=30
    dense_hits = [(20, 0.9), (40, 0.5), (10, 0.4)]   # D=40
    fused = HybridRetriever._rrf(sparse_hits, dense_hits, k_rrf=60)

    assert math.isclose(fused[10], 1 / 61 + 1 / 63, abs_tol=1e-6)
    assert math.isclose(fused[20], 1 / 62 + 1 / 61, abs_tol=1e-6)
    assert math.isclose(fused[30], 1 / 63, abs_tol=1e-6)
    assert math.isclose(fused[40], 1 / 62, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# 3. Weighted fusion formula verified.                                        #
# --------------------------------------------------------------------------- #


def test_weighted_fusion_formula():
    # Sparse: [(10, 3.0), (20, 1.0)]  -> minmax: {10:1.0, 20:0.0}
    # Dense:  [(20, 0.9), (30, 0.1)]  -> minmax: {20:1.0, 30:0.0}
    # w=(0.5,0.5):
    #   10: 0.5*1.0
    #   20: 0.5*0.0 + 0.5*1.0
    #   30: 0.5*0.0
    sparse = BM25Retriever()

    class _Dense:
        def add_documents(self, docs):
            self.n = len(docs)

        def query(self, q, k=10):
            return [(20, 0.9), (30, 0.1)]

    class _Sparse(BM25Retriever):
        def query(self, q, k=10):
            return [(10, 3.0), (20, 1.0)]

    s = _Sparse()
    d = _Dense()
    h = HybridRetriever(
        sparse_retriever=s,
        dense_retriever=d,
        fusion="weighted",
        weights=(0.5, 0.5),
    )
    h.add_documents(["a a", "b b", "c c", "d d"])  # needed by BM25 base
    hits = dict(h.query("anything", k=10))
    assert math.isclose(hits[10], 0.5, abs_tol=1e-6)
    assert math.isclose(hits[20], 0.5, abs_tol=1e-6)
    assert math.isclose(hits[30], 0.0, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# 4. Top-k ordering stable.                                                   #
# --------------------------------------------------------------------------- #


def test_topk_ordering_stable(corpus):
    sparse = BM25Retriever()
    h = HybridRetriever(sparse, embed_fn=char_count_embed)
    h.add_documents(corpus)
    r1 = h.query("apple banana", k=5)
    r2 = h.query("apple banana", k=5)
    assert r1 == r2
    # scores monotonically non-increasing
    scores = [s for _i, s in r1]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# 5. k > corpus size returns at most N items.                                 #
# --------------------------------------------------------------------------- #


def test_k_greater_than_corpus(corpus):
    sparse = BM25Retriever()
    h = HybridRetriever(sparse, embed_fn=char_count_embed)
    h.add_documents(corpus)
    hits = h.query("apple banana", k=1000)
    assert len(hits) <= len(corpus)
    assert len(hits) >= 1


# --------------------------------------------------------------------------- #
# 6. Empty query returns [].                                                  #
# --------------------------------------------------------------------------- #


def test_empty_query_returns_empty(corpus):
    sparse = BM25Retriever()
    h = HybridRetriever(sparse, embed_fn=char_count_embed)
    h.add_documents(corpus)
    assert h.query("", k=5) == []


# --------------------------------------------------------------------------- #
# 7. Determinism across fresh retriever instances.                            #
# --------------------------------------------------------------------------- #


def test_determinism_across_instances(corpus):
    h1 = HybridRetriever(BM25Retriever(), embed_fn=char_count_embed)
    h1.add_documents(corpus)
    h2 = HybridRetriever(BM25Retriever(), embed_fn=char_count_embed)
    h2.add_documents(corpus)
    assert h1.query("apples bananas", k=5) == h2.query("apples bananas", k=5)


# --------------------------------------------------------------------------- #
# 8. Sparse-only mode (weights=(1,0)) with fusion='weighted' == BM25.         #
# --------------------------------------------------------------------------- #


def test_sparse_only_weighted_matches_bm25(corpus):
    sparse = BM25Retriever()
    h = HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=None,
        embed_fn=None,
        fusion="weighted",
        weights=(1.0, 0.0),
    )
    h.add_documents(corpus)

    plain = BM25Retriever()
    plain.add_documents(corpus)

    hy = h.query("apple banana", k=5)
    bm = plain.query("apple banana", k=5)
    # Same doc_id ordering; scores differ (hybrid uses min-max norm) but ranks
    # must match exactly.
    assert [d for d, _ in hy] == [d for d, _ in bm]


# --------------------------------------------------------------------------- #
# 9. embed_fn rejected if not callable.                                       #
# --------------------------------------------------------------------------- #


def test_embed_fn_must_be_callable():
    with pytest.raises(TypeError):
        HybridRetriever(BM25Retriever(), embed_fn="not a function")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 10. Invalid fusion name raises.                                             #
# --------------------------------------------------------------------------- #


def test_invalid_fusion_raises():
    with pytest.raises(ValueError):
        HybridRetriever(BM25Retriever(), embed_fn=char_count_embed, fusion="bogus")


# --------------------------------------------------------------------------- #
# 11. add_documents with empty list raises.                                   #
# --------------------------------------------------------------------------- #


def test_add_documents_empty_raises():
    h = HybridRetriever(BM25Retriever(), embed_fn=char_count_embed)
    with pytest.raises(ValueError):
        h.add_documents([])


# --------------------------------------------------------------------------- #
# 12. Performance: 100-doc corpus query in <2s.                               #
# --------------------------------------------------------------------------- #


def test_performance_100_docs_under_2s():
    import random

    random.seed(0)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    docs = [" ".join(random.choice(vocab) for _ in range(20)) for _ in range(100)]
    h = HybridRetriever(BM25Retriever(), embed_fn=char_count_embed)
    h.add_documents(docs)
    t0 = time.perf_counter()
    _ = h.query("alpha beta gamma", k=10)
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"hybrid query took {dt:.3f}s on 100 docs"


# --------------------------------------------------------------------------- #
# Bonus coverage: several edge cases caught in review.                        #
# --------------------------------------------------------------------------- #


def test_rrf_requires_dense_backend():
    # RRF without dense should raise at construction (no silent fallback).
    with pytest.raises(ValueError):
        HybridRetriever(BM25Retriever(), fusion="rrf")


def test_both_dense_and_embed_fn_rejected():
    sparse = BM25Retriever()
    dense = _CosineDenseRetriever(char_count_embed)
    with pytest.raises(ValueError):
        HybridRetriever(
            sparse, dense_retriever=dense, embed_fn=char_count_embed
        )


def test_double_add_documents_raises(corpus):
    h = HybridRetriever(BM25Retriever(), embed_fn=char_count_embed)
    h.add_documents(corpus)
    with pytest.raises(RuntimeError):
        h.add_documents(corpus)


def test_query_before_indexing_raises():
    h = HybridRetriever(BM25Retriever(), embed_fn=char_count_embed)
    with pytest.raises(RuntimeError):
        h.query("x", k=1)


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        HybridRetriever(
            BM25Retriever(), embed_fn=char_count_embed, weights=(-1.0, 1.0)
        )


def test_bad_k_rrf_rejected():
    with pytest.raises(ValueError):
        HybridRetriever(
            BM25Retriever(), embed_fn=char_count_embed, k_rrf=0
        )


def test_passed_in_dense_backend(corpus):
    sparse = BM25Retriever()
    dense = _CosineDenseRetriever(ascii_sum_embed)
    h = HybridRetriever(sparse, dense_retriever=dense, fusion="rrf")
    h.add_documents(corpus)
    hits = h.query("fox", k=3)
    assert hits  # ranked list is non-empty
    assert all(0 <= d < len(corpus) for d, _s in hits)
