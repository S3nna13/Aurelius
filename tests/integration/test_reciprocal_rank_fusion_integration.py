"""Integration tests for the standalone fusion utilities.

Covers three things:

1. Public re-export through :mod:`src.retrieval` (the four functions and
   the ``fuse`` dispatcher are importable from the package root).
2. Regression: the existing :class:`HybridRetriever` continues to work end-
   to-end after we appended new names to ``src/retrieval/__init__.py``.
3. Using ``fuse("rrf", ...)`` to combine the ranked outputs of two
   independent :class:`BM25Retriever` instances built on two different
   corpora (the canonical "fuse-two-BM25-shards" use case).
"""

from __future__ import annotations

from src.retrieval import (
    FUSION_REGISTRY,
    RETRIEVER_REGISTRY,
    BM25Retriever,
    HybridRetriever,
    borda_count,
    comb_mnz,
    comb_sum,
    fuse,
    reciprocal_rank_fusion,
)


def _embed(text: str) -> list[float]:
    t = text.lower()
    return [float(t.count(c)) for c in "abcdefghij"]


def test_all_public_symbols_importable_from_package():
    assert callable(reciprocal_rank_fusion)
    assert callable(borda_count)
    assert callable(comb_sum)
    assert callable(comb_mnz)
    assert callable(fuse)
    assert set(FUSION_REGISTRY) == {"rrf", "borda", "combsum", "combmnz"}


def test_hybrid_retriever_regression_still_works():
    # Appending to src/retrieval/__init__.py must not have broken the
    # existing retriever registry or the HybridRetriever construction path.
    assert RETRIEVER_REGISTRY["bm25"] is BM25Retriever
    assert RETRIEVER_REGISTRY["hybrid_rrf"] is HybridRetriever

    corpus = [
        "the cat sat on the mat",
        "deep learning with transformers",
        "bm25 is a bag of words ranking function",
        "reciprocal rank fusion combines rankers",
        "dense retrieval embeds text into vectors",
    ]
    hybrid = HybridRetriever(sparse_retriever=BM25Retriever(), embed_fn=_embed, fusion="rrf")
    hybrid.add_documents(corpus)
    hits = hybrid.query("rank fusion", k=3)
    assert len(hits) == 3
    # Output is still (int_id, float) pairs as before.
    for doc_id, score in hits:
        assert isinstance(doc_id, int)
        assert isinstance(score, float)


def test_fuse_rrf_combines_two_bm25_corpora_queries():
    # Build two independent BM25 shards. Each shard uses its own id space,
    # so we namespace doc ids with a shard prefix before fusing, which is
    # exactly how a real multi-shard retriever would use this surface.
    shard_a = [
        "reciprocal rank fusion is a classic late-fusion technique",
        "bm25 is a sparse lexical ranking function",
        "python standard library is battery included",
    ]
    shard_b = [
        "late fusion of ranked lists outperforms single rankers",
        "dense retrieval uses learned embeddings",
        "ranking functions combine into fused rankings",
    ]

    ret_a = BM25Retriever()
    ret_a.add_documents(shard_a)
    ret_b = BM25Retriever()
    ret_b.add_documents(shard_b)

    q = "rank fusion"
    hits_a = [(f"a:{i}", s) for i, s in ret_a.query(q, k=3)]
    hits_b = [(f"b:{i}", s) for i, s in ret_b.query(q, k=3)]

    assert len(hits_a) > 0 and len(hits_b) > 0

    fused = fuse("rrf", [hits_a, hits_b], k=60, top_n=5)
    assert len(fused) <= 5
    # Scores are strictly descending (ties allowed).
    for i in range(len(fused) - 1):
        assert fused[i][1] >= fused[i + 1][1]
    # Every returned id should come from one of the two shards.
    for doc_id, _ in fused:
        assert doc_id.startswith("a:") or doc_id.startswith("b:")
    # The two shard-0 docs (both directly about "rank fusion") must appear.
    returned_ids = {d for d, _ in fused}
    assert "a:0" in returned_ids
    assert "b:0" in returned_ids


def test_fuse_dispatcher_matches_direct_calls_integration():
    rankings = [
        [("doc1", 0.9), ("doc2", 0.6), ("doc3", 0.3)],
        [("doc2", 0.8), ("doc3", 0.5), ("doc4", 0.2)],
    ]
    assert fuse("rrf", rankings, k=60) == reciprocal_rank_fusion(rankings, k=60)
    assert fuse("borda", rankings) == borda_count(rankings)
    assert fuse("combsum", rankings) == comb_sum(rankings)
    assert fuse("combmnz", rankings) == comb_mnz(rankings)
