"""Integration tests for the hybrid retriever and the retrieval registry."""

from __future__ import annotations

from src.retrieval import RETRIEVER_REGISTRY, BM25Retriever, HybridRetriever


def _embed(text: str) -> list[float]:
    t = text.lower()
    return [float(t.count(c)) for c in "abcdefghij"]


def test_registry_has_bm25_and_hybrid_rrf():
    assert "bm25" in RETRIEVER_REGISTRY
    assert "hybrid_rrf" in RETRIEVER_REGISTRY
    assert RETRIEVER_REGISTRY["bm25"] is BM25Retriever
    assert RETRIEVER_REGISTRY["hybrid_rrf"] is HybridRetriever


def test_end_to_end_construct_via_registry():
    bm25_cls = RETRIEVER_REGISTRY["bm25"]
    hybrid_cls = RETRIEVER_REGISTRY["hybrid_rrf"]

    corpus = [
        "the cat sat on the mat",
        "deep learning with transformers",
        "bm25 is a bag of words ranking function",
        "reciprocal rank fusion combines rankers",
        "apples bananas cherries dates elderberries",
    ]

    sparse = bm25_cls()
    hybrid = hybrid_cls(sparse_retriever=sparse, embed_fn=_embed, fusion="rrf")
    hybrid.add_documents(corpus)

    hits = hybrid.query("reciprocal rank fusion", k=3)
    assert hits, "hybrid returned nothing for an in-corpus query"
    # The obviously-correct doc (index 3) should be ranked first.
    assert hits[0][0] == 3
    # All returned doc_ids must be in-range.
    assert all(0 <= d < len(corpus) for d, _s in hits)
    # Scores are ordered descending.
    scores = [s for _d, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_weighted_mode_end_to_end():
    corpus = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
    ]
    hybrid = HybridRetriever(
        sparse_retriever=BM25Retriever(),
        embed_fn=_embed,
        fusion="weighted",
        weights=(0.3, 0.7),
    )
    hybrid.add_documents(corpus)
    hits = hybrid.query("alpha", k=2)
    assert hits
    assert hits[0][0] == 0


def test_sparse_only_hybrid_matches_plain_bm25_end_to_end():
    corpus = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "alpha alpha epsilon",
    ]
    plain = BM25Retriever()
    plain.add_documents(corpus)
    hybrid = HybridRetriever(
        sparse_retriever=BM25Retriever(),
        fusion="weighted",
        weights=(1.0, 0.0),
    )
    hybrid.add_documents(corpus)
    assert [d for d, _ in hybrid.query("alpha", k=5)] == [d for d, _ in plain.query("alpha", k=5)]
