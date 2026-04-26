"""Integration tests: MMR registration + BM25 -> MMR end-to-end pipeline."""

from __future__ import annotations

import torch

from src.retrieval import (
    RERANKER_REGISTRY,
    BM25Retriever,
    JaccardDiversityReranker,
    MMRReranker,
)


def test_registry_has_mmr_and_cross_encoder():
    assert "mmr" in RERANKER_REGISTRY
    assert RERANKER_REGISTRY["mmr"] is MMRReranker
    # Prior registrations must remain intact (additive append contract).
    assert "cross_encoder" in RERANKER_REGISTRY
    assert "jaccard_mmr" in RERANKER_REGISTRY
    assert RERANKER_REGISTRY["jaccard_mmr"] is JaccardDiversityReranker


def test_bm25_then_mmr_end_to_end():
    # Corpus of 4 docs: two near-duplicates about cats, two unique topics.
    corpus = [
        "cats are small furry felines that purr",  # 0
        "cats are small furry felines that purr loudly",  # 1 near-duplicate of 0
        "dogs bark loudly and chase balls",  # 2
        "python is a programming language",  # 3
    ]
    bm25 = BM25Retriever()
    bm25.add_documents(corpus)

    # BM25 ranks candidates by lexical match to the query.
    hits = bm25.query("furry felines purr loudly", k=4)
    assert len(hits) > 0

    # Construct simple bag-of-words embeddings so MMR can penalize the
    # near-duplicates on the cosine side.
    vocab = sorted({tok for doc in corpus for tok in doc.lower().split()})
    vocab_idx = {t: i for i, t in enumerate(vocab)}

    def embed(text: str) -> torch.Tensor:
        v = torch.zeros(len(vocab), dtype=torch.float32)
        for tok in text.lower().split():
            v[vocab_idx[tok]] += 1.0
        return v

    embeddings = {str(i): embed(corpus[i]) for i in range(len(corpus))}
    ranked = [(str(doc_id), float(score)) for doc_id, score in hits]

    rr = MMRReranker(lambda_=0.3, similarity="cosine")
    out = rr.rerank(ranked, embeddings, k=3)

    assert len(out) <= 3
    # The two cat docs are near-duplicates; MMR should not place them
    # back-to-back at the top under lambda=0.5.
    top_two = [d for d, _ in out[:2]]
    assert not (set(top_two) == {"0", "1"})


def test_registry_instantiation_via_name():
    # Callers typically look up by name; confirm the common path works.
    cls = RERANKER_REGISTRY["mmr"]
    inst = cls(lambda_=0.7, similarity="cosine")
    assert isinstance(inst, MMRReranker)
    emb = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.0, 1.0]),
    }
    out = inst.rerank([("a", 0.9), ("b", 0.4)], emb, k=2)
    assert [d for d, _ in out] == ["a", "b"]
