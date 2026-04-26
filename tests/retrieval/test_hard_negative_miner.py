"""Unit tests for :mod:`src.retrieval.hard_negative_miner`.

No foreign imports; only internal surfaces (BM25Retriever) and torch.
"""

from __future__ import annotations

import pytest
import torch

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hard_negative_miner import (
    STRATEGIES,
    HardNegative,
    HardNegativeMiner,
)

# ---------------------------------------------------------------------- #
# Fakes                                                                    #
# ---------------------------------------------------------------------- #


class _FakeEmbedder:
    """Deterministic bag-of-character embedder for tests.

    We map each text to a fixed-width vector of normalized character
    counts over a small alphabet. Identical inputs always yield identical
    outputs, which is what the determinism test depends on.
    """

    _ALPHA = "abcdefghijklmnopqrstuvwxyz "

    def encode(self, texts: list[str]) -> torch.Tensor:
        d = len(self._ALPHA)
        out = torch.zeros(len(texts), d, dtype=torch.float32)
        for i, t in enumerate(texts):
            for c in t.lower():
                j = self._ALPHA.find(c)
                if j >= 0:
                    out[i, j] += 1.0
        return out


# ---------------------------------------------------------------------- #
# Fixtures                                                                 #
# ---------------------------------------------------------------------- #


@pytest.fixture
def corpus() -> list[str]:
    return [
        "the cat sat on the mat",
        "the dog chased the cat",
        "birds fly in the sky",
        "a cat is a feline animal",
        "quantum chromodynamics is hard",
    ]


@pytest.fixture
def indexed_bm25(corpus: list[str]) -> BM25Retriever:
    r = BM25Retriever()
    r.add_documents(corpus)
    return r


# ---------------------------------------------------------------------- #
# Tests                                                                    #
# ---------------------------------------------------------------------- #


def test_bm25_hard_selects_topk_non_positive(
    indexed_bm25: BM25Retriever, corpus: list[str]
) -> None:
    miner = HardNegativeMiner(retriever=indexed_bm25, strategy="bm25_hard", k=2)
    pos = "the cat sat on the mat"
    negs = miner.mine("cat feline", pos, corpus)
    assert len(negs) == 2
    assert all(isinstance(n, HardNegative) for n in negs)
    assert all(n.doc_id != pos for n in negs)
    assert all(n.reason == "bm25_hard" for n in negs)
    # At least one of the cat-containing docs should surface.
    assert any("cat" in n.doc_id or "feline" in n.doc_id for n in negs)


def test_embedding_hard_selects_topk_nearest(corpus: list[str]) -> None:
    miner = HardNegativeMiner(embedder=_FakeEmbedder(), strategy="embedding_hard", k=3)
    pos = "the cat sat on the mat"
    negs = miner.mine("cat", pos, corpus)
    assert len(negs) == 3
    assert all(n.doc_id != pos for n in negs)
    assert all(n.reason == "embedding_hard" for n in negs)
    # Scores are cosine similarities in [-1, 1]; descending.
    scores = [n.score for n in negs]
    assert scores == sorted(scores, reverse=True)


def test_in_batch_returns_n_minus_one(corpus: list[str]) -> None:
    miner = HardNegativeMiner(strategy="in_batch", k=2)
    pairs = [
        ("q1", corpus[0]),
        ("q2", corpus[1]),
        ("q3", corpus[2]),
        ("q4", corpus[3]),
    ]
    batches = miner.mine_in_batch(pairs)
    assert len(batches) == 4
    for i, negs in enumerate(batches):
        assert len(negs) == 3
        ids = {n.doc_id for n in negs}
        assert pairs[i][1] not in ids
        assert all(n.reason == "in_batch" for n in negs)


def test_k_larger_than_corpus_minus_positive_returns_all(
    indexed_bm25: BM25Retriever, corpus: list[str]
) -> None:
    miner = HardNegativeMiner(retriever=indexed_bm25, strategy="bm25_hard", k=100)
    pos = corpus[0]
    # Query matches several but not all; BM25 only returns docs with
    # overlap. Use a broad query so all corpus docs get scored.
    negs = miner.mine("the cat dog bird quantum a is", pos, corpus)
    ids = {n.doc_id for n in negs}
    assert pos not in ids
    assert len(ids) <= len(corpus) - 1
    # Embedding-hard will always score the full corpus:
    emb_miner = HardNegativeMiner(embedder=_FakeEmbedder(), strategy="embedding_hard", k=100)
    negs2 = emb_miner.mine("anything", pos, corpus)
    assert len(negs2) == len(corpus) - 1


def test_missing_positive_raises(indexed_bm25: BM25Retriever, corpus: list[str]) -> None:
    miner = HardNegativeMiner(retriever=indexed_bm25, strategy="bm25_hard", k=2)
    with pytest.raises(ValueError, match="positive_doc_id"):
        miner.mine("q", "not in corpus at all", corpus)


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        HardNegativeMiner(strategy="magic", k=2)


def test_missing_retriever_for_bm25_raises(corpus: list[str]) -> None:
    miner = HardNegativeMiner(strategy="bm25_hard", k=2)
    with pytest.raises(ValueError, match="requires a retriever"):
        miner.mine("q", corpus[0], corpus)


def test_missing_embedder_for_embedding_hard_raises(corpus: list[str]) -> None:
    miner = HardNegativeMiner(strategy="embedding_hard", k=2)
    with pytest.raises(ValueError, match="requires an embedder"):
        miner.mine("q", corpus[0], corpus)


def test_determinism(corpus: list[str]) -> None:
    emb = _FakeEmbedder()
    miner_a = HardNegativeMiner(embedder=emb, strategy="embedding_hard", k=3)
    miner_b = HardNegativeMiner(embedder=emb, strategy="embedding_hard", k=3)
    a = miner_a.mine("cat", corpus[0], corpus)
    b = miner_b.mine("cat", corpus[0], corpus)
    assert [(n.doc_id, n.score) for n in a] == [(n.doc_id, n.score) for n in b]

    # BM25 determinism too.
    r1 = BM25Retriever()
    r1.add_documents(corpus)
    r2 = BM25Retriever()
    r2.add_documents(corpus)
    m1 = HardNegativeMiner(retriever=r1, strategy="bm25_hard", k=3)
    m2 = HardNegativeMiner(retriever=r2, strategy="bm25_hard", k=3)
    assert [n.doc_id for n in m1.mine("cat", corpus[0], corpus)] == [
        n.doc_id for n in m2.mine("cat", corpus[0], corpus)
    ]


def test_empty_corpus_raises(indexed_bm25: BM25Retriever) -> None:
    miner = HardNegativeMiner(retriever=indexed_bm25, strategy="bm25_hard", k=2)
    with pytest.raises(ValueError, match="corpus must be non-empty"):
        miner.mine("q", "x", [])


def test_mine_batch_returns_list_of_lists_with_correct_length(
    corpus: list[str],
) -> None:
    # Use embedding strategy so we don't have to pre-index.
    miner = HardNegativeMiner(embedder=_FakeEmbedder(), strategy="embedding_hard", k=2)
    queries = ["cat", "dog", "bird"]
    positives = [corpus[0], corpus[1], corpus[2]]
    out = miner.mine_batch(queries, positives, corpus)
    assert isinstance(out, list)
    assert len(out) == 3
    for negs, pos in zip(out, positives):
        assert isinstance(negs, list)
        assert len(negs) == 2
        assert all(n.doc_id != pos for n in negs)


def test_overlapping_positives_handled(corpus: list[str]) -> None:
    """A doc that is the positive for one query must still be mineable
    as a negative for a *different* query whose positive is a different
    doc.
    """
    miner = HardNegativeMiner(
        embedder=_FakeEmbedder(), strategy="embedding_hard", k=len(corpus) - 1
    )
    # For query 1, positive = corpus[0]; corpus[1] must be eligible.
    negs_for_0 = miner.mine("cat", corpus[0], corpus)
    ids_0 = {n.doc_id for n in negs_for_0}
    assert corpus[1] in ids_0
    assert corpus[0] not in ids_0

    # For query 2, positive = corpus[1]; corpus[0] must be eligible.
    negs_for_1 = miner.mine("dog", corpus[1], corpus)
    ids_1 = {n.doc_id for n in negs_for_1}
    assert corpus[0] in ids_1
    assert corpus[1] not in ids_1


def test_in_batch_requires_two_pairs() -> None:
    miner = HardNegativeMiner(strategy="in_batch", k=1)
    with pytest.raises(ValueError, match="at least 2 pairs"):
        miner.mine_in_batch([("q", "p")])


def test_mine_with_in_batch_strategy_via_mine_raises(
    corpus: list[str],
) -> None:
    miner = HardNegativeMiner(strategy="in_batch", k=1)
    with pytest.raises(ValueError, match="mine_in_batch"):
        miner.mine("q", corpus[0], corpus)


def test_strategies_constant_exposed() -> None:
    assert set(STRATEGIES) == {"bm25_hard", "embedding_hard", "in_batch"}


def test_bad_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        HardNegativeMiner(strategy="bm25_hard", k=0)
    with pytest.raises(ValueError, match="k must be"):
        HardNegativeMiner(strategy="bm25_hard", k=-1)
