"""Integration tests: hard_negative_miner exposed via src.retrieval.

Verifies that prior __init__ exports survive intact and that an
end-to-end mining run with a real :class:`BM25Retriever` produces
sensible negatives.
"""

from __future__ import annotations

import pytest

import src.retrieval as retrieval_pkg
from src.retrieval import (
    HARD_NEGATIVE_STRATEGIES,
    BM25Retriever,
    HardNegative,
    HardNegativeMiner,
)


def test_package_exports_new_surface() -> None:
    assert HardNegativeMiner is retrieval_pkg.HardNegativeMiner
    assert HardNegative is retrieval_pkg.HardNegative
    assert set(HARD_NEGATIVE_STRATEGIES) == {
        "bm25_hard",
        "embedding_hard",
        "in_batch",
    }


def test_prior_exports_intact() -> None:
    # Spot-check a broad set of previously-exported names so an accidental
    # rewrite of __init__.py that drops any of them will fail here.
    for name in [
        "BM25Retriever",
        "Chunk",
        "CorpusIndexer",
        "HybridRetriever",
        "RETRIEVER_REGISTRY",
        "EMBEDDING_REGISTRY",
        "RERANKER_REGISTRY",
        "reciprocal_rank_fusion",
        "borda_count",
        "comb_sum",
        "comb_mnz",
        "fuse",
        "FUSION_REGISTRY",
        "DenseEmbedder",
        "EmbedderConfig",
        "EmbeddingTrainer",
        "InfoNCELoss",
        "CodeAwareTokenizer",
        "KEYWORDS",
        "SUPPORTED_LANGUAGES",
        "INSTRUCTION_PREFIXES",
        "InstructionPrefixEmbedder",
        "MMRReranker",
        "JaccardDiversityReranker",
        "cosine_similarity",
        "jaccard_similarity",
    ]:
        assert hasattr(retrieval_pkg, name), f"retrieval package lost {name}"


def test_end_to_end_bm25_hard_negatives() -> None:
    corpus = [
        "neural networks are trained with gradient descent",
        "gradient descent optimizes loss functions",
        "transformers use self-attention mechanisms",
        "self-attention scales quadratically with sequence length",
        "the weather is nice today",
        "pizza is delicious with cheese",
    ]
    retr = BM25Retriever()
    retr.add_documents(corpus)
    miner = HardNegativeMiner(retriever=retr, strategy="bm25_hard", k=3)

    query = "neural networks trained with gradient descent"
    positive = corpus[0]
    negatives = miner.mine(query, positive, corpus)

    assert 1 <= len(negatives) <= 3
    ids = [n.doc_id for n in negatives]
    assert positive not in ids
    # All returned docs must come from the corpus.
    assert all(n.doc_id in corpus for n in negatives)
    # The "gradient descent" doc shares "gradient" and "descent" with the
    # query; it should surface as a hard negative ahead of unrelated
    # weather / pizza docs.
    assert negatives[0].doc_id == "gradient descent optimizes loss functions"
    # Scores must be descending.
    scores = [n.score for n in negatives]
    assert scores == sorted(scores, reverse=True)


def test_end_to_end_mine_batch_and_in_batch() -> None:
    corpus = [
        "apples are red fruit",
        "bananas are yellow fruit",
        "cars drive on roads",
        "boats sail on water",
    ]
    retr = BM25Retriever()
    retr.add_documents(corpus)
    miner = HardNegativeMiner(retriever=retr, strategy="bm25_hard", k=2)

    batch = miner.mine_batch(
        queries=["red fruit", "yellow fruit"],
        positive_doc_ids=[corpus[0], corpus[1]],
        corpus=corpus,
    )
    assert len(batch) == 2
    assert all(isinstance(lst, list) for lst in batch)
    assert corpus[0] not in [n.doc_id for n in batch[0]]
    assert corpus[1] not in [n.doc_id for n in batch[1]]

    # in-batch uses a freshly constructed miner (retriever not required).
    ib_miner = HardNegativeMiner(strategy="in_batch", k=1)
    pairs = [("q1", corpus[0]), ("q2", corpus[1]), ("q3", corpus[2])]
    ib = ib_miner.mine_in_batch(pairs)
    assert [len(x) for x in ib] == [2, 2, 2]


def test_bad_strategy_is_friendly() -> None:
    with pytest.raises(ValueError):
        HardNegativeMiner(strategy="not_a_real_strategy")
