"""Integration tests: ColBERT scorer wired into RERANKER_REGISTRY."""

from __future__ import annotations

import torch

from src.retrieval import RERANKER_REGISTRY
from src.retrieval.colbert_late_interaction import ColBERTConfig, ColBERTScorer


def test_colbert_in_reranker_registry() -> None:
    assert "colbert" in RERANKER_REGISTRY
    assert RERANKER_REGISTRY["colbert"] is ColBERTScorer


def test_prior_registry_entries_intact() -> None:
    # Entries registered before colbert must still be present.
    for key in ("cross_encoder", "mmr", "jaccard_mmr"):
        assert key in RERANKER_REGISTRY, f"prior reranker '{key}' missing"


def test_registry_construct_and_score() -> None:
    cls = RERANKER_REGISTRY["colbert"]
    scorer = cls(ColBERTConfig(embed_dim=8, normalize=True))
    q = torch.randn(3, 8)
    score = scorer.score_pair(q, q.clone())
    # Identical tokens with L2 norm -> score == Nq.
    assert abs(score - 3.0) < 1e-4
