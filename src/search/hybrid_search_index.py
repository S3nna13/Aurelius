"""Hybrid search combining BM25 and semantic (TF-IDF) via Reciprocal Rank Fusion.

RRF from Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods". License: MIT.
"""

from __future__ import annotations

from .bm25_index import _MAX_DOC_ID_LEN, _MAX_TEXT_LEN, _MAX_TOP_K, BM25Index
from .semantic_search import SemanticSearch


class HybridSearchIndex:
    def __init__(
        self, rrf_k: int = 60, bm25_weight: float = 0.5, semantic_weight: float = 0.5
    ) -> None:
        if abs(bm25_weight + semantic_weight - 1.0) > 1e-6:
            raise ValueError("bm25_weight + semantic_weight must equal 1.0")
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self._bm25 = BM25Index()
        self._semantic = SemanticSearch()

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        if len(doc_id) > _MAX_DOC_ID_LEN:
            raise ValueError(f"doc_id exceeds {_MAX_DOC_ID_LEN} chars")
        if len(text) > _MAX_TEXT_LEN:
            raise ValueError(f"text exceeds {_MAX_TEXT_LEN} chars")
        self._bm25.add(doc_id, text, metadata)
        self._semantic.add(doc_id, text, metadata)

    def remove(self, doc_id: str) -> None:
        self._bm25.remove(doc_id)
        self._semantic.remove(doc_id)

    def _rrf_fuse(
        self, *rank_lists: list[tuple[str, float]], top_k: int
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranks in rank_lists:
            for rank, (doc_id, _) in enumerate(ranks, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > _MAX_TOP_K:
            raise ValueError(f"top_k must be <= {_MAX_TOP_K}")
        if len(self._bm25) == 0:
            return []
        if not text.strip():
            return []
        candidate_k = min(top_k * 2, len(self._bm25))
        bm25_results = self._bm25.query(text, top_k=candidate_k)
        semantic_results = self._semantic.query(text, top_k=candidate_k)
        return self._rrf_fuse(bm25_results, semantic_results, top_k=top_k)

    def query_bm25(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return self._bm25.query(text, top_k=top_k)

    def query_semantic(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return self._semantic.query(text, top_k=top_k)

    def __len__(self) -> int:
        return len(self._bm25)


HYBRID_INDEX = HybridSearchIndex()
