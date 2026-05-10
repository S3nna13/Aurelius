from __future__ import annotations

from collections.abc import Callable, Sequence


class CrossEncoderReranker:
    """Cross-encoder reranker for precision re-ranking.

    Uses a pluggable ``score_fn`` callable (typically a cross-encoder
    model) that takes a query-document pair and returns a relevance
    score.

    In production, connect to a Cohere/BGE/ColBERT reranker model.
    For development, the default scoring uses a simple token-overlap
    heuristic.
    """

    def __init__(
        self,
        score_fn: Callable[[str, str], float] | None = None,
        top_k: int = 10,
    ) -> None:
        self._score_fn = score_fn or self._default_score
        self.top_k = top_k

    @staticmethod
    def _default_score(query: str, doc: str) -> float:
        query_tokens = set(query.lower().split())
        doc_tokens = set(doc.lower().split())
        if not query_tokens:
            return 0.0
        intersection = query_tokens & doc_tokens
        return len(intersection) / len(query_tokens) if query_tokens else 0.0

    def rerank(
        self, query: str, documents: Sequence[str], doc_ids: Sequence[int] | None = None
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Returns list of ``(doc_id, score)`` sorted by descending score.
        """
        if not documents:
            return []

        scored: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            score = self._score_fn(query, doc)
            scored.append((idx, score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        k_eff = min(self.top_k, len(scored))
        return scored[:k_eff]

    def rerank_with_ids(
        self, query: str, documents: list[tuple[int, str]]
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        scored: list[tuple[int, float]] = []
        for doc_id, doc in documents:
            score = self._score_fn(query, doc)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        k_eff = min(self.top_k, len(scored))
        return scored[:k_eff]
