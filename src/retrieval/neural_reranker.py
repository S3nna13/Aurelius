"""Neural reranker: cross-encoder scoring, listwise reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RerankScore:
    query: str
    document: str
    score: float
    rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker using term-overlap proxy scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name

    def _score_pair(self, query: str, doc: str) -> float:
        query_terms = set(query.lower().split())
        doc_terms = set(doc.lower().split())
        overlap = query_terms & doc_terms
        return len(overlap) / max(1, len(query.split()))

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankScore]:
        scored = [(doc, self._score_pair(query, doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return [
            RerankScore(query=query, document=doc, score=score, rank=idx + 1)
            for idx, (doc, score) in enumerate(scored)
        ]

    def batch_rerank(
        self,
        query: str,
        doc_batches: list[list[str]],
    ) -> list[RerankScore]:
        flat: list[str] = []
        for batch in doc_batches:
            flat.extend(batch)
        return self.rerank(query, flat)


class ListwiseReranker:
    """Listwise reranker using softmax score normalisation."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def softmax_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        t = self.temperature
        shifted = [s / t for s in scores]
        max_val = max(shifted)
        exps = [math.exp(s - max_val) for s in shifted]
        total = sum(exps)
        return [e / total for e in exps]

    def rerank(
        self,
        documents: list[str],
        scores: list[float],
    ) -> list[tuple[str, float]]:
        soft = self.softmax_scores(scores)
        paired = list(zip(documents, soft))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired
