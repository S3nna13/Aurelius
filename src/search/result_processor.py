"""Search result deduplication and relevance reranking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    id: str
    score: float
    content: str
    source: str = ""


@dataclass
class ResultDeduplicator:
    """Remove near-duplicate search results by content similarity."""

    threshold: float = 0.85

    def deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        unique: list[SearchResult] = []
        for r in results:
            if not self._is_duplicate(r.content, [u.content for u in unique]):
                unique.append(r)
        return unique

    def _is_duplicate(self, content: str, existing: list[str]) -> bool:
        words = set(content.lower().split())
        if not words:
            return False
        for ex in existing:
            ex_words = set(ex.lower().split())
            jaccard = len(words & ex_words) / len(words | ex_words) if (words | ex_words) else 0
            if jaccard > self.threshold:
                return True
        return False


@dataclass
class ResultReranker:
    """Rerank results by combining original score with relevance boost."""

    def rerank(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        query_words = set(query.lower().split())
        scored = []
        for r in results:
            overlap = len(query_words & set(r.content.lower().split()))
            boost = overlap / max(len(query_words), 1) * 0.3
            scored.append((r, r.score + boost))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]


RESULT_DEDUP = ResultDeduplicator()
RESULT_RERANKER = ResultReranker()