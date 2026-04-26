from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class RerankStrategy(Enum):
    RELEVANCE = "relevance"
    RECENCY = "recency"
    HYBRID = "hybrid"
    DIVERSITY = "diversity"


@dataclass
class MemoryItem:
    content: str
    score: float = 0.0
    timestamp: float = 0.0


def rerank_by_relevance(items: Sequence[MemoryItem]) -> list[MemoryItem]:
    return sorted(items, key=lambda x: x.score, reverse=True)


def rerank_by_recency(items: Sequence[MemoryItem]) -> list[MemoryItem]:
    return sorted(items, key=lambda x: x.timestamp, reverse=True)


class MemoryRetrievalReranker:
    def __init__(self, strategy: RerankStrategy = RerankStrategy.RELEVANCE) -> None:
        self.strategy = strategy

    def rerank(self, items: Sequence[MemoryItem]) -> list[MemoryItem]:
        if not items:
            return []
        if self.strategy == RerankStrategy.RELEVANCE:
            return rerank_by_relevance(items)
        elif self.strategy == RerankStrategy.RECENCY:
            return rerank_by_recency(items)
        elif self.strategy == RerankStrategy.HYBRID:
            max_score = max(i.score for i in items) or 1
            max_time = max(i.timestamp for i in items) or 1
            scored = [
                MemoryItem(
                    content=i.content,
                    score=0.5 * (i.score / max_score) + 0.5 * (i.timestamp / max_time),
                    timestamp=i.timestamp,
                )
                for i in items
            ]
            return sorted(scored, key=lambda x: x.score, reverse=True)
        elif self.strategy == RerankStrategy.DIVERSITY:
            seen: set[str] = set()
            deduped: list[MemoryItem] = []
            for item in sorted(items, key=lambda x: x.score, reverse=True):
                if item.content not in seen:
                    seen.add(item.content)
                    deduped.append(item)
            return deduped
        return list(items)


MEMORY_RERANKER = MemoryRetrievalReranker()
