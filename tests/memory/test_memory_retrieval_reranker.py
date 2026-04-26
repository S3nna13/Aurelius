"""Tests for memory_retrieval_reranker — re-rank retrieved memories."""

from __future__ import annotations

from src.memory.memory_retrieval_reranker import (
    MemoryItem,
    MemoryRetrievalReranker,
    RerankStrategy,
    rerank_by_recency,
    rerank_by_relevance,
)


class TestMemoryItem:
    def test_item_creation(self):
        item = MemoryItem(content="hello world", score=0.8, timestamp=1000)
        assert item.content == "hello world"
        assert item.score == 0.8
        assert item.timestamp == 1000


class TestRerankByRelevance:
    def test_relevance_preserves_order(self):
        items = [
            MemoryItem("a", score=0.9),
            MemoryItem("b", score=0.5),
            MemoryItem("c", score=0.7),
        ]
        reranked = rerank_by_relevance(items)
        assert [i.content for i in reranked] == ["a", "c", "b"]

    def test_empty_list(self):
        assert rerank_by_relevance([]) == []


class TestRerankByRecency:
    def test_recency_orders_by_timestamp(self):
        items = [
            MemoryItem("old", score=0.5, timestamp=100),
            MemoryItem("new", score=0.5, timestamp=300),
            MemoryItem("mid", score=0.5, timestamp=200),
        ]
        reranked = rerank_by_recency(items)
        assert [i.content for i in reranked] == ["new", "mid", "old"]


class TestMemoryRetrievalReranker:
    def test_rerank_relevance_strategy(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.RELEVANCE)
        items = [MemoryItem("b", 0.3), MemoryItem("a", 0.9), MemoryItem("c", 0.6)]
        reranked = reranker.rerank(items)
        assert reranked[0].content == "a"

    def test_rerank_recency_strategy(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.RECENCY)
        items = [
            MemoryItem("old", score=0.9, timestamp=100),
            MemoryItem("new", score=0.1, timestamp=500),
        ]
        reranked = reranker.rerank(items)
        assert reranked[0].content == "new"

    def test_hybrid_blends_scores(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.HYBRID)
        items = [
            MemoryItem("rel", score=0.9, timestamp=100),
            MemoryItem("rec", score=0.1, timestamp=500),
        ]
        reranked = reranker.rerank(items)
        assert len(reranked) == 2

    def test_diversity_selects_unique_content(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.DIVERSITY)
        items = [
            MemoryItem("same idea", score=0.9),
            MemoryItem("same idea", score=0.8),
            MemoryItem("different", score=0.7),
        ]
        reranked = reranker.rerank(items)
        assert len(reranked) == 2
        assert reranked[0].content == "same idea"
        assert reranked[1].content == "different"

    def test_rerank_empty(self):
        reranker = MemoryRetrievalReranker()
        assert reranker.rerank([]) == []
