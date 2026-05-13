"""Tests: plugins/memory/memory_retrieval_reranker.py — Reranking strategies for memory retrieval."""

from __future__ import annotations

import pytest

from plugins.memory.memory_retrieval_reranker import (
    MEMORY_RERANKER,
    MemoryItem,
    MemoryRetrievalReranker,
    RerankStrategy,
    rerank_by_recency,
    rerank_by_relevance,
)


@pytest.fixture
def reranker():
    return MemoryRetrievalReranker(strategy=RerankStrategy.RELEVANCE)


class TestRerankStrategy:
    def test_all_values(self):
        assert RerankStrategy.RELEVANCE.value == "relevance"
        assert RerankStrategy.RECENCY.value == "recency"
        assert RerankStrategy.HYBRID.value == "hybrid"
        assert RerankStrategy.DIVERSITY.value == "diversity"


class TestMemoryItem:
    def test_item_defaults(self):
        item = MemoryItem(content="hello", score=0.5, timestamp=123.0)
        assert item.content == "hello"
        assert item.score == 0.5
        assert item.timestamp == 123.0


class TestRerankByRelevance:
    def test_orders_by_score_descending(self):
        items = [
            MemoryItem("low", score=0.1),
            MemoryItem("high", score=0.9),
            MemoryItem("mid", score=0.5),
        ]
        result = rerank_by_relevance(items)
        assert [i.content for i in result] == ["high", "mid", "low"]


class TestRerankByRecency:
    def test_orders_by_timestamp_descending(self):
        items = [
            MemoryItem("old", timestamp=100.0),
            MemoryItem("new", timestamp=300.0),
            MemoryItem("mid", timestamp=200.0),
        ]
        result = rerank_by_recency(items)
        assert [i.content for i in result] == ["new", "mid", "old"]


class TestMemoryRetrievalReranker:
    def test_default_strategy(self):
        reranker = MemoryRetrievalReranker()
        assert reranker.strategy == RerankStrategy.RELEVANCE

    def test_rerank_relevance(self, reranker):
        items = [
            MemoryItem("low", score=0.2),
            MemoryItem("high", score=0.8),
        ]
        result = reranker.rerank(items)
        assert [i.content for i in result] == ["high", "low"]

    def test_rerank_recency(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.RECENCY)
        items = [
            MemoryItem("old", timestamp=10.0),
            MemoryItem("new", timestamp=99.0),
        ]
        result = reranker.rerank(items)
        assert [i.content for i in result] == ["new", "old"]

    def test_rerank_hybrid_balances_score_and_time(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.HYBRID)
        items = [
            MemoryItem("old_high", score=1.0, timestamp=0.0),  # max score, min time
            MemoryItem("new_low", score=0.0, timestamp=1.0),   # min score, max time
        ]
        result = reranker.rerank(items)
        # Both get equal hybrid score of 0.5 — stable sort may pick either, just verify they're both there
        assert len(result) == 2
        contents = {i.content for i in result}
        assert contents == {"old_high", "new_low"}

    def test_rerank_hybrid_max_time_zero_guard(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.HYBRID)
        items = [MemoryItem("only", score=0.0, timestamp=0.0)]
        result = reranker.rerank(items)
        assert len(result) == 1

    def test_rerank_diversity_removes_duplicate_content(self):
        reranker = MemoryRetrievalReranker(strategy=RerankStrategy.DIVERSITY)
        items = [
            MemoryItem("a", score=0.8),
            MemoryItem("b", score=0.7),
            MemoryItem("a", score=0.6),  # duplicate content
            MemoryItem("c", score=0.5),
        ]
        result = reranker.rerank(items)
        contents = [i.content for i in result]
        assert contents.count("a") == 1
        assert len(result) == 3

    def test_rerank_unknown_strategy_returns_input(self):
        reranker = MemoryRetrievalReranker()
        items = [MemoryItem("test")]
        # Access private attribute to inject unknown strategy
        reranker.strategy = "unknown"  # type: ignore[assignment]
        result = reranker.rerank(items)
        assert len(result) == 1

    def test_rerank_empty(self, reranker):
        assert reranker.rerank([]) == []


class TestSingleton:
    def test_module_singleton(self):
        assert isinstance(MEMORY_RERANKER, MemoryRetrievalReranker)
