"""Tests for result processor."""

from __future__ import annotations

from src.search.result_processor import ResultDeduplicator, ResultReranker, SearchResult


class TestResultDeduplicator:
    def test_identical_content_removed(self):
        rd = ResultDeduplicator(threshold=0.9)
        results = [
            SearchResult("1", 1.0, "hello world"),
            SearchResult("2", 0.9, "hello world"),
        ]
        deduped = rd.deduplicate(results)
        assert len(deduped) == 1

    def test_different_content_kept(self):
        rd = ResultDeduplicator()
        results = [
            SearchResult("1", 1.0, "hello world"),
            SearchResult("2", 0.9, "completely different text"),
        ]
        deduped = rd.deduplicate(results)
        assert len(deduped) == 2


class TestResultReranker:
    def test_rerank_boosts_relevant(self):
        rr = ResultReranker()
        results = [
            SearchResult("1", 1.0, "python programming language"),
            SearchResult("2", 0.9, "java programming language"),
        ]
        reranked = rr.rerank(results, "python")
        assert reranked[0].id == "1"
