"""Tests for src/search/result_ranker.py (~45 tests)."""

from __future__ import annotations

import pytest

from src.search.result_ranker import (
    RESULT_RANKER,
    RankedResult,
    ResultRanker,
)

# ---------------------------------------------------------------------------
# RankedResult
# ---------------------------------------------------------------------------


class TestRankedResult:
    def test_result_id_field(self):
        r = RankedResult(result_id="abc", score=0.9, rank=1)
        assert r.result_id == "abc"

    def test_score_field(self):
        r = RankedResult(result_id="abc", score=0.9, rank=1)
        assert r.score == 0.9

    def test_rank_field(self):
        r = RankedResult(result_id="abc", score=0.9, rank=1)
        assert r.rank == 1

    def test_default_source(self):
        r = RankedResult(result_id="abc", score=0.9, rank=1)
        assert r.source == ""

    def test_custom_source(self):
        r = RankedResult(result_id="abc", score=0.9, rank=1, source="web")
        assert r.source == "web"


# ---------------------------------------------------------------------------
# ResultRanker
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def setup_method(self):
        self.ranker = ResultRanker(rrf_k=60)

    def test_empty_input(self):
        result = self.ranker.reciprocal_rank_fusion([])
        assert result == []

    def test_single_list(self):
        result = self.ranker.reciprocal_rank_fusion([["a", "b", "c"]])
        assert len(result) == 3

    def test_single_list_score_formula(self):
        result = self.ranker.reciprocal_rank_fusion([["a"]])
        doc_id, score = result[0]
        assert doc_id == "a"
        assert abs(score - 1.0 / (60 + 1)) < 1e-9

    def test_doc_in_all_lists_scores_higher(self):
        list1 = ["a", "b", "c"]
        list2 = ["b", "d", "e"]
        result = self.ranker.reciprocal_rank_fusion([list1, list2])
        scores = {doc_id: score for doc_id, score in result}
        # "b" appears in both lists
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["d"]

    def test_returns_sorted_descending(self):
        result = self.ranker.reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_docs_included(self):
        result = self.ranker.reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
        doc_ids = {d for d, _ in result}
        assert doc_ids == {"a", "b", "c", "d"}

    def test_custom_k(self):
        ranker = ResultRanker(rrf_k=10)
        result = ranker.reciprocal_rank_fusion([["x"]])
        _, score = result[0]
        assert abs(score - 1.0 / (10 + 1)) < 1e-9

    def test_multiple_lists_union(self):
        lists = [["a", "b"], ["b", "c"], ["c", "d"]]
        result = self.ranker.reciprocal_rank_fusion(lists)
        doc_ids = {d for d, _ in result}
        assert doc_ids == {"a", "b", "c", "d"}

    def test_single_doc_repeated_lists(self):
        result = self.ranker.reciprocal_rank_fusion([["x"], ["x"], ["x"]])
        assert len(result) == 1
        _, score = result[0]
        expected = 3 * (1.0 / (60 + 1))
        assert abs(score - expected) < 1e-9

    def test_empty_sublists(self):
        result = self.ranker.reciprocal_rank_fusion([[], []])
        assert result == []


class TestNormalizeScores:
    def setup_method(self):
        self.ranker = ResultRanker()

    def test_min_becomes_zero(self):
        results = [
            RankedResult(result_id="a", score=1.0, rank=1),
            RankedResult(result_id="b", score=2.0, rank=2),
            RankedResult(result_id="c", score=3.0, rank=3),
        ]
        normed = self.ranker.normalize_scores(results)
        scores = {r.result_id: r.score for r in normed}
        assert scores["a"] == pytest.approx(0.0)

    def test_max_becomes_one(self):
        results = [
            RankedResult(result_id="a", score=1.0, rank=1),
            RankedResult(result_id="b", score=2.0, rank=2),
            RankedResult(result_id="c", score=3.0, rank=3),
        ]
        normed = self.ranker.normalize_scores(results)
        scores = {r.result_id: r.score for r in normed}
        assert scores["c"] == pytest.approx(1.0)

    def test_all_equal_set_to_one(self):
        results = [
            RankedResult(result_id="a", score=5.0, rank=1),
            RankedResult(result_id="b", score=5.0, rank=2),
        ]
        normed = self.ranker.normalize_scores(results)
        assert all(r.score == 1.0 for r in normed)

    def test_empty_list(self):
        assert self.ranker.normalize_scores([]) == []

    def test_single_result_all_equal(self):
        results = [RankedResult(result_id="a", score=0.5, rank=1)]
        normed = self.ranker.normalize_scores(results)
        assert normed[0].score == 1.0

    def test_preserves_result_ids(self):
        results = [
            RankedResult(result_id="x", score=1.0, rank=1),
            RankedResult(result_id="y", score=2.0, rank=2),
        ]
        normed = self.ranker.normalize_scores(results)
        ids = [r.result_id for r in normed]
        assert ids == ["x", "y"]

    def test_preserves_rank(self):
        results = [
            RankedResult(result_id="a", score=1.0, rank=5),
        ]
        normed = self.ranker.normalize_scores(results)
        assert normed[0].rank == 5

    def test_scores_in_range(self):
        results = [RankedResult(result_id=f"d{i}", score=float(i), rank=i + 1) for i in range(10)]
        normed = self.ranker.normalize_scores(results)
        for r in normed:
            assert 0.0 <= r.score <= 1.0


class TestDeduplicate:
    def setup_method(self):
        self.ranker = ResultRanker()

    def test_no_duplicates_unchanged(self):
        results = [
            RankedResult(result_id="a", score=1.0, rank=1),
            RankedResult(result_id="b", score=0.9, rank=2),
        ]
        deduped = self.ranker.deduplicate(results)
        assert len(deduped) == 2

    def test_duplicates_removed(self):
        results = [
            RankedResult(result_id="a", score=1.0, rank=1),
            RankedResult(result_id="a", score=0.5, rank=2),
        ]
        deduped = self.ranker.deduplicate(results)
        assert len(deduped) == 1

    def test_keeps_first_occurrence(self):
        results = [
            RankedResult(result_id="a", score=1.0, rank=1),
            RankedResult(result_id="a", score=0.5, rank=2),
        ]
        deduped = self.ranker.deduplicate(results)
        assert deduped[0].score == 1.0

    def test_preserves_order(self):
        results = [
            RankedResult(result_id="b", score=0.5, rank=2),
            RankedResult(result_id="a", score=1.0, rank=1),
            RankedResult(result_id="b", score=0.3, rank=3),
        ]
        deduped = self.ranker.deduplicate(results)
        assert [r.result_id for r in deduped] == ["b", "a"]

    def test_empty_list(self):
        assert self.ranker.deduplicate([]) == []

    def test_all_unique(self):
        results = [RankedResult(result_id=f"d{i}", score=float(i), rank=i + 1) for i in range(5)]
        deduped = self.ranker.deduplicate(results)
        assert len(deduped) == 5

    def test_all_same_id(self):
        results = [RankedResult(result_id="x", score=float(i), rank=i + 1) for i in range(5)]
        deduped = self.ranker.deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].score == 0.0  # first occurrence


class TestToRanked:
    def setup_method(self):
        self.ranker = ResultRanker()

    def test_basic_to_ranked(self):
        items = ["a", "b", "c"]
        scores = [0.9, 0.8, 0.7]
        results = self.ranker.to_ranked(items, scores)
        assert len(results) == 3

    def test_rank_starts_at_one(self):
        results = self.ranker.to_ranked(["a"], [0.5])
        assert results[0].rank == 1

    def test_rank_increments(self):
        results = self.ranker.to_ranked(["a", "b", "c"], [0.9, 0.8, 0.7])
        assert [r.rank for r in results] == [1, 2, 3]

    def test_scores_assigned(self):
        results = self.ranker.to_ranked(["a", "b"], [0.9, 0.4])
        assert results[0].score == 0.9
        assert results[1].score == 0.4

    def test_result_ids_assigned(self):
        results = self.ranker.to_ranked(["x", "y"], [1.0, 0.5])
        assert results[0].result_id == "x"
        assert results[1].result_id == "y"

    def test_source_assigned(self):
        results = self.ranker.to_ranked(["a"], [0.5], source="web")
        assert results[0].source == "web"

    def test_default_source_empty(self):
        results = self.ranker.to_ranked(["a"], [0.5])
        assert results[0].source == ""

    def test_empty_input(self):
        results = self.ranker.to_ranked([], [])
        assert results == []


class TestGlobalInstance:
    def test_result_ranker_exists(self):
        assert RESULT_RANKER is not None

    def test_result_ranker_is_instance(self):
        assert isinstance(RESULT_RANKER, ResultRanker)

    def test_result_ranker_default_k(self):
        assert RESULT_RANKER.rrf_k == 60
