"""Tests for src/search/search_analytics.py  (>=28 tests)."""

from __future__ import annotations

import pytest

from src.search.search_analytics import (
    SEARCH_ANALYTICS_REGISTRY,
    QueryLog,
    SearchAnalytics,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_exists(self):
        assert SEARCH_ANALYTICS_REGISTRY is not None

    def test_registry_has_default_key(self):
        assert "default" in SEARCH_ANALYTICS_REGISTRY

    def test_registry_default_is_class(self):
        assert SEARCH_ANALYTICS_REGISTRY["default"] is SearchAnalytics


# ---------------------------------------------------------------------------
# QueryLog frozen dataclass
# ---------------------------------------------------------------------------


class TestQueryLogFrozen:
    def test_query_log_is_frozen(self):
        log = QueryLog(query="q", result_count=1, duration_ms=10.0, timestamp=0.0)
        with pytest.raises((TypeError, AttributeError)):
            log.query = "changed"  # type: ignore[misc]

    def test_query_log_fields(self):
        log = QueryLog(
            query="hello", result_count=5, duration_ms=42.0, timestamp=99.0, clicked_result="doc1"
        )
        assert log.query == "hello"
        assert log.result_count == 5
        assert log.duration_ms == 42.0
        assert log.timestamp == 99.0
        assert log.clicked_result == "doc1"

    def test_query_log_clicked_result_default_empty(self):
        log = QueryLog(query="q", result_count=0, duration_ms=1.0, timestamp=0.0)
        assert log.clicked_result == ""


# ---------------------------------------------------------------------------
# log_query
# ---------------------------------------------------------------------------


class TestLogQuery:
    def test_log_query_returns_query_log(self):
        sa = SearchAnalytics()
        log = sa.log_query("test", result_count=3, duration_ms=12.5)
        assert isinstance(log, QueryLog)

    def test_log_query_stores_entry(self):
        sa = SearchAnalytics()
        sa.log_query("alpha", result_count=1, duration_ms=5.0)
        assert sa.stats()["total_queries"] == 1

    def test_log_query_fields_correct(self):
        sa = SearchAnalytics()
        log = sa.log_query("search me", result_count=7, duration_ms=20.0, clicked_result="url1")
        assert log.query == "search me"
        assert log.result_count == 7
        assert log.duration_ms == 20.0
        assert log.clicked_result == "url1"

    def test_log_query_timestamp_positive(self):
        sa = SearchAnalytics()
        log = sa.log_query("q", result_count=0, duration_ms=1.0)
        assert log.timestamp >= 0.0

    def test_log_query_multiple_entries(self):
        sa = SearchAnalytics()
        for i in range(5):
            sa.log_query(f"query{i}", result_count=i, duration_ms=float(i))
        assert sa.stats()["total_queries"] == 5


# ---------------------------------------------------------------------------
# top_queries
# ---------------------------------------------------------------------------


class TestTopQueries:
    def test_top_queries_sorted_desc_by_count(self):
        sa = SearchAnalytics()
        for _ in range(3):
            sa.log_query("popular", result_count=1, duration_ms=1.0)
        for _ in range(1):
            sa.log_query("rare", result_count=1, duration_ms=1.0)
        top = sa.top_queries(n=2)
        assert top[0] == ("popular", 3)
        assert top[1] == ("rare", 1)

    def test_top_queries_returns_tuples(self):
        sa = SearchAnalytics()
        sa.log_query("q", result_count=1, duration_ms=1.0)
        top = sa.top_queries(n=5)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)

    def test_top_queries_n_limits_output(self):
        sa = SearchAnalytics()
        for i in range(10):
            sa.log_query(f"q{i}", result_count=1, duration_ms=1.0)
        assert len(sa.top_queries(n=3)) <= 3

    def test_top_queries_empty_analytics(self):
        sa = SearchAnalytics()
        assert sa.top_queries() == []


# ---------------------------------------------------------------------------
# zero_result_queries
# ---------------------------------------------------------------------------


class TestZeroResultQueries:
    def test_zero_result_queries_returns_unique_sorted(self):
        sa = SearchAnalytics()
        sa.log_query("beta", result_count=0, duration_ms=1.0)
        sa.log_query("alpha", result_count=0, duration_ms=1.0)
        sa.log_query("beta", result_count=0, duration_ms=1.0)  # duplicate
        zrq = sa.zero_result_queries()
        assert zrq == ["alpha", "beta"]

    def test_zero_result_excludes_queries_with_results(self):
        sa = SearchAnalytics()
        sa.log_query("found", result_count=5, duration_ms=1.0)
        sa.log_query("notfound", result_count=0, duration_ms=1.0)
        zrq = sa.zero_result_queries()
        assert "found" not in zrq
        assert "notfound" in zrq

    def test_zero_result_empty_when_all_have_results(self):
        sa = SearchAnalytics()
        sa.log_query("good", result_count=1, duration_ms=1.0)
        assert sa.zero_result_queries() == []


# ---------------------------------------------------------------------------
# mean_duration_ms
# ---------------------------------------------------------------------------


class TestMeanDuration:
    def test_mean_duration_empty_returns_zero(self):
        sa = SearchAnalytics()
        assert sa.mean_duration_ms() == 0.0

    def test_mean_duration_single_entry(self):
        sa = SearchAnalytics()
        sa.log_query("q", result_count=1, duration_ms=50.0)
        assert sa.mean_duration_ms() == 50.0

    def test_mean_duration_multiple_entries(self):
        sa = SearchAnalytics()
        sa.log_query("a", result_count=1, duration_ms=10.0)
        sa.log_query("b", result_count=1, duration_ms=20.0)
        sa.log_query("c", result_count=1, duration_ms=30.0)
        assert abs(sa.mean_duration_ms() - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# click_through_rate
# ---------------------------------------------------------------------------


class TestClickThroughRate:
    def test_ctr_empty_returns_zero(self):
        sa = SearchAnalytics()
        assert sa.click_through_rate() == 0.0

    def test_ctr_all_clicked(self):
        sa = SearchAnalytics()
        sa.log_query("a", result_count=1, duration_ms=1.0, clicked_result="doc1")
        sa.log_query("b", result_count=1, duration_ms=1.0, clicked_result="doc2")
        assert sa.click_through_rate() == 1.0

    def test_ctr_none_clicked(self):
        sa = SearchAnalytics()
        sa.log_query("a", result_count=1, duration_ms=1.0)
        sa.log_query("b", result_count=1, duration_ms=1.0)
        assert sa.click_through_rate() == 0.0

    def test_ctr_half_clicked(self):
        sa = SearchAnalytics()
        sa.log_query("a", result_count=1, duration_ms=1.0, clicked_result="doc1")
        sa.log_query("b", result_count=1, duration_ms=1.0)
        assert abs(sa.click_through_rate() - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_dict(self):
        sa = SearchAnalytics()
        assert isinstance(sa.stats(), dict)

    def test_stats_has_required_keys(self):
        sa = SearchAnalytics()
        keys = sa.stats().keys()
        for k in ("total_queries", "unique_queries", "mean_duration_ms", "zero_result_rate", "ctr"):
            assert k in keys

    def test_stats_total_queries(self):
        sa = SearchAnalytics()
        sa.log_query("q", result_count=1, duration_ms=1.0)
        sa.log_query("q", result_count=1, duration_ms=1.0)
        assert sa.stats()["total_queries"] == 2

    def test_stats_unique_queries(self):
        sa = SearchAnalytics()
        sa.log_query("alpha", result_count=1, duration_ms=1.0)
        sa.log_query("alpha", result_count=1, duration_ms=1.0)
        sa.log_query("beta", result_count=1, duration_ms=1.0)
        assert sa.stats()["unique_queries"] == 2

    def test_stats_zero_result_rate(self):
        sa = SearchAnalytics()
        sa.log_query("a", result_count=0, duration_ms=1.0)
        sa.log_query("b", result_count=5, duration_ms=1.0)
        assert abs(sa.stats()["zero_result_rate"] - 0.5) < 1e-9

    def test_stats_ctr(self):
        sa = SearchAnalytics()
        sa.log_query("a", result_count=1, duration_ms=1.0, clicked_result="d1")
        sa.log_query("b", result_count=1, duration_ms=1.0)
        assert abs(sa.stats()["ctr"] - 0.5) < 1e-9

    def test_stats_empty(self):
        sa = SearchAnalytics()
        s = sa.stats()
        assert s["total_queries"] == 0
        assert s["unique_queries"] == 0
        assert s["mean_duration_ms"] == 0.0
        assert s["zero_result_rate"] == 0.0
        assert s["ctr"] == 0.0


# ---------------------------------------------------------------------------
# max_log eviction
# ---------------------------------------------------------------------------


class TestMaxLogEviction:
    def test_max_log_evicts_oldest(self):
        sa = SearchAnalytics(max_log=3)
        sa.log_query("first", result_count=1, duration_ms=1.0)
        sa.log_query("second", result_count=1, duration_ms=1.0)
        sa.log_query("third", result_count=1, duration_ms=1.0)
        sa.log_query("fourth", result_count=1, duration_ms=1.0)
        # total should still be 3 (max_log)
        assert sa.stats()["total_queries"] == 3
        # "first" should have been evicted; "fourth" should be present
        queries = [log.query for log in sa._logs]
        assert "first" not in queries
        assert "fourth" in queries

    def test_max_log_size_never_exceeded(self):
        sa = SearchAnalytics(max_log=5)
        for i in range(20):
            sa.log_query(f"q{i}", result_count=i, duration_ms=1.0)
        assert sa.stats()["total_queries"] == 5


# ---------------------------------------------------------------------------
# query_volume_by_hour
# ---------------------------------------------------------------------------


class TestQueryVolumeByHour:
    def test_query_volume_returns_dict(self):
        sa = SearchAnalytics()
        sa.log_query("q", result_count=1, duration_ms=1.0)
        vol = sa.query_volume_by_hour()
        assert isinstance(vol, dict)

    def test_query_volume_counts_recent_queries(self):
        sa = SearchAnalytics()
        log = sa.log_query("q", result_count=1, duration_ms=1.0)
        # Pass now=timestamp so offset is exactly 0
        vol = sa.query_volume_by_hour(now=log.timestamp)
        assert vol.get(0, 0) >= 1

    def test_query_volume_empty_analytics(self):
        sa = SearchAnalytics()
        assert sa.query_volume_by_hour() == {}
