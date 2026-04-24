"""Tests for src/monitoring/log_aggregator.py"""
import time

import pytest

from src.monitoring.log_aggregator import LogAggregator, LogEntry, LogLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh() -> LogAggregator:
    return LogAggregator(max_entries=100)


# ---------------------------------------------------------------------------
# LogLevel enum
# ---------------------------------------------------------------------------

def test_log_level_values():
    assert LogLevel.DEBUG == "DEBUG"
    assert LogLevel.INFO == "INFO"
    assert LogLevel.WARNING == "WARNING"
    assert LogLevel.ERROR == "ERROR"
    assert LogLevel.CRITICAL == "CRITICAL"


# ---------------------------------------------------------------------------
# Basic logging
# ---------------------------------------------------------------------------

def test_log_appends_entry():
    la = fresh()
    la.log(LogLevel.INFO, "app", "hello")
    entries = la.query()
    assert len(entries) == 1
    assert entries[0].message == "hello"
    assert entries[0].level == LogLevel.INFO
    assert entries[0].logger == "app"


def test_log_context_stored():
    la = fresh()
    la.log(LogLevel.DEBUG, "svc", "msg", context={"key": "value"})
    entry = la.query()[0]
    assert entry.context["key"] == "value"


def test_log_context_defaults_empty():
    la = fresh()
    la.log(LogLevel.INFO, "svc", "msg")
    entry = la.query()[0]
    assert entry.context == {}


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

def test_ring_buffer_overflow():
    la = LogAggregator(max_entries=5)
    for i in range(10):
        la.log(LogLevel.INFO, "x", f"msg{i}")
    entries = la.query(limit=10)
    assert len(entries) == 5
    # Oldest 5 messages should have been evicted
    messages = [e.message for e in entries]
    assert "msg9" in messages


# ---------------------------------------------------------------------------
# query filters
# ---------------------------------------------------------------------------

def test_query_by_level():
    la = fresh()
    la.log(LogLevel.DEBUG, "a", "d1")
    la.log(LogLevel.ERROR, "a", "e1")
    la.log(LogLevel.ERROR, "a", "e2")
    errs = la.query(level=LogLevel.ERROR)
    assert len(errs) == 2
    dbgs = la.query(level=LogLevel.DEBUG)
    assert len(dbgs) == 1


def test_query_by_logger():
    la = fresh()
    la.log(LogLevel.INFO, "svc-a", "a")
    la.log(LogLevel.INFO, "svc-b", "b")
    result = la.query(logger="svc-a")
    assert all(e.logger == "svc-a" for e in result)
    assert len(result) == 1


def test_query_limit():
    la = fresh()
    for i in range(20):
        la.log(LogLevel.INFO, "x", f"m{i}")
    result = la.query(limit=5)
    assert len(result) == 5


def test_query_since_filter():
    la = fresh()
    la.log(LogLevel.INFO, "a", "old")
    cutoff = time.monotonic()
    la.log(LogLevel.INFO, "a", "new")
    result = la.query(since=cutoff)
    assert len(result) == 1
    assert result[0].message == "new"


# ---------------------------------------------------------------------------
# count_by_level
# ---------------------------------------------------------------------------

def test_count_by_level_all_levels():
    la = fresh()
    la.log(LogLevel.DEBUG, "a", "d")
    la.log(LogLevel.INFO, "a", "i")
    la.log(LogLevel.INFO, "a", "i2")
    la.log(LogLevel.ERROR, "a", "e")
    counts = la.count_by_level()
    assert counts[LogLevel.DEBUG] == 1
    assert counts[LogLevel.INFO] == 2
    assert counts[LogLevel.ERROR] == 1
    assert counts[LogLevel.WARNING] == 0
    assert counts[LogLevel.CRITICAL] == 0


def test_count_by_level_empty():
    la = fresh()
    counts = la.count_by_level()
    assert all(v == 0 for v in counts.values())


# ---------------------------------------------------------------------------
# get_error_rate
# ---------------------------------------------------------------------------

def test_get_error_rate_non_negative():
    la = fresh()
    la.log(LogLevel.ERROR, "a", "boom")
    rate = la.get_error_rate(window_seconds=60.0)
    assert rate >= 0.0


def test_get_error_rate_zero_when_no_errors():
    la = fresh()
    la.log(LogLevel.INFO, "a", "fine")
    rate = la.get_error_rate(window_seconds=60.0)
    assert rate == 0.0


def test_get_error_rate_counts_critical():
    la = fresh()
    la.log(LogLevel.CRITICAL, "a", "fatal")
    rate = la.get_error_rate(window_seconds=60.0)
    assert rate > 0.0
