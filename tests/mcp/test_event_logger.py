"""Tests for src/mcp/event_logger.py — ≥28 test cases."""

from __future__ import annotations

import dataclasses

import pytest

from src.mcp.event_logger import (
    EventLevel,
    LogEvent,
    MCPEventLogger,
    MCP_EVENT_LOGGER_REGISTRY,
)


# ---------------------------------------------------------------------------
# LogEvent frozen dataclass
# ---------------------------------------------------------------------------

class TestLogEventFrozen:
    def test_is_frozen(self):
        ev = LogEvent(
            event_id="abc",
            level=EventLevel.INFO,
            message="m",
            context={},
            timestamp=0.0,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ev.message = "changed"  # type: ignore[misc]

    def test_fields_accessible(self):
        ev = LogEvent(
            event_id="id1",
            level=EventLevel.ERROR,
            message="fail",
            context={"k": "v"},
            timestamp=1.0,
        )
        assert ev.event_id == "id1"
        assert ev.level is EventLevel.ERROR
        assert ev.message == "fail"
        assert ev.context == {"k": "v"}
        assert ev.timestamp == 1.0


# ---------------------------------------------------------------------------
# EventLevel enum
# ---------------------------------------------------------------------------

class TestEventLevel:
    def test_ordering_values(self):
        assert EventLevel.DEBUG.value < EventLevel.INFO.value
        assert EventLevel.INFO.value < EventLevel.WARNING.value
        assert EventLevel.WARNING.value < EventLevel.ERROR.value
        assert EventLevel.ERROR.value < EventLevel.CRITICAL.value

    def test_all_five_members(self):
        names = {lvl.name for lvl in EventLevel}
        assert names == {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


# ---------------------------------------------------------------------------
# log() basics
# ---------------------------------------------------------------------------

class TestLog:
    def test_log_info_returns_log_event(self):
        logger = MCPEventLogger()
        ev = logger.log(EventLevel.INFO, "hello")
        assert isinstance(ev, LogEvent)

    def test_log_stores_message(self):
        logger = MCPEventLogger()
        ev = logger.log(EventLevel.INFO, "stored")
        assert ev.message == "stored"

    def test_log_stores_level(self):
        logger = MCPEventLogger()
        ev = logger.log(EventLevel.ERROR, "err")
        assert ev.level is EventLevel.ERROR

    def test_log_auto_event_id_length(self):
        logger = MCPEventLogger()
        ev = logger.log(EventLevel.INFO, "x")
        assert len(ev.event_id) == 10

    def test_log_auto_event_id_is_hex(self):
        logger = MCPEventLogger()
        ev = logger.log(EventLevel.INFO, "x")
        int(ev.event_id, 16)  # should not raise

    def test_log_stores_context(self):
        logger = MCPEventLogger()
        ev = logger.log(EventLevel.INFO, "ctx", {"key": "val"})
        assert ev.context == {"key": "val"}

    def test_log_no_context_defaults_empty(self):
        logger = MCPEventLogger()
        ev = logger.log(EventLevel.INFO, "no-ctx")
        assert ev.context == {}

    def test_log_below_min_level_returns_none(self):
        logger = MCPEventLogger(min_level=EventLevel.INFO)
        result = logger.log(EventLevel.DEBUG, "debug msg")
        assert result is None

    def test_log_at_min_level_stored(self):
        logger = MCPEventLogger(min_level=EventLevel.WARNING)
        ev = logger.log(EventLevel.WARNING, "warn")
        assert ev is not None

    def test_log_above_min_level_stored(self):
        logger = MCPEventLogger(min_level=EventLevel.WARNING)
        ev = logger.log(EventLevel.ERROR, "err")
        assert ev is not None


# ---------------------------------------------------------------------------
# max_events eviction
# ---------------------------------------------------------------------------

class TestMaxEventsEviction:
    def test_oldest_evicted_at_capacity(self):
        logger = MCPEventLogger(max_events=3)
        ev1 = logger.log(EventLevel.INFO, "first")
        logger.log(EventLevel.INFO, "second")
        logger.log(EventLevel.INFO, "third")
        logger.log(EventLevel.INFO, "fourth")  # should evict ev1
        all_msgs = [e.message for e in logger.query(limit=10)]
        assert "first" not in all_msgs
        assert "fourth" in all_msgs

    def test_max_events_not_exceeded(self):
        logger = MCPEventLogger(max_events=5)
        for i in range(10):
            logger.log(EventLevel.INFO, f"msg-{i}")
        assert logger.stats()["total"] == 5


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def test_debug_method(self):
        logger = MCPEventLogger(min_level=EventLevel.DEBUG)
        ev = logger.debug("dbg", x=1)
        assert ev is not None
        assert ev.level is EventLevel.DEBUG
        assert ev.context == {"x": 1}

    def test_info_method(self):
        logger = MCPEventLogger()
        ev = logger.info("inf", y=2)
        assert ev is not None
        assert ev.level is EventLevel.INFO

    def test_warning_method(self):
        logger = MCPEventLogger()
        ev = logger.warning("wrn")
        assert ev is not None
        assert ev.level is EventLevel.WARNING

    def test_error_method(self):
        logger = MCPEventLogger()
        ev = logger.error("err")
        assert ev is not None
        assert ev.level is EventLevel.ERROR

    def test_critical_method(self):
        logger = MCPEventLogger()
        ev = logger.critical("crit", code=500)
        assert ev is not None
        assert ev.level is EventLevel.CRITICAL
        assert ev.context == {"code": 500}

    def test_debug_below_default_min_returns_none(self):
        logger = MCPEventLogger(min_level=EventLevel.INFO)
        assert logger.debug("ignored") is None


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    def _populated(self) -> MCPEventLogger:
        logger = MCPEventLogger(min_level=EventLevel.DEBUG)
        logger.debug("alpha debug")
        logger.info("beta info")
        logger.warning("gamma warning")
        logger.error("delta error")
        logger.critical("epsilon critical")
        return logger

    def test_query_all(self):
        logger = self._populated()
        results = logger.query(limit=100)
        assert len(results) == 5

    def test_query_by_level(self):
        logger = self._populated()
        results = logger.query(level=EventLevel.ERROR)
        assert all(e.level is EventLevel.ERROR for e in results)
        assert len(results) == 1

    def test_query_by_contains(self):
        logger = self._populated()
        results = logger.query(contains="beta")
        assert len(results) == 1
        assert results[0].message == "beta info"

    def test_query_contains_case_insensitive(self):
        logger = self._populated()
        results = logger.query(contains="GAMMA")
        assert len(results) == 1

    def test_query_newest_first(self):
        logger = MCPEventLogger()
        logger.info("oldest")
        logger.info("middle")
        logger.info("newest")
        results = logger.query(limit=10)
        assert results[0].message == "newest"
        assert results[-1].message == "oldest"

    def test_query_limit(self):
        logger = MCPEventLogger()
        for i in range(10):
            logger.info(f"msg-{i}")
        results = logger.query(limit=3)
        assert len(results) == 3

    def test_query_combined_level_and_contains(self):
        logger = MCPEventLogger(min_level=EventLevel.DEBUG)
        logger.debug("debug relevant")
        logger.info("info relevant")
        logger.info("info unrelated")
        results = logger.query(level=EventLevel.INFO, contains="relevant")
        assert len(results) == 1
        assert results[0].level is EventLevel.INFO
        assert results[0].message == "info relevant"

    def test_query_no_match_returns_empty(self):
        logger = MCPEventLogger()
        logger.info("something")
        results = logger.query(contains="zzz_impossible_zzz")
        assert results == []


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_total(self):
        logger = MCPEventLogger()
        logger.info("a")
        logger.info("b")
        logger.error("c")
        s = logger.stats()
        assert s["total"] == 3

    def test_stats_by_level(self):
        logger = MCPEventLogger(min_level=EventLevel.DEBUG)
        logger.debug("d")
        logger.info("i")
        logger.info("i2")
        logger.warning("w")
        logger.error("e")
        logger.critical("c")
        s = logger.stats()
        assert s["by_level"]["DEBUG"] == 1
        assert s["by_level"]["INFO"] == 2
        assert s["by_level"]["WARNING"] == 1
        assert s["by_level"]["ERROR"] == 1
        assert s["by_level"]["CRITICAL"] == 1

    def test_stats_all_levels_present(self):
        logger = MCPEventLogger()
        s = logger.stats()
        assert set(s["by_level"].keys()) == {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_events(self):
        logger = MCPEventLogger()
        logger.info("a")
        logger.info("b")
        logger.clear()
        assert logger.stats()["total"] == 0

    def test_clear_then_log_works(self):
        logger = MCPEventLogger()
        logger.info("a")
        logger.clear()
        ev = logger.info("b")
        assert ev is not None
        assert logger.stats()["total"] == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in MCP_EVENT_LOGGER_REGISTRY

    def test_default_is_mcp_event_logger(self):
        assert MCP_EVENT_LOGGER_REGISTRY["default"] is MCPEventLogger

    def test_default_instantiable(self):
        cls = MCP_EVENT_LOGGER_REGISTRY["default"]
        logger = cls()
        assert isinstance(logger, MCPEventLogger)
