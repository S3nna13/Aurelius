"""Tests for src/cli/debug_commands.py — ~45 tests."""

import pytest

from src.cli.debug_commands import (
    DEBUG_COMMANDS,
    DebugCommands,
    DebugSnapshot,
    LogLevel,
)

# ---------------------------------------------------------------------------
# LogLevel enum
# ---------------------------------------------------------------------------


class TestLogLevel:
    def test_debug_value(self):
        assert LogLevel.DEBUG == "debug"

    def test_info_value(self):
        assert LogLevel.INFO == "info"

    def test_warning_value(self):
        assert LogLevel.WARNING == "warning"

    def test_error_value(self):
        assert LogLevel.ERROR == "error"

    def test_critical_value(self):
        assert LogLevel.CRITICAL == "critical"

    def test_is_str(self):
        assert isinstance(LogLevel.INFO, str)

    def test_five_members(self):
        assert len(LogLevel) == 5


# ---------------------------------------------------------------------------
# DebugSnapshot dataclass
# ---------------------------------------------------------------------------


class TestDebugSnapshot:
    def test_fields_present(self):
        s = DebugSnapshot(
            timestamp="2026-01-01T00:00:00",
            log_level="info",
            traces_enabled=False,
            metrics={},
        )
        assert s.timestamp == "2026-01-01T00:00:00"
        assert s.log_level == "info"
        assert s.traces_enabled is False
        assert s.metrics == {}


# ---------------------------------------------------------------------------
# DebugCommands — fresh instance per test
# ---------------------------------------------------------------------------


@pytest.fixture
def dc():
    return DebugCommands()


class TestDefaults:
    def test_default_log_level_is_info(self, dc):
        assert dc.get_log_level() == "info"

    def test_default_traces_disabled(self, dc):
        assert dc.traces_enabled() is False

    def test_default_metrics_empty_in_snapshot(self, dc):
        assert dc.snapshot().metrics == {}


class TestSetLogLevel:
    def test_valid_level_returns_true(self, dc):
        assert dc.set_log_level("debug") is True

    def test_invalid_level_returns_false(self, dc):
        assert dc.set_log_level("verbose") is False

    def test_case_insensitive_upper(self, dc):
        assert dc.set_log_level("DEBUG") is True

    def test_case_insensitive_mixed(self, dc):
        assert dc.set_log_level("Warning") is True

    def test_sets_the_level(self, dc):
        dc.set_log_level("error")
        assert dc.get_log_level() == "error"

    def test_invalid_does_not_change_level(self, dc):
        dc.set_log_level("bogus")
        assert dc.get_log_level() == "info"

    def test_all_valid_levels(self, dc):
        for level in ("debug", "info", "warning", "error", "critical"):
            assert dc.set_log_level(level) is True

    def test_empty_string_invalid(self, dc):
        assert dc.set_log_level("") is False


class TestGetLogLevel:
    def test_returns_string(self, dc):
        assert isinstance(dc.get_log_level(), str)

    def test_returns_correct_after_set(self, dc):
        dc.set_log_level("critical")
        assert dc.get_log_level() == "critical"


class TestTraces:
    def test_enable_traces(self, dc):
        dc.enable_traces()
        assert dc.traces_enabled() is True

    def test_disable_traces(self, dc):
        dc.enable_traces()
        dc.disable_traces()
        assert dc.traces_enabled() is False

    def test_disable_traces_already_off(self, dc):
        dc.disable_traces()
        assert dc.traces_enabled() is False

    def test_enable_then_disable_then_enable(self, dc):
        dc.enable_traces()
        dc.disable_traces()
        dc.enable_traces()
        assert dc.traces_enabled() is True


class TestRecordMetric:
    def test_metric_appears_in_snapshot(self, dc):
        dc.record_metric("latency", 1.5)
        assert dc.snapshot().metrics["latency"] == 1.5

    def test_multiple_metrics(self, dc):
        dc.record_metric("a", 1.0)
        dc.record_metric("b", 2.0)
        snap = dc.snapshot()
        assert snap.metrics["a"] == 1.0
        assert snap.metrics["b"] == 2.0

    def test_overwrite_metric(self, dc):
        dc.record_metric("x", 10.0)
        dc.record_metric("x", 20.0)
        assert dc.snapshot().metrics["x"] == 20.0


class TestSnapshot:
    def test_returns_debug_snapshot(self, dc):
        assert isinstance(dc.snapshot(), DebugSnapshot)

    def test_timestamp_is_string(self, dc):
        assert isinstance(dc.snapshot().timestamp, str)

    def test_timestamp_not_empty(self, dc):
        assert dc.snapshot().timestamp != ""

    def test_timestamp_iso_format(self, dc):
        ts = dc.snapshot().timestamp
        # ISO8601 from datetime.utcnow().isoformat() contains 'T'
        assert "T" in ts

    def test_snapshot_log_level_reflects_current(self, dc):
        dc.set_log_level("warning")
        assert dc.snapshot().log_level == "warning"

    def test_snapshot_traces_reflects_current(self, dc):
        dc.enable_traces()
        assert dc.snapshot().traces_enabled is True

    def test_snapshot_metrics_is_copy(self, dc):
        dc.record_metric("k", 1.0)
        snap = dc.snapshot()
        snap.metrics["k"] = 999.0
        # Internal state should be unaffected
        assert dc.snapshot().metrics["k"] == 1.0


class TestReset:
    def test_reset_restores_log_level_to_info(self, dc):
        dc.set_log_level("critical")
        dc.reset()
        assert dc.get_log_level() == "info"

    def test_reset_disables_traces(self, dc):
        dc.enable_traces()
        dc.reset()
        assert dc.traces_enabled() is False

    def test_reset_clears_metrics(self, dc):
        dc.record_metric("x", 5.0)
        dc.reset()
        assert dc.snapshot().metrics == {}

    def test_reset_multiple_times(self, dc):
        dc.set_log_level("error")
        dc.reset()
        dc.reset()
        assert dc.get_log_level() == "info"


# ---------------------------------------------------------------------------
# DEBUG_COMMANDS singleton
# ---------------------------------------------------------------------------


class TestDebugCommandsSingleton:
    def test_exists(self):
        assert DEBUG_COMMANDS is not None

    def test_is_debug_commands_instance(self):
        assert isinstance(DEBUG_COMMANDS, DebugCommands)
