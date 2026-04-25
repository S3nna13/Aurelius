"""Tests for log shipping."""
from __future__ import annotations

import json
from io import StringIO

import pytest

from src.monitoring.log_ship import (
    LogRecord,
    LogShipper,
    StdoutLogDestination,
)


class InMemoryDestination:
    """Test helper that stores records in memory."""
    def __init__(self):
        self.sent: list[list[dict]] = []

    def send(self, records: list[dict]) -> None:
        self.sent.append(records)


class TestLogShipper:
    def test_ship_buffers_and_flushes_at_batch_size(self):
        dest = InMemoryDestination()
        shipper = LogShipper(destination=dest, batch_size=3)

        shipper.ship(LogRecord("t1", "INFO", "msg1"))
        shipper.ship(LogRecord("t2", "INFO", "msg2"))
        assert len(dest.sent) == 0  # not flushed yet
        shipper.ship(LogRecord("t3", "WARN", "msg3"))
        assert len(dest.sent) == 1  # auto-flushed
        assert len(dest.sent[0]) == 3

    def test_flush_manually(self):
        dest = InMemoryDestination()
        shipper = LogShipper(destination=dest, batch_size=10)
        shipper.ship(LogRecord("t1", "ERROR", "fail"))
        shipper.flush()
        assert len(dest.sent) == 1

    def test_flush_empty_buffer(self):
        dest = InMemoryDestination()
        shipper = LogShipper(destination=dest)
        shipper.flush()
        assert len(dest.sent) == 0

    def test_stdout_destination(self, capsys):
        dest = StdoutLogDestination()
        dest.send([{"message": "test"}])
        captured = capsys.readouterr()
        assert json.loads(captured.out)["message"] == "test"