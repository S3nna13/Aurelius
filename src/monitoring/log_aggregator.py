"""Log aggregator: ring-buffered structured log store with query support."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    timestamp: float
    level: LogLevel
    logger: str
    message: str
    context: dict = field(default_factory=dict)


class LogAggregator:
    """Thread-naïve ring-buffered log store."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._buffer: deque[LogEntry] = deque(maxlen=max_entries)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def log(
        self,
        level: LogLevel,
        logger: str,
        message: str,
        context: dict | None = None,
    ) -> None:
        """Append a log entry to the ring buffer."""
        entry = LogEntry(
            timestamp=time.monotonic(),
            level=level,
            logger=logger,
            message=message,
            context=context or {},
        )
        self._buffer.append(entry)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query(
        self,
        level: LogLevel | None = None,
        logger: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Return up to *limit* entries matching the given filters (newest last)."""
        results: list[LogEntry] = []
        for entry in self._buffer:
            if level is not None and entry.level != level:
                continue
            if logger is not None and entry.logger != logger:
                continue
            if since is not None and entry.timestamp < since:
                continue
            results.append(entry)
        return results[-limit:]

    def count_by_level(self) -> dict[LogLevel, int]:
        """Return a mapping of log level -> entry count."""
        counts: dict[LogLevel, int] = {lvl: 0 for lvl in LogLevel}
        for entry in self._buffer:
            counts[entry.level] += 1
        return counts

    def get_error_rate(self, window_seconds: float = 60.0) -> float:
        """Return the rate of ERROR/CRITICAL entries per second in the last *window_seconds*."""
        cutoff = time.monotonic() - window_seconds
        error_count = sum(
            1
            for e in self._buffer
            if e.timestamp >= cutoff and e.level in (LogLevel.ERROR, LogLevel.CRITICAL)
        )
        return error_count / window_seconds if window_seconds > 0 else 0.0


# Module-level singleton
_LOG_AGGREGATOR = LogAggregator()

try:
    from src.monitoring import MONITORING_REGISTRY as _REG  # type: ignore[import]

    _REG["log_aggregator"] = _LOG_AGGREGATOR
except Exception:  # noqa: S110
    pass

MONITORING_REGISTRY: dict[str, object] = {"log_aggregator": _LOG_AGGREGATOR}
