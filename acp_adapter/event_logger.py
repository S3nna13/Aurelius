"""Aurelius MCP — event_logger.py

Structured event logger for MCP operations.  Events are stored in memory with
level filtering, substring querying, and capacity-based eviction of oldest entries.
All logic is pure Python stdlib; no external deps.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class EventLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass(frozen=True)
class LogEvent:
    event_id: str
    level: EventLevel
    message: str
    context: dict
    timestamp: float


# ---------------------------------------------------------------------------
# MCPEventLogger
# ---------------------------------------------------------------------------


class MCPEventLogger:
    """In-memory structured event logger for MCP operations."""

    def __init__(
        self,
        min_level: EventLevel = EventLevel.INFO,
        max_events: int = 10000,
    ) -> None:
        self._min_level = min_level
        self._max_events = max_events
        # deque keeps insertion order; oldest at left, newest at right
        self._events: deque[LogEvent] = deque()

    # ------------------------------------------------------------------
    # Core log method
    # ------------------------------------------------------------------

    def log(
        self,
        level: EventLevel,
        message: str,
        context: dict | None = None,
    ) -> LogEvent | None:
        """Record a log event.

        Returns None if *level* is below min_level.
        Evicts the oldest event when at max_events capacity.
        """
        if level.value < self._min_level.value:
            return None

        if len(self._events) >= self._max_events:
            self._events.popleft()

        event = LogEvent(
            event_id=uuid.uuid4().hex[:10],
            level=level,
            message=message,
            context=dict(context) if context else {},
            timestamp=time.monotonic(),
        )
        self._events.append(event)
        return event

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def debug(self, message: str, **context: object) -> LogEvent | None:
        return self.log(EventLevel.DEBUG, message, dict(context))

    def info(self, message: str, **context: object) -> LogEvent | None:
        return self.log(EventLevel.INFO, message, dict(context))

    def warning(self, message: str, **context: object) -> LogEvent | None:
        return self.log(EventLevel.WARNING, message, dict(context))

    def error(self, message: str, **context: object) -> LogEvent | None:
        return self.log(EventLevel.ERROR, message, dict(context))

    def critical(self, message: str, **context: object) -> LogEvent | None:
        return self.log(EventLevel.CRITICAL, message, dict(context))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        level: EventLevel | None = None,
        contains: str | None = None,
        limit: int = 100,
    ) -> list[LogEvent]:
        """Return events filtered by *level* and/or message substring *contains*.

        Results are newest-first and capped at *limit*.
        """
        results: list[LogEvent] = []
        needle = contains.lower() if contains is not None else None

        # Iterate newest-first
        for event in reversed(self._events):
            if level is not None and event.level is not level:
                continue
            if needle is not None and needle not in event.message.lower():
                continue
            results.append(event)
            if len(results) >= limit:
                break

        return results

    # ------------------------------------------------------------------
    # Stats & management
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return total count and per-level breakdown."""
        by_level: dict[str, int] = {lvl.name: 0 for lvl in EventLevel}
        for event in self._events:
            by_level[event.level.name] += 1
        return {"total": len(self._events), "by_level": by_level}

    def clear(self) -> None:
        """Remove all stored events."""
        self._events.clear()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MCP_EVENT_LOGGER_REGISTRY: dict[str, type[MCPEventLogger]] = {"default": MCPEventLogger}
