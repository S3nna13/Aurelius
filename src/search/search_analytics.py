"""Aurelius search – query analytics and performance tracking."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryLog:
    """Immutable record of a single search event."""

    query: str
    result_count: int
    duration_ms: float
    timestamp: float
    clicked_result: str = ""


class SearchAnalytics:
    """Tracks search query logs and exposes aggregate analytics."""

    def __init__(self, max_log: int = 10000) -> None:
        self._max_log = max_log
        self._logs: list[QueryLog] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def log_query(
        self,
        query: str,
        result_count: int,
        duration_ms: float,
        clicked_result: str = "",
    ) -> QueryLog:
        """Create a :class:`QueryLog` entry, store it, and return it.

        If the log is at capacity the oldest entry is evicted (FIFO).
        Timestamp is set to ``time.monotonic()``.
        """
        entry = QueryLog(
            query=query,
            result_count=result_count,
            duration_ms=duration_ms,
            timestamp=time.monotonic(),
            clicked_result=clicked_result,
        )
        if len(self._logs) >= self._max_log:
            self._logs.pop(0)
        self._logs.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def top_queries(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the *n* most frequent queries as (query, count) sorted desc."""
        counter: Counter[str] = Counter(log.query for log in self._logs)
        return counter.most_common(n)

    def zero_result_queries(self) -> list[str]:
        """Unique queries that returned zero results, sorted lexicographically."""
        return sorted({log.query for log in self._logs if log.result_count == 0})

    def mean_duration_ms(self) -> float:
        """Average query duration in milliseconds; 0.0 if no logs."""
        if not self._logs:
            return 0.0
        return sum(log.duration_ms for log in self._logs) / len(self._logs)

    def click_through_rate(self) -> float:
        """Fraction of logged queries where a result was clicked."""
        if not self._logs:
            return 0.0
        clicked = sum(1 for log in self._logs if log.clicked_result)
        return clicked / len(self._logs)

    def query_volume_by_hour(self, now: float | None = None) -> dict[int, int]:
        """Return {hour_offset: count} where hour_offset = floor((now - ts) / 3600).

        Hour offset 0 is the most-recent hour; larger values are older.
        Uses ``time.monotonic()`` when *now* is not provided.
        """
        if now is None:
            now = time.monotonic()
        volume: dict[int, int] = {}
        for log in self._logs:
            offset = math.floor((now - log.timestamp) / 3600)
            volume[offset] = volume.get(offset, 0) + 1
        return volume

    def stats(self) -> dict[str, object]:
        """Return a summary statistics dictionary."""
        total = len(self._logs)
        unique = len({log.query for log in self._logs})
        zero_count = sum(1 for log in self._logs if log.result_count == 0)
        zero_rate = zero_count / total if total else 0.0
        return {
            "total_queries": total,
            "unique_queries": unique,
            "mean_duration_ms": self.mean_duration_ms(),
            "zero_result_rate": zero_rate,
            "ctr": self.click_through_rate(),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SEARCH_ANALYTICS_REGISTRY: dict[str, type] = {"default": SearchAnalytics}
