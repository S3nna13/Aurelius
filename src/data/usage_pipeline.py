"""Usage data pipeline backed by an embedded Rust store.

Memory-efficient ingestion and querying of user activity events.
The storage layer is a ``redb`` database accessed through PyO3 bindings
(``aurelius_usage_store``), keeping the hot path in Rust and Python's
heap usage minimal.

Example::

    from src.data.usage_pipeline import UsagePipeline

    pipe = UsagePipeline("/var/lib/aurelius/usage.redb")
    pipe.log("user_123", "chat_completion", {"model": "aurelius-v1", "tokens": 150})
    for ev in pipe.query("user_123", limit=10):
        print(ev["event_type"], ev["payload"])
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

try:
    from aurelius_usage_store import UsageStore
except ImportError as _exc:  # pragma: no cover
    raise RuntimeError(
        "aurelius_usage_store Rust extension is not built. "
        "Run: cd crates/usage-store && maturin develop"
    ) from _exc


DEFAULT_DB_PATH: str = os.environ.get(
    "AURELIUS_USAGE_DB", "/var/lib/aurelius/usage.redb"
)


class UsagePipeline:
    """High-level wrapper around the Rust ``UsageStore``.

    Provides automatic connection lifecycle management and convenience
    methods for common query patterns.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._path = Path(db_path or DEFAULT_DB_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store = UsageStore(str(self._path))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def log(
        self,
        user_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> str:
        """Persist a usage event. Returns the event ULID."""
        return self._store.log_event(user_id, event_type, payload)

    def log_chat(
        self,
        user_id: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float | None = None,
    ) -> str:
        """Convenience helper for chat-completion events."""
        payload: dict[str, Any] = {
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        return self.log(user_id, "chat_completion", payload)

    def log_tool_call(
        self,
        user_id: str,
        tool_name: str,
        success: bool,
        duration_ms: float | None = None,
    ) -> str:
        """Convenience helper for tool-execution events."""
        payload: dict[str, Any] = {"tool": tool_name, "success": success}
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        return self.log(user_id, "tool_call", payload)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        user_id: str,
        *,
        since: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return usage events for *user_id*, newest first."""
        return self._store.query_events(user_id, since, event_type, limit)

    def count(self, user_id: str) -> int:
        """Total event count for *user_id*."""
        return self._store.event_count(user_id)

    def recent_chat_events(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Shortcut for recent chat_completion events."""
        return self.query(user_id, event_type="chat_completion", limit=limit)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying database handle.

        The Rust ``UsageStore`` drops the ``redb`` handle on ``__del__``,
        so explicit close is optional but recommended in long-running
        servers.
        """
        del self._store


#: Global singleton for convenience in CLI / server contexts.
_DEFAULT_PIPELINE: UsagePipeline | None = None


def get_pipeline(db_path: str | None = None) -> UsagePipeline:
    """Return the default global pipeline instance."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = UsagePipeline(db_path)
    return _DEFAULT_PIPELINE


def _close_default() -> None:  # pragma: no cover
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is not None:
        _DEFAULT_PIPELINE.close()
        _DEFAULT_PIPELINE = None


atexit.register(_close_default)
