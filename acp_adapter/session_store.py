"""Aurelius MCP — session_store.py

Persistent session store for MCP sessions with TTL-based expiry.
All logic is pure Python stdlib; no external deps.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MCPSession:
    session_id: str
    client_id: str
    created_at: float
    data: dict = field(default_factory=dict)
    expires_at: float | None = None


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """In-memory session store with capacity limits and TTL expiry."""

    def __init__(
        self,
        max_sessions: int = 1000,
        default_ttl_s: float = 3600.0,
    ) -> None:
        self._max_sessions = max_sessions
        self._default_ttl_s = default_ttl_s
        self._sessions: dict[str, MCPSession] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, session: MCPSession, now: float) -> bool:
        return session.expires_at is not None and now > session.expires_at

    def _now(self, now: float | None) -> float:
        return now if now is not None else time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        client_id: str,
        data: dict | None = None,
        ttl_s: float | None = None,
    ) -> MCPSession:
        """Create and store a new session.

        Raises ValueError if the store is at capacity.
        """
        if len(self._sessions) >= self._max_sessions:
            raise ValueError(f"Session store at capacity ({self._max_sessions} sessions)")

        now = time.monotonic()
        session_id = uuid.uuid4().hex[:16]
        effective_ttl = ttl_s if ttl_s is not None else self._default_ttl_s
        expires_at = now + effective_ttl

        session = MCPSession(
            session_id=session_id,
            client_id=client_id,
            created_at=now,
            data=dict(data) if data else {},
            expires_at=expires_at,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str, now: float | None = None) -> MCPSession | None:
        """Return the session or None if not found / expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._is_expired(session, self._now(now)):
            return None
        return session

    def update(self, session_id: str, data: dict, now: float | None = None) -> bool:
        """Merge *data* into session.data.  Returns False if missing/expired."""
        session = self.get(session_id, now=now)
        if session is None:
            return False
        session.data.update(data)
        return True

    def delete(self, session_id: str) -> bool:
        """Remove a session.  Returns True if it existed, False otherwise."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def expire_all(self, now: float | None = None) -> int:
        """Remove all expired sessions.  Returns the count removed."""
        ts = self._now(now)
        to_delete = [
            sid for sid, session in self._sessions.items() if self._is_expired(session, ts)
        ]
        for sid in to_delete:
            del self._sessions[sid]
        return len(to_delete)

    def active_count(self, now: float | None = None) -> int:
        """Return the number of non-expired sessions."""
        ts = self._now(now)
        return sum(1 for session in self._sessions.values() if not self._is_expired(session, ts))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SESSION_STORE_REGISTRY: dict[str, type[SessionStore]] = {"default": SessionStore}
