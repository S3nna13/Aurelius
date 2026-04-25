"""Aurelius MCP — session_manager.py

Manages MCP session lifecycle: creation, activity tracking, idle pruning,
and graceful closure.  All logic is pure Python stdlib; no external deps.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class SessionState(str, Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class MCPSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.INITIALIZING
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MCPSessionManager
# ---------------------------------------------------------------------------

class MCPSessionManager:
    """Create and manage the lifecycle of MCPSession objects."""

    def __init__(self) -> None:
        self._sessions: dict[str, MCPSession] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_session(self, metadata: dict | None = None) -> MCPSession:
        """Create a new session in INITIALIZING state, transition to ACTIVE."""
        session = MCPSession(metadata=dict(metadata) if metadata else {})
        self._sessions[session.session_id] = session
        session.state = SessionState.ACTIVE
        return session

    def get_session(self, session_id: str) -> MCPSession | None:
        """Return the session with *session_id*, or None if not found."""
        return self._sessions.get(session_id)

    def update_activity(self, session_id: str) -> None:
        """Bump *last_activity* to now for the given session."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.last_activity = time.time()
        if session.state == SessionState.IDLE:
            session.state = SessionState.ACTIVE

    def close_session(self, session_id: str) -> None:
        """Transition a session CLOSING → CLOSED."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.state = SessionState.CLOSING
        session.state = SessionState.CLOSED

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def prune_idle(self, idle_timeout: float = 300.0) -> int:
        """Close sessions whose last_activity is older than *idle_timeout* seconds.

        Returns the number of sessions pruned.
        """
        cutoff = time.time() - idle_timeout
        pruned = 0
        for session in list(self._sessions.values()):
            if session.state in (SessionState.CLOSED, SessionState.CLOSING):
                continue
            if session.last_activity < cutoff:
                self.close_session(session.session_id)
                pruned += 1
        return pruned

    def list_active(self) -> list[MCPSession]:
        """Return all sessions in ACTIVE or IDLE state."""
        return [
            s for s in self._sessions.values()
            if s.state in (SessionState.ACTIVE, SessionState.IDLE)
        ]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sessions)

    def all_sessions(self) -> list[MCPSession]:
        return list(self._sessions.values())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MCP_REGISTRY: dict[str, Any] = {}
MCP_REGISTRY["session_manager"] = MCPSessionManager()
