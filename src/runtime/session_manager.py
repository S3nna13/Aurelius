"""Inference session lifecycle manager for Aurelius runtime."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto


class SessionState(Enum):
    ACTIVE = auto()
    IDLE = auto()
    EXPIRED = auto()
    TERMINATED = auto()


@dataclass
class Session:
    session_id: str
    model_id: str
    created_at: float
    last_active: float
    state: SessionState = SessionState.ACTIVE
    metadata: dict = field(default_factory=dict)


class SessionManager:
    """Manages inference session lifecycle including creation, expiry, and cleanup."""

    def __init__(
        self,
        max_sessions: int = 100,
        idle_timeout_s: float = 300.0,
    ) -> None:
        self.max_sessions = max_sessions
        self.idle_timeout_s = idle_timeout_s
        self._sessions: dict[str, Session] = {}

    def create(self, model_id: str, metadata: dict | None = None) -> Session:
        """Create a new session. Raises ValueError if max_sessions reached."""
        if len(self._sessions) >= self.max_sessions:
            raise ValueError(
                f"Cannot create session: max_sessions limit ({self.max_sessions}) reached."
            )
        now = time.monotonic()
        session_id = uuid.uuid4().hex[:12]
        session = Session(
            session_id=session_id,
            model_id=model_id,
            created_at=now,
            last_active=now,
            state=SessionState.ACTIVE,
            metadata=metadata if metadata is not None else {},
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Return the Session with the given id, or None if not found."""
        return self._sessions.get(session_id)

    def touch(self, session_id: str) -> bool:
        """Update last_active and set state to ACTIVE. Returns False if not found."""
        if session_id not in self._sessions:
            return False
        session = self._sessions[session_id]
        session.last_active = time.monotonic()
        session.state = SessionState.ACTIVE
        return True

    def expire_idle(self, now: float | None = None) -> list[str]:
        """Mark idle sessions as EXPIRED and return their ids."""
        if now is None:
            now = time.monotonic()
        expired_ids: list[str] = []
        for session in self._sessions.values():
            if session.state in (SessionState.ACTIVE, SessionState.IDLE):
                if (now - session.last_active) > self.idle_timeout_s:
                    session.state = SessionState.EXPIRED
                    expired_ids.append(session.session_id)
        return expired_ids

    def terminate(self, session_id: str) -> bool:
        """Set session state to TERMINATED. Returns False if not found."""
        if session_id not in self._sessions:
            return False
        self._sessions[session_id].state = SessionState.TERMINATED
        return True

    def active_count(self) -> int:
        """Return count of sessions with state == ACTIVE."""
        return sum(
            1 for s in self._sessions.values() if s.state is SessionState.ACTIVE
        )

    def cleanup(self) -> int:
        """Remove TERMINATED and EXPIRED sessions. Returns count removed."""
        removable = [
            sid
            for sid, s in self._sessions.items()
            if s.state in (SessionState.TERMINATED, SessionState.EXPIRED)
        ]
        for sid in removable:
            del self._sessions[sid]
        return len(removable)


SESSION_MANAGER_REGISTRY: dict[str, type[SessionManager]] = {"default": SessionManager}
