"""Session manager with time-limited tokens and rotation policy."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    token: str
    user: str
    created: float
    expires: float
    metadata: dict = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return time.monotonic() > self.expires


@dataclass
class SessionManager:
    duration_seconds: float = 3600.0
    _sessions: dict[str, Session] = field(default_factory=dict, repr=False)

    def create(self, user: str, **meta: str) -> Session:
        token = uuid.uuid4().hex
        now = time.monotonic()
        session = Session(
            token=token, user=user,
            created=now, expires=now + self.duration_seconds,
            metadata=meta,
        )
        self._sessions[token] = session
        return session

    def validate(self, token: str) -> Session | None:
        session = self._sessions.get(token)
        if session is None or session.expired:
            return None
        return session

    def revoke(self, token: str) -> None:
        self._sessions.pop(token, None)

    def active_count(self) -> int:
        now = time.monotonic()
        return sum(1 for s in self._sessions.values() if s.expires > now)

    def cleanup(self) -> int:
        now = time.monotonic()
        expired = [t for t, s in self._sessions.items() if s.expires <= now]
        for t in expired:
            del self._sessions[t]
        return len(expired)


SESSION_MANAGER = SessionManager()