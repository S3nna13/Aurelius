"""Multi-session persistence and restore for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli (MIT, multi-tab session lifecycle), Anthropic Claude Code
(MIT, streaming output), clean-room reimplementation with original Aurelius design.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


class SessionManagerError(Exception):
    """Raised when the session manager encounters malformed state or input."""


class SessionState(enum.Enum):
    """Lifecycle state of an :class:`AureliusSession`."""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    CLOSED = "closed"


_STATE_STYLES: dict[SessionState, str] = {
    SessionState.ACTIVE: "bold green",
    SessionState.PAUSED: "yellow",
    SessionState.ARCHIVED: "dim",
    SessionState.CLOSED: "dim red",
}


@dataclass
class AureliusSession:
    """A single named Aurelius session.

    Attributes:
        session_id: Unique identifier (UUID string).
        name: Human-readable session name.
        state: Current :class:`SessionState`.
        created_at: Unix timestamp of session creation.
        updated_at: Unix timestamp of the last state mutation.
        metadata: Arbitrary key/value metadata.
        transcript_entries: Ordered list of serialised transcript dicts.
        task_count: Number of tasks associated with this session.
    """

    session_id: str
    name: str
    state: SessionState
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    transcript_entries: list[dict[str, Any]] = field(default_factory=list)
    task_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of this session."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "transcript_entries": list(self.transcript_entries),
            "task_count": self.task_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AureliusSession:
        """Restore an :class:`AureliusSession` from a serialised snapshot.

        Args:
            data: A dict previously produced by :meth:`to_dict`.

        Raises:
            SessionManagerError: If required keys are missing.
        """
        try:
            return cls(
                session_id=data["session_id"],
                name=data["name"],
                state=SessionState(data["state"]),
                created_at=float(data["created_at"]),
                updated_at=float(data["updated_at"]),
                metadata=dict(data.get("metadata", {})),
                transcript_entries=list(data.get("transcript_entries", [])),
                task_count=int(data.get("task_count", 0)),
            )
        except (KeyError, ValueError) as exc:
            raise SessionManagerError(f"failed to restore AureliusSession: {exc}") from exc


class SessionManager:
    """Manages a collection of :class:`AureliusSession` objects.

    Supports creation, retrieval, state transitions, serialisation, and
    Rich table rendering.  Suitable for multi-tab or multi-session terminal
    applications (kimi-cli parity).

    Attributes:
        _sessions: Internal dict mapping *session_id* → :class:`AureliusSession`.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, AureliusSession] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> AureliusSession:
        """Create a new :class:`AureliusSession` with state :attr:`SessionState.ACTIVE`.

        Args:
            name: Human-readable session name (non-empty string).
            metadata: Optional key/value metadata dict.

        Returns:
            The newly created :class:`AureliusSession`.

        Raises:
            SessionManagerError: If *name* is not a non-empty string.
        """
        if not isinstance(name, str) or not name.strip():
            raise SessionManagerError("name must be a non-empty string")
        now = time.time()
        session = AureliusSession(
            session_id=str(uuid.uuid4()),
            name=name,
            state=SessionState.ACTIVE,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> AureliusSession:
        """Retrieve a session by *session_id*.

        Args:
            session_id: The UUID string of the session.

        Returns:
            The matching :class:`AureliusSession`.

        Raises:
            SessionManagerError: If no session with that id exists.
        """
        if session_id not in self._sessions:
            raise SessionManagerError(f"unknown session: {session_id!r}")
        return self._sessions[session_id]

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def update_state(self, session_id: str, state: SessionState) -> None:
        """Mutate the state of a session.

        Args:
            session_id: Target session identifier.
            state: New :class:`SessionState`.

        Raises:
            SessionManagerError: If the session does not exist or *state* is
                not a :class:`SessionState`.
        """
        if not isinstance(state, SessionState):
            raise SessionManagerError(f"state must be a SessionState, got {type(state).__name__}")
        session = self.get(session_id)
        session.state = state
        session.updated_at = time.time()

    def archive(self, session_id: str) -> None:
        """Set session state to :attr:`SessionState.ARCHIVED`.

        Args:
            session_id: Target session identifier.
        """
        self.update_state(session_id, SessionState.ARCHIVED)

    def close(self, session_id: str) -> None:
        """Set session state to :attr:`SessionState.CLOSED`.

        Args:
            session_id: Target session identifier.
        """
        self.update_state(session_id, SessionState.CLOSED)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_sessions(
        self,
        state_filter: SessionState | None = None,
    ) -> list[AureliusSession]:
        """Return all sessions, optionally filtered by *state_filter*.

        Args:
            state_filter: When given, only sessions with this state are returned.

        Returns:
            List of matching :class:`AureliusSession` instances.
        """
        sessions = list(self._sessions.values())
        if state_filter is not None:
            sessions = [s for s in sessions if s.state == state_filter]
        return sessions

    # ------------------------------------------------------------------
    # Multi-tab switching (kimi-cli parity)
    # ------------------------------------------------------------------

    def switch_to(self, session_id: str) -> AureliusSession:
        """Activate a session, pausing all currently ACTIVE sessions.

        All sessions currently in :attr:`SessionState.ACTIVE` (except the
        target) are transitioned to :attr:`SessionState.PAUSED`.  The target
        session is set to :attr:`SessionState.ACTIVE`.

        Args:
            session_id: The session to activate.

        Returns:
            The newly activated :class:`AureliusSession`.

        Raises:
            SessionManagerError: If the session does not exist.
        """
        # Verify target exists first (raises if not found).
        target = self.get(session_id)
        now = time.time()
        for sid, session in self._sessions.items():
            if sid != session_id and session.state == SessionState.ACTIVE:
                session.state = SessionState.PAUSED
                session.updated_at = now
        target.state = SessionState.ACTIVE
        target.updated_at = now
        return target

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all sessions.

        Returns:
            A dict with a ``sessions`` key mapping *session_id* → session dict.
        """
        return {"sessions": {sid: session.to_dict() for sid, session in self._sessions.items()}}

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Restore sessions from a snapshot previously produced by :meth:`save_to_dict`.

        Existing sessions are replaced entirely.

        Args:
            data: A snapshot dict.

        Raises:
            SessionManagerError: If *data* is malformed.
        """
        if not isinstance(data, dict):
            raise SessionManagerError("data must be a dict")
        sessions_raw = data.get("sessions", {})
        if not isinstance(sessions_raw, dict):
            raise SessionManagerError("data['sessions'] must be a dict")
        restored: dict[str, AureliusSession] = {}
        for sid, raw in sessions_raw.items():
            session = AureliusSession.from_dict(raw)
            restored[sid] = session
        self._sessions = restored

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, console: Console) -> None:
        """Render all sessions as a Rich Table.

        Args:
            console: A :class:`~rich.console.Console` to print to.
        """
        table = Table(
            title="Aurelius Sessions",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("State", justify="center")
        table.add_column("Created", justify="right")
        table.add_column("Tasks", justify="right")

        for session in self._sessions.values():
            state_style = _STATE_STYLES.get(session.state, "")
            state_label = Text(session.state.value.upper(), style=state_style)
            created = _format_timestamp(session.created_at)
            table.add_row(
                session.session_id[:8] + "…",
                session.name,
                state_label,
                created,
                str(session.task_count),
            )

        console.print(table)


# ---------------------------------------------------------------------------
# Registry and default singleton
# ---------------------------------------------------------------------------

#: Named pool of :class:`SessionManager` instances.
SESSION_MANAGER_REGISTRY: dict[str, SessionManager] = {}

#: Default singleton :class:`SessionManager` used when no named manager is specified.
DEFAULT_SESSION_MANAGER: SessionManager = SessionManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_timestamp(ts: float) -> str:
    """Format a Unix timestamp as a human-readable local time string."""
    import datetime

    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
