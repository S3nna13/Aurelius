"""Unit tests for src.ui.session_manager.

18 tests covering all public behaviour of SessionManager.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.session_manager import (
    DEFAULT_SESSION_MANAGER,
    SESSION_MANAGER_REGISTRY,
    AureliusSession,
    SessionManager,
    SessionManagerError,
    SessionState,
)

# ---------------------------------------------------------------------------
# AureliusSession dataclass
# ---------------------------------------------------------------------------


def test_aurelius_session_to_dict_round_trip() -> None:
    import time

    now = time.time()
    session = AureliusSession(
        session_id="abc-123",
        name="Test Session",
        state=SessionState.ACTIVE,
        created_at=now,
        updated_at=now,
        metadata={"key": "val"},
        transcript_entries=[{"role": "user", "content": "hi"}],
        task_count=3,
    )
    d = session.to_dict()
    restored = AureliusSession.from_dict(d)
    assert restored.session_id == session.session_id
    assert restored.name == session.name
    assert restored.state == session.state
    assert restored.task_count == 3


def test_aurelius_session_from_dict_missing_key_raises() -> None:
    with pytest.raises(SessionManagerError):
        AureliusSession.from_dict({"session_id": "x"})  # missing required keys


# ---------------------------------------------------------------------------
# SessionState enum
# ---------------------------------------------------------------------------


def test_session_state_variants_exist() -> None:
    assert SessionState.ACTIVE
    assert SessionState.PAUSED
    assert SessionState.ARCHIVED
    assert SessionState.CLOSED


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


def test_create_returns_active_session() -> None:
    manager = SessionManager()
    session = manager.create("My Session")
    assert isinstance(session, AureliusSession)
    assert session.state == SessionState.ACTIVE
    assert session.name == "My Session"
    assert len(session.session_id) > 0


def test_create_assigns_unique_ids() -> None:
    manager = SessionManager()
    s1 = manager.create("Session A")
    s2 = manager.create("Session B")
    assert s1.session_id != s2.session_id


def test_create_with_metadata() -> None:
    manager = SessionManager()
    session = manager.create("Meta Session", metadata={"project": "Aurelius"})
    assert session.metadata["project"] == "Aurelius"


def test_create_empty_name_raises() -> None:
    manager = SessionManager()
    with pytest.raises(SessionManagerError):
        manager.create("")


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


def test_get_retrieves_by_session_id() -> None:
    manager = SessionManager()
    session = manager.create("Retrieve Me")
    retrieved = manager.get(session.session_id)
    assert retrieved.session_id == session.session_id


def test_get_unknown_id_raises_session_manager_error() -> None:
    manager = SessionManager()
    with pytest.raises(SessionManagerError):
        manager.get("nonexistent-id")


# ---------------------------------------------------------------------------
# update_state() / archive() / close()
# ---------------------------------------------------------------------------


def test_update_state_changes_state() -> None:
    manager = SessionManager()
    session = manager.create("State Test")
    manager.update_state(session.session_id, SessionState.PAUSED)
    assert manager.get(session.session_id).state == SessionState.PAUSED


def test_update_state_invalid_type_raises() -> None:
    manager = SessionManager()
    session = manager.create("Bad State")
    with pytest.raises(SessionManagerError):
        manager.update_state(session.session_id, "active")  # type: ignore[arg-type]


def test_archive_sets_archived_state() -> None:
    manager = SessionManager()
    session = manager.create("Archive Me")
    manager.archive(session.session_id)
    assert manager.get(session.session_id).state == SessionState.ARCHIVED


def test_close_sets_closed_state() -> None:
    manager = SessionManager()
    session = manager.create("Close Me")
    manager.close(session.session_id)
    assert manager.get(session.session_id).state == SessionState.CLOSED


# ---------------------------------------------------------------------------
# list_sessions()
# ---------------------------------------------------------------------------


def test_list_sessions_returns_all() -> None:
    manager = SessionManager()
    manager.create("A")
    manager.create("B")
    manager.create("C")
    sessions = manager.list_sessions()
    assert len(sessions) == 3


def test_list_sessions_state_filter_active_only() -> None:
    manager = SessionManager()
    s1 = manager.create("Active")
    s2 = manager.create("To Archive")
    manager.archive(s2.session_id)
    active = manager.list_sessions(state_filter=SessionState.ACTIVE)
    assert len(active) == 1
    assert active[0].session_id == s1.session_id


# ---------------------------------------------------------------------------
# switch_to()
# ---------------------------------------------------------------------------


def test_switch_to_pauses_other_active_sessions() -> None:
    manager = SessionManager()
    s1 = manager.create("Session 1")
    s2 = manager.create("Session 2")
    # Both are ACTIVE after creation.
    manager.switch_to(s2.session_id)
    assert manager.get(s1.session_id).state == SessionState.PAUSED
    assert manager.get(s2.session_id).state == SessionState.ACTIVE


def test_switch_to_unknown_id_raises() -> None:
    manager = SessionManager()
    with pytest.raises(SessionManagerError):
        manager.switch_to("bogus-id")


# ---------------------------------------------------------------------------
# save_to_dict() / load_from_dict() round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip_restores_sessions() -> None:
    manager = SessionManager()
    s1 = manager.create("Persist A")
    s2 = manager.create("Persist B")
    manager.archive(s2.session_id)

    snapshot = manager.save_to_dict()

    new_manager = SessionManager()
    new_manager.load_from_dict(snapshot)

    restored_s1 = new_manager.get(s1.session_id)
    restored_s2 = new_manager.get(s2.session_id)
    assert restored_s1.name == "Persist A"
    assert restored_s1.state == SessionState.ACTIVE
    assert restored_s2.state == SessionState.ARCHIVED


def test_load_from_dict_invalid_type_raises() -> None:
    manager = SessionManager()
    with pytest.raises(SessionManagerError):
        manager.load_from_dict("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_does_not_crash() -> None:
    manager = SessionManager()
    manager.create("Alpha")
    manager.create("Beta")
    console = Console(record=True)
    manager.render(console)
    output = console.export_text()
    assert len(output) > 0


def test_render_empty_manager_does_not_crash() -> None:
    manager = SessionManager()
    console = Console(record=True)
    manager.render(console)


# ---------------------------------------------------------------------------
# Registry / singleton / error class
# ---------------------------------------------------------------------------


def test_default_session_manager_is_session_manager_instance() -> None:
    assert isinstance(DEFAULT_SESSION_MANAGER, SessionManager)


def test_session_manager_registry_is_dict() -> None:
    assert isinstance(SESSION_MANAGER_REGISTRY, dict)


def test_session_manager_error_is_exception_subclass() -> None:
    assert issubclass(SessionManagerError, Exception)
