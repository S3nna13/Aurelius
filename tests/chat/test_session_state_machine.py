"""Tests for src/chat/session_state_machine.py."""

import pytest

from src.chat.session_state_machine import (
    ChatSessionStateMachine,
    SessionEvent,
    SessionPhase,
    SessionTransition,
)


# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------

def test_session_event_values():
    assert SessionEvent.START == "start"
    assert SessionEvent.END == "end"
    assert len(SessionEvent) == 8


def test_session_phase_values():
    assert SessionPhase.IDLE == "idle"
    assert SessionPhase.ENDED == "ended"
    assert len(SessionPhase) == 6


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_phase_is_idle():
    sm = ChatSessionStateMachine()
    assert sm.current_phase == SessionPhase.IDLE


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------

def test_idle_to_awaiting_input():
    sm = ChatSessionStateMachine()
    phase = sm.transition(SessionEvent.START)
    assert phase == SessionPhase.AWAITING_INPUT


def test_awaiting_input_to_generating():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.START)
    phase = sm.transition(SessionEvent.USER_TURN)
    assert phase == SessionPhase.GENERATING


def test_generating_to_awaiting_input():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.START)
    sm.transition(SessionEvent.USER_TURN)
    phase = sm.transition(SessionEvent.ASSISTANT_TURN)
    assert phase == SessionPhase.AWAITING_INPUT


def test_generating_to_tool_execution():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.START)
    sm.transition(SessionEvent.USER_TURN)
    phase = sm.transition(SessionEvent.TOOL_CALL)
    assert phase == SessionPhase.TOOL_EXECUTION


def test_tool_execution_to_generating():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.START)
    sm.transition(SessionEvent.USER_TURN)
    sm.transition(SessionEvent.TOOL_CALL)
    phase = sm.transition(SessionEvent.TOOL_RESULT)
    assert phase == SessionPhase.GENERATING


# ---------------------------------------------------------------------------
# Universal events (PAUSE / RESUME / END)
# ---------------------------------------------------------------------------

def test_pause_from_generating():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.START)
    sm.transition(SessionEvent.USER_TURN)
    phase = sm.transition(SessionEvent.PAUSE)
    assert phase == SessionPhase.PAUSED


def test_resume_from_paused():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.START)
    sm.transition(SessionEvent.USER_TURN)
    sm.transition(SessionEvent.PAUSE)
    phase = sm.transition(SessionEvent.RESUME)
    assert phase == SessionPhase.AWAITING_INPUT


def test_end_from_any_phase():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.START)
    phase = sm.transition(SessionEvent.END)
    assert phase == SessionPhase.ENDED


def test_pause_from_idle():
    sm = ChatSessionStateMachine()
    phase = sm.transition(SessionEvent.PAUSE)
    assert phase == SessionPhase.PAUSED


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------

def test_invalid_user_turn_from_idle():
    sm = ChatSessionStateMachine()
    with pytest.raises(ValueError, match="Invalid transition"):
        sm.transition(SessionEvent.USER_TURN)


def test_cannot_transition_after_ended():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.END)
    with pytest.raises(ValueError, match="ENDED"):
        sm.transition(SessionEvent.START)


# ---------------------------------------------------------------------------
# can_transition
# ---------------------------------------------------------------------------

def test_can_transition_true():
    sm = ChatSessionStateMachine()
    assert sm.can_transition(SessionEvent.START) is True


def test_can_transition_false():
    sm = ChatSessionStateMachine()
    assert sm.can_transition(SessionEvent.USER_TURN) is False


def test_can_transition_false_after_ended():
    sm = ChatSessionStateMachine()
    sm.transition(SessionEvent.END)
    assert sm.can_transition(SessionEvent.PAUSE) is False


# ---------------------------------------------------------------------------
# SessionTransition dataclass
# ---------------------------------------------------------------------------

def test_session_transition_dataclass():
    t = SessionTransition(
        from_phase=SessionPhase.IDLE,
        event=SessionEvent.START,
        to_phase=SessionPhase.AWAITING_INPUT,
    )
    assert t.from_phase == SessionPhase.IDLE
    assert t.event == SessionEvent.START
    assert t.to_phase == SessionPhase.AWAITING_INPUT


def test_transitions_list_non_empty():
    assert len(ChatSessionStateMachine.TRANSITIONS) >= 5
