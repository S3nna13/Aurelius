"""Session state machine for managing chat session lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SessionEvent(StrEnum):
    START = "start"
    USER_TURN = "user_turn"
    ASSISTANT_TURN = "assistant_turn"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PAUSE = "pause"
    RESUME = "resume"
    END = "end"


class SessionPhase(StrEnum):
    IDLE = "idle"
    AWAITING_INPUT = "awaiting_input"
    GENERATING = "generating"
    TOOL_EXECUTION = "tool_execution"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class SessionTransition:
    from_phase: SessionPhase
    event: SessionEvent
    to_phase: SessionPhase


# Sentinel for "any phase"
_ANY = None


class ChatSessionStateMachine:
    """Finite state machine governing valid chat session transitions."""

    TRANSITIONS: list[SessionTransition] = [
        SessionTransition(SessionPhase.IDLE, SessionEvent.START, SessionPhase.AWAITING_INPUT),
        SessionTransition(
            SessionPhase.AWAITING_INPUT, SessionEvent.USER_TURN, SessionPhase.GENERATING
        ),
        SessionTransition(
            SessionPhase.GENERATING, SessionEvent.ASSISTANT_TURN, SessionPhase.AWAITING_INPUT
        ),
        SessionTransition(
            SessionPhase.GENERATING, SessionEvent.TOOL_CALL, SessionPhase.TOOL_EXECUTION
        ),
        SessionTransition(
            SessionPhase.TOOL_EXECUTION, SessionEvent.TOOL_RESULT, SessionPhase.GENERATING
        ),
    ]

    # Events that apply to *any* phase (with lower priority than phase-specific ones)
    _UNIVERSAL_EVENTS: dict[SessionEvent, SessionPhase] = {
        SessionEvent.PAUSE: SessionPhase.PAUSED,
        SessionEvent.END: SessionPhase.ENDED,
    }

    # RESUME is a special universal event — it always goes to AWAITING_INPUT
    _RESUME_TARGET: SessionPhase = SessionPhase.AWAITING_INPUT

    def __init__(self) -> None:
        self.current_phase: SessionPhase = SessionPhase.IDLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_target(self, event: SessionEvent) -> SessionPhase | None:
        """Return the target phase for *event* given current phase, or None."""
        # Phase-specific transitions take priority
        for t in self.TRANSITIONS:
            if t.from_phase == self.current_phase and t.event == event:
                return t.to_phase

        # Universal events
        if event in self._UNIVERSAL_EVENTS:
            return self._UNIVERSAL_EVENTS[event]

        if event == SessionEvent.RESUME:
            return self._RESUME_TARGET

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_transition(self, event: SessionEvent) -> bool:
        """Return True if *event* is valid in the current phase."""
        # Cannot leave ENDED
        if self.current_phase == SessionPhase.ENDED:
            return False
        return self._find_target(event) is not None

    def transition(self, event: SessionEvent) -> SessionPhase:
        """Apply *event* and return the new phase.

        Raises ValueError if the transition is not valid.
        """
        if self.current_phase == SessionPhase.ENDED:
            raise ValueError(f"Session is already ENDED; cannot process event {event!r}")

        target = self._find_target(event)
        if target is None:
            raise ValueError(f"Invalid transition: phase={self.current_phase!r}, event={event!r}")

        self.current_phase = target
        return self.current_phase
