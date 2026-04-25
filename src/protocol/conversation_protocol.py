"""Conversation protocol: turn validation, role sequencing, state machine."""
from __future__ import annotations

from enum import Enum


class ConversationState(str, Enum):
    INIT = "init"
    USER_TURN = "user_turn"
    ASSISTANT_TURN = "assistant_turn"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ENDED = "ended"


class TurnValidationError(Exception):
    pass


_VALID_TRANSITIONS: dict[ConversationState, dict[str, ConversationState]] = {
    ConversationState.INIT: {
        "user": ConversationState.USER_TURN,
    },
    ConversationState.USER_TURN: {
        "assistant": ConversationState.ASSISTANT_TURN,
    },
    ConversationState.ASSISTANT_TURN: {
        "user": ConversationState.USER_TURN,
        "tool_call": ConversationState.TOOL_CALL,
        "end": ConversationState.ENDED,
    },
    ConversationState.TOOL_CALL: {
        "tool_result": ConversationState.TOOL_RESULT,
    },
    ConversationState.TOOL_RESULT: {
        "assistant": ConversationState.ASSISTANT_TURN,
    },
    ConversationState.ENDED: {},
}


class ConversationProtocol:
    def __init__(self) -> None:
        self._state = ConversationState.INIT
        self._history: list[tuple[str, ConversationState]] = []

    @property
    def state(self) -> ConversationState:
        return self._state

    def transition(self, role: str) -> ConversationState:
        if self._state == ConversationState.ENDED:
            raise TurnValidationError("conversation ended")
        transitions = _VALID_TRANSITIONS.get(self._state, {})
        if role not in transitions:
            valid = list(transitions.keys())
            raise TurnValidationError(
                f"Invalid role {role!r} from state {self._state.value!r}; "
                f"valid roles: {valid}"
            )
        new_state = transitions[role]
        self._history.append((role, new_state))
        self._state = new_state
        return new_state

    def reset(self) -> None:
        self._state = ConversationState.INIT
        self._history = []

    def valid_next_roles(self) -> list[str]:
        return list(_VALID_TRANSITIONS.get(self._state, {}).keys())

    def history(self) -> list[tuple[str, ConversationState]]:
        return list(self._history)


CONVERSATION_PROTOCOL = ConversationProtocol()
