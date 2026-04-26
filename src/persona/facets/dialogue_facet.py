"""Dialogue facet — state machine for multi-turn conversation management."""

from __future__ import annotations

from enum import StrEnum

from ..unified_persona import PersonaFacet


class DialogueState(StrEnum):
    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"
    TASK_EXECUTION = "task_execution"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    CLOSING = "closing"
    ERROR = "error"


_CLARIFICATION_KEYWORDS = frozenset(["what do you mean", "could you explain", "clarify"])
_CLOSING_KEYWORDS = frozenset(["thank you", "thanks", "bye", "goodbye", "done"])


def create_dialogue_facet(required_slots: list[str] | None = None) -> PersonaFacet:
    return PersonaFacet(
        facet_type="dialogue",
        config={"required_slots": required_slots or [], "initial_state": "greeting"},
    )


def classify_transition(current_state: DialogueState, user_input: str, slots_filled: list[str] | None = None) -> DialogueState:
    text = user_input.lower()

    if any(kw in text for kw in _CLOSING_KEYWORDS):
        return DialogueState.CLOSING

    if any(kw in text for kw in _CLARIFICATION_KEYWORDS):
        return DialogueState.CLARIFICATION

    if current_state == DialogueState.GREETING:
        return DialogueState.INFORMATION_GATHERING

    if current_state == DialogueState.INFORMATION_GATHERING:
        if slots_filled:
            return DialogueState.TASK_EXECUTION
        return DialogueState.INFORMATION_GATHERING

    if current_state == DialogueState.TASK_EXECUTION:
        return DialogueState.CONFIRMATION

    if current_state == DialogueState.CLARIFICATION:
        return DialogueState.INFORMATION_GATHERING

    if current_state == DialogueState.CONFIRMATION:
        if any(w in text for w in ("yes", "correct", "exactly", "right", "sure", "okay", "ok")):
            return DialogueState.CLOSING
        return DialogueState.INFORMATION_GATHERING

    return DialogueState.INFORMATION_GATHERING