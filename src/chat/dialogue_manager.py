from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class DialogueState(str, Enum):
    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"
    TASK_EXECUTION = "task_execution"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    CLOSING = "closing"
    ERROR = "error"


_CLARIFICATION_KEYWORDS = frozenset(
    ["what do you mean", "could you explain", "clarify"]
)
_CLOSING_KEYWORDS = frozenset(["thank you", "thanks", "bye", "goodbye", "done"])


@dataclass
class Turn:
    turn_id: int
    role: str
    content: str
    intent: str | None = None
    entities: dict = field(default_factory=dict)
    state: DialogueState = DialogueState.INFORMATION_GATHERING
    timestamp: float = field(default_factory=time.time)


@dataclass
class DialogueContext:
    dialogue_id: str
    turns: list[Turn]
    current_state: DialogueState
    slot_values: dict
    goal: str | None
    created_at: float


class DialogueManager:
    """Multi-turn dialogue manager with state tracking and slot filling."""

    def __init__(self, slots: list[str] | None = None) -> None:
        self._required_slots: list[str] = slots or []
        self._dialogues: dict[str, DialogueContext] = {}

    def create_dialogue(self, goal: str | None = None) -> DialogueContext:
        ctx = DialogueContext(
            dialogue_id=str(uuid.uuid4()),
            turns=[],
            current_state=DialogueState.GREETING,
            slot_values={},
            goal=goal,
            created_at=time.time(),
        )
        self._dialogues[ctx.dialogue_id] = ctx
        return ctx

    def add_turn(
        self,
        dialogue_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        entities: dict | None = None,
    ) -> Turn:
        ctx = self._dialogues[dialogue_id]
        if entities:
            for k, v in entities.items():
                if k in self._required_slots:
                    ctx.slot_values[k] = v

        incoming_user_content = content if role == "user" else None
        new_state = self.update_state(ctx, incoming_user_content=incoming_user_content)
        ctx.current_state = new_state

        turn = Turn(
            turn_id=len(ctx.turns),
            role=role,
            content=content,
            intent=intent,
            entities=entities or {},
            state=new_state,
        )
        ctx.turns.append(turn)
        return turn

    def update_state(
        self,
        ctx: DialogueContext,
        incoming_user_content: str | None = None,
    ) -> DialogueState:
        has_prior_turns = bool(ctx.turns)

        if not has_prior_turns and incoming_user_content is None:
            return DialogueState.GREETING

        if incoming_user_content is not None:
            candidate = incoming_user_content.lower()
        else:
            candidate = ""
            for t in reversed(ctx.turns):
                if t.role == "user":
                    candidate = t.content.lower()
                    break

        if not has_prior_turns and incoming_user_content is not None:
            return DialogueState.GREETING

        if any(kw in candidate for kw in _CLOSING_KEYWORDS):
            return DialogueState.CLOSING

        if any(kw in candidate for kw in _CLARIFICATION_KEYWORDS):
            return DialogueState.CLARIFICATION

        if self._required_slots and all(
            s in ctx.slot_values for s in self._required_slots
        ):
            return DialogueState.CONFIRMATION

        if not self._required_slots:
            return DialogueState.TASK_EXECUTION

        filled = sum(1 for s in self._required_slots if s in ctx.slot_values)
        ratio = filled / len(self._required_slots)
        return (
            DialogueState.TASK_EXECUTION
            if ratio >= 0.5
            else DialogueState.INFORMATION_GATHERING
        )

    def get_context(self, dialogue_id: str) -> DialogueContext | None:
        return self._dialogues.get(dialogue_id)

    def get_history(
        self, dialogue_id: str, last_n: int | None = None
    ) -> list[Turn]:
        ctx = self._dialogues.get(dialogue_id)
        if ctx is None:
            return []
        return ctx.turns if last_n is None else ctx.turns[-last_n:]

    def fill_slot(self, dialogue_id: str, slot: str, value: str) -> None:
        ctx = self._dialogues[dialogue_id]
        ctx.slot_values[slot] = value

    def is_complete(self, dialogue_id: str) -> bool:
        ctx = self._dialogues.get(dialogue_id)
        if ctx is None:
            return False
        if ctx.current_state is DialogueState.CLOSING:
            return True
        if self._required_slots:
            return all(s in ctx.slot_values for s in self._required_slots)
        return False

    def list_dialogues(self) -> list[str]:
        return list(self._dialogues.keys())
