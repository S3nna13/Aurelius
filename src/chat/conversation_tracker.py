"""Multi-turn conversation state tracker for agent dialogues."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Turn:
    role: str  # user, assistant, system, tool
    content: str
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationState:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def last_n(self, n: int = 5) -> list[Turn]:
        return self.turns[-n:]

    def token_count(self) -> int:
        return sum(len(t.content.split()) for t in self.turns)

    def summary(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": len(self.turns),
            "tokens": self.token_count(),
            "last_role": self.turns[-1].role if self.turns else None,
        }


@dataclass
class ConversationTracker:
    _conversations: dict[str, ConversationState] = field(default_factory=dict, repr=False)

    def start(self, conversation_id: str) -> ConversationState:
        state = ConversationState(conversation_id=conversation_id)
        self._conversations[conversation_id] = state
        return state

    def get(self, conversation_id: str) -> ConversationState | None:
        return self._conversations.get(conversation_id)

    def add_turn(self, conversation_id: str, turn: Turn) -> None:
        state = self.get(conversation_id)
        if state is None:
            state = self.start(conversation_id)
        state.add_turn(turn)


CONVERSATION_TRACKER = ConversationTracker()