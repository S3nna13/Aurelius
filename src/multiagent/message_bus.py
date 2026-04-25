"""Multi-agent message bus for inter-agent communication."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    msg_type: str
    payload: Any = None
    reply_to: str | None = None


@dataclass
class MessageBus:
    _inbox: dict[str, list[AgentMessage]] = field(default_factory=dict, repr=False)

    def send(self, message: AgentMessage) -> None:
        self._inbox.setdefault(message.recipient, []).append(message)

    def receive(self, agent_id: str) -> list[AgentMessage]:
        return self._inbox.pop(agent_id, [])

    def pending_count(self, agent_id: str | None = None) -> int:
        if agent_id:
            return len(self._inbox.get(agent_id, []))
        return sum(len(v) for v in self._inbox.values())


MESSAGE_BUS = MessageBus()