"""Multi-agent conversation: N agents exchanging messages with routing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    USER_PROXY = "user_proxy"


@dataclass
class AgentProfile:
    name: str
    role: AgentRole
    system_prompt: str = ""
    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class ConversationMessage:
    from_agent: str
    to_agent: str | None
    content: str
    thread_id: str = "main"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MultiAgentConversation:
    def __init__(self, max_turns: int = 50) -> None:
        self.max_turns = max_turns
        self._agents: dict[str, AgentProfile] = {}
        self._messages: list[ConversationMessage] = []

    def add_agent(self, profile: AgentProfile) -> None:
        self._agents[profile.agent_id] = profile

    def send(
        self,
        from_id: str,
        content: str,
        to_id: str | None = None,
        thread_id: str = "main",
    ) -> ConversationMessage:
        msg = ConversationMessage(
            from_agent=from_id,
            to_agent=to_id,
            content=content,
            thread_id=thread_id,
        )
        self._messages.append(msg)
        return msg

    def messages(
        self,
        thread_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[ConversationMessage]:
        result = self._messages
        if thread_id is not None:
            result = [m for m in result if m.thread_id == thread_id]
        if agent_id is not None:
            result = [
                m for m in result if m.from_agent == agent_id or m.to_agent == agent_id
            ]
        return result

    def agents(self) -> list[AgentProfile]:
        return list(self._agents.values())

    def turn_count(self) -> int:
        return len(self._messages)

    def thread_ids(self) -> list[str]:
        seen: list[str] = []
        for m in self._messages:
            if m.thread_id not in seen:
                seen.append(m.thread_id)
        return seen
