"""Agent runtime — multi-agent orchestration with lifecycle management."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: str
    msg_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    role: str = "general"
    system_prompt: str = "You are a helpful AI agent."
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 20
    status: AgentStatus = AgentStatus.IDLE


class AgentRuntime:
    """Runtime for managing multiple agents with message passing."""

    def __init__(self, persist_path: str | None = None):
        self.agents: dict[str, AgentSpec] = {}
        self.mailbox: list[AgentMessage] = []
        self._handlers: dict[str, Callable] = {}
        self._persist_path = persist_path

    def register_agent(self, spec: AgentSpec) -> str:
        self.agents[spec.id] = spec
        if self._persist_path:
            self._save()
        return spec.id

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            del self.agents[agent_id]
            if self._persist_path:
                self._save()
            return True
        return False

    def _save(self) -> None:
        with suppress(Exception):
            from .agent_persistence import AgentPersistence

            ap = AgentPersistence(self._persist_path)
            ap.save_agents(self)

    def send_message(self, msg: AgentMessage) -> None:
        self.mailbox.append(msg)

    def get_messages(self, agent_id: str) -> list[AgentMessage]:
        return [m for m in self.mailbox if m.recipient == agent_id]

    def route_message(self, content: str, sender: str, recipient: str) -> AgentMessage:
        msg = AgentMessage(sender=sender, recipient=recipient, content=content)
        self.send_message(msg)
        return msg

    def get_agent(self, agent_id: str) -> AgentSpec | None:
        return self.agents.get(agent_id)

    def list_agents_by_role(self, role: str) -> list[AgentSpec]:
        return [a for a in self.agents.values() if a.role == role]
