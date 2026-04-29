"""Agent runtime — multi-agent orchestration with lifecycle management."""

from __future__ import annotations

import uuid
from collections.abc import Callable
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

    def __init__(self):
        self.agents: dict[str, AgentSpec] = {}
        self.mailbox: list[AgentMessage] = []
        self._handlers: dict[str, Callable] = {}

    def register_agent(self, spec: AgentSpec) -> str:
        self.agents[spec.id] = spec
        return spec.id

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
