"""Manages agent lifecycle: spawn, pause, terminate."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"
    CRASHED = "crashed"


@dataclass
class AgentRecord:
    agent_id: str
    name: str
    status: AgentStatus
    created_at: float
    metadata: dict = field(default_factory=dict)


class AgentLifecycleManager:
    def __init__(self, max_agents: int = 50) -> None:
        self.max_agents = max_agents
        self._agents: dict[str, AgentRecord] = {}

    def spawn(self, name: str, metadata: dict | None = None) -> AgentRecord:
        active = [
            r for r in self._agents.values()
            if r.status not in (AgentStatus.TERMINATED, AgentStatus.CRASHED)
        ]
        if len(active) >= self.max_agents:
            raise ValueError(
                f"Cannot spawn agent '{name}': max_agents limit ({self.max_agents}) reached."
            )
        agent_id = uuid.uuid4().hex[:10]
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            status=AgentStatus.INITIALIZING,
            created_at=time.monotonic(),
            metadata=dict(metadata) if metadata else {},
        )
        self._agents[agent_id] = record
        return record

    def start(self, agent_id: str) -> bool:
        record = self._agents.get(agent_id)
        if record is None:
            return False
        if record.status != AgentStatus.INITIALIZING:
            return False
        record.status = AgentStatus.RUNNING
        return True

    def pause(self, agent_id: str) -> bool:
        record = self._agents.get(agent_id)
        if record is None:
            return False
        if record.status != AgentStatus.RUNNING:
            return False
        record.status = AgentStatus.PAUSED
        return True

    def resume(self, agent_id: str) -> bool:
        record = self._agents.get(agent_id)
        if record is None:
            return False
        if record.status != AgentStatus.PAUSED:
            return False
        record.status = AgentStatus.RUNNING
        return True

    def terminate(self, agent_id: str) -> bool:
        record = self._agents.get(agent_id)
        if record is None:
            return False
        record.status = AgentStatus.TERMINATED
        return True

    def crash(self, agent_id: str, reason: str = "") -> bool:
        record = self._agents.get(agent_id)
        if record is None:
            return False
        record.status = AgentStatus.CRASHED
        record.metadata["crash_reason"] = reason
        return True

    def get(self, agent_id: str) -> AgentRecord | None:
        return self._agents.get(agent_id)

    def list_by_status(self, status: AgentStatus) -> list[AgentRecord]:
        return sorted(
            (r for r in self._agents.values() if r.status == status),
            key=lambda r: r.agent_id,
        )

    def active_count(self) -> int:
        return sum(
            1 for r in self._agents.values()
            if r.status in (AgentStatus.RUNNING, AgentStatus.PAUSED)
        )


AGENT_LIFECYCLE_REGISTRY: dict[str, type] = {"default": AgentLifecycleManager}
