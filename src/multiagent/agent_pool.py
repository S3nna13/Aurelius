"""Agent pool: spawn/retire agents, health monitoring, pool scaling."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class PoolAgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    RETIRED = "retired"


@dataclass
class PoolAgent:
    name: str
    status: PoolAgentStatus = PoolAgentStatus.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    agent_id: str = field(default_factory=_short_id)


class AgentPool:
    def __init__(self, min_size: int = 1, max_size: int = 10) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self._agents: dict[str, PoolAgent] = {}

    def spawn(self, name: str) -> PoolAgent | None:
        if self.pool_size() >= self.max_size:
            return None
        agent = PoolAgent(name=name)
        self._agents[agent.agent_id] = agent
        return agent

    def retire(self, agent_id: str) -> bool:
        if agent_id not in self._agents:
            return False
        agent = self._agents[agent_id]
        agent.status = PoolAgentStatus.DRAINING
        agent.status = PoolAgentStatus.RETIRED
        return True

    def active_agents(self) -> list[PoolAgent]:
        return [
            a for a in self._agents.values()
            if a.status in (PoolAgentStatus.IDLE, PoolAgentStatus.BUSY)
        ]

    def assign_task(self, agent_id: str) -> bool:
        if agent_id not in self._agents:
            return False
        agent = self._agents[agent_id]
        if agent.status != PoolAgentStatus.IDLE:
            return False
        agent.status = PoolAgentStatus.BUSY
        return True

    def complete_task(self, agent_id: str, success: bool = True) -> bool:
        if agent_id not in self._agents:
            return False
        agent = self._agents[agent_id]
        agent.status = PoolAgentStatus.IDLE
        if success:
            agent.tasks_completed += 1
        else:
            agent.tasks_failed += 1
        return True

    def scale_to(self, n: int) -> int:
        target = max(self.min_size, min(self.max_size, n))
        current = self.pool_size()

        if current < target:
            for i in range(target - current):
                self.spawn(f"agent-scaled-{i}")
        elif current > target:
            # retire idle agents first, then busy
            to_retire = current - target
            active = self.active_agents()
            idle = [a for a in active if a.status == PoolAgentStatus.IDLE]
            for agent in idle[:to_retire]:
                self.retire(agent.agent_id)
                to_retire -= 1
                if to_retire == 0:
                    break
            if to_retire > 0:
                busy = [a for a in active if a.status == PoolAgentStatus.BUSY]
                for agent in busy[:to_retire]:
                    self.retire(agent.agent_id)

        return self.pool_size()

    def pool_size(self) -> int:
        return sum(1 for a in self._agents.values() if a.status != PoolAgentStatus.RETIRED)

    def utilization(self) -> float:
        active = self.active_agents()
        if not active:
            return 0.0
        busy = sum(1 for a in active if a.status == PoolAgentStatus.BUSY)
        return busy / len(active)


AGENT_POOL = AgentPool()
