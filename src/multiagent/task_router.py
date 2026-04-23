"""Task router: capability-based routing, load balancing, priority queuing."""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class RoutingStrategy(str, Enum):
    CAPABILITY = "capability"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY = "priority"


@dataclass
class _AgentEntry:
    agent_id: str
    capabilities: list[str]
    load: int = 0
    priority: int = 0


class TaskRouter:
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.CAPABILITY) -> None:
        self.strategy = strategy
        self._agents: dict[str, _AgentEntry] = {}
        self._rr_index: int = 0

    def register(
        self,
        agent_id: str,
        capabilities: list[str],
        load: int = 0,
        priority: int = 0,
    ) -> None:
        self._agents[agent_id] = _AgentEntry(
            agent_id=agent_id,
            capabilities=capabilities,
            load=load,
            priority=priority,
        )

    def route(self, task: str, required_capability: str | None = None) -> str | None:
        if not self._agents:
            return None

        agents = list(self._agents.values())

        if self.strategy == RoutingStrategy.CAPABILITY:
            if required_capability is not None:
                candidates = [a for a in agents if required_capability in a.capabilities]
                if not candidates:
                    return None
            else:
                candidates = agents
            return min(candidates, key=lambda a: a.load).agent_id

        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            keys = list(self._agents.keys())
            agent_id = keys[self._rr_index % len(keys)]
            self._rr_index += 1
            return agent_id

        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return min(agents, key=lambda a: a.load).agent_id

        elif self.strategy == RoutingStrategy.PRIORITY:
            if required_capability is not None:
                candidates = [a for a in agents if required_capability in a.capabilities]
                if not candidates:
                    return None
            else:
                candidates = agents
            return max(candidates, key=lambda a: a.priority).agent_id

        return None

    def update_load(self, agent_id: str, delta: int = 1) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].load += delta

    def agents_by_capability(self, cap: str) -> list[str]:
        return [a.agent_id for a in self._agents.values() if cap in a.capabilities]

    def registered_agents(self) -> list[str]:
        return list(self._agents.keys())


TASK_ROUTER = TaskRouter()
