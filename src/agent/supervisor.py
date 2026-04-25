"""Agent supervisor for lifecycle and health management.

Tracks state transitions, enforces restart budgets, and surfaces crashes.
Fail closed: unhealthy agents are marked crashed, not silently restarted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    CRASHED = "crashed"
    TERMINATED = "terminated"


@dataclass
class AgentRecord:
    agent_id: str
    state: AgentState = AgentState.IDLE
    start_count: int = 0
    crash_count: int = 0
    last_start: float | None = None
    last_crash: float | None = None
    task_count: int = 0


@dataclass
class SupervisorConfig:
    max_restarts: int = 3
    restart_window_seconds: float = 60.0
    health_check_interval_seconds: float = 30.0


class AgentSupervisor:
    """Supervisor that tracks agent state without spawning OS processes."""

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        self._config = config or SupervisorConfig()
        self._agents: dict[str, AgentRecord] = {}

    def register(self, agent_id: str) -> AgentRecord:
        """Register a new agent as idle."""
        record = AgentRecord(agent_id=agent_id)
        self._agents[agent_id] = record
        return record

    def start(self, agent_id: str) -> AgentRecord:
        """Transition agent to RUNNING."""
        record = self._agents.get(agent_id)
        if record is None:
            record = self.register(agent_id)
        if record.state == AgentState.TERMINATED:
            raise RuntimeError(f"Agent {agent_id} is terminated and cannot be restarted")
        record.state = AgentState.RUNNING
        record.start_count += 1
        record.last_start = time.monotonic()
        return record

    def stop(self, agent_id: str) -> AgentRecord:
        """Transition agent to IDLE."""
        record = self._agents.get(agent_id)
        if record is None:
            raise KeyError(f"Agent {agent_id} not registered")
        record.state = AgentState.IDLE
        return record

    def terminate(self, agent_id: str) -> AgentRecord:
        """Permanently terminate an agent."""
        record = self._agents.get(agent_id)
        if record is None:
            raise KeyError(f"Agent {agent_id} not registered")
        record.state = AgentState.TERMINATED
        return record

    def crash(self, agent_id: str, reason: str = "") -> AgentRecord:
        """Record a crash and transition to CRASHED if restart budget exhausted."""
        record = self._agents.get(agent_id)
        if record is None:
            raise KeyError(f"Agent {agent_id} not registered")
        record.crash_count += 1
        record.last_crash = time.monotonic()
        if not self._can_restart(record):
            record.state = AgentState.CRASHED
        else:
            record.state = AgentState.IDLE
        return record

    def _can_restart(self, record: AgentRecord) -> bool:
        if record.crash_count > self._config.max_restarts:
            return False
        if record.last_start is not None and record.last_crash is not None:
            elapsed = record.last_crash - record.last_start
            if elapsed < self._config.restart_window_seconds:
                return False
        return True

    def health(self, agent_id: str) -> dict[str, Any]:
        """Return a health snapshot for *agent_id*."""
        record = self._agents.get(agent_id)
        if record is None:
            return {"status": "unknown", "registered": False}
        return {
            "status": record.state.value,
            "registered": True,
            "start_count": record.start_count,
            "crash_count": record.crash_count,
            "task_count": record.task_count,
        }

    def increment_task(self, agent_id: str) -> None:
        record = self._agents.get(agent_id)
        if record is not None:
            record.task_count += 1

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())


# Module-level registry
SUPERVISOR_REGISTRY: dict[str, AgentSupervisor] = {}
DEFAULT_SUPERVISOR = AgentSupervisor()
SUPERVISOR_REGISTRY["default"] = DEFAULT_SUPERVISOR
