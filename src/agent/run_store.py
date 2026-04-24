from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_VALID_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING},
    RunStatus.RUNNING: {RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.CANCELLED},
    RunStatus.FAILED: {RunStatus.RUNNING},
    RunStatus.COMPLETED: set(),
    RunStatus.CANCELLED: set(),
}


@dataclass
class AgentRunState:
    run_id: str
    agent_id: str
    status: RunStatus
    retry_budget: int
    max_retries: int
    checkpoint_log: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: Optional[str] = None


class AgentRunStore:
    """In-memory durable agent run store with lifecycle FSM."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRunState] = {}

    def create(self, agent_id: str, max_retries: int = 3) -> AgentRunState:
        run_id = str(uuid.uuid4())
        now = time.time()
        state = AgentRunState(
            run_id=run_id,
            agent_id=agent_id,
            status=RunStatus.PENDING,
            retry_budget=max_retries,
            max_retries=max_retries,
            created_at=now,
            updated_at=now,
        )
        self._runs[run_id] = state
        return state

    def get(self, run_id: str) -> Optional[AgentRunState]:
        return self._runs.get(run_id)

    def transition(self, run_id: str, new_status: RunStatus) -> AgentRunState:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"Run {run_id!r} not found")

        allowed = _VALID_TRANSITIONS[state.status]

        if new_status == RunStatus.RUNNING and state.status == RunStatus.FAILED:
            if state.retry_budget <= 0:
                raise ValueError(
                    f"Cannot retry run {run_id!r}: retry_budget exhausted"
                )
            state.retry_budget -= 1
        elif new_status not in allowed:
            raise ValueError(
                f"Invalid transition {state.status!r} -> {new_status!r} for run {run_id!r}"
            )

        state.status = new_status
        state.updated_at = time.time()
        return state

    def checkpoint(self, run_id: str, step: int, state: dict) -> None:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"Run {run_id!r} not found")
        run.checkpoint_log.append({"step": step, "state": state, "ts": time.time()})
        run.updated_at = time.time()

    def retry(self, run_id: str) -> AgentRunState:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"Run {run_id!r} not found")
        if state.status != RunStatus.FAILED:
            raise ValueError(
                f"retry() requires FAILED status, got {state.status!r}"
            )
        if state.retry_budget <= 0:
            raise RuntimeError(
                f"Run {run_id!r} has no remaining retry budget"
            )
        return self.transition(run_id, RunStatus.RUNNING)

    def list_by_status(self, status: RunStatus) -> list[AgentRunState]:
        return [r for r in self._runs.values() if r.status == status]

    def list_all(self) -> list[AgentRunState]:
        return list(self._runs.values())


AGENT_RUN_STORE = AgentRunStore()
