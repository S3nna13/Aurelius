from __future__ import annotations

import json
import sqlite3
import time
import uuid

from src.agent.run_store import AgentRunState, RunStatus

_VALID_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING},
    RunStatus.RUNNING: {RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.CANCELLED},
    RunStatus.FAILED: {RunStatus.RUNNING},
    RunStatus.COMPLETED: set(),
    RunStatus.CANCELLED: set(),
}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    status TEXT NOT NULL,
    retry_budget INTEGER NOT NULL,
    max_retries INTEGER NOT NULL,
    checkpoint_log TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    error TEXT
)
"""


class SQLiteRunStore:
    """Durable SQLite-backed agent run store. Same interface as AgentRunStore."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def _row_to_state(self, row: sqlite3.Row) -> AgentRunState:
        return AgentRunState(
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            status=RunStatus(row["status"]),
            retry_budget=row["retry_budget"],
            max_retries=row["max_retries"],
            checkpoint_log=json.loads(row["checkpoint_log"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
        )

    def _state_to_row(self, state: AgentRunState) -> tuple:
        return (
            state.run_id,
            state.agent_id,
            state.status.value,
            state.retry_budget,
            state.max_retries,
            json.dumps(state.checkpoint_log),
            state.created_at,
            state.updated_at,
            state.error,
        )

    def create(self, agent_id: str, max_retries: int = 3) -> AgentRunState:
        now = time.time()
        state = AgentRunState(
            run_id=str(uuid.uuid4()),
            agent_id=agent_id,
            status=RunStatus.PENDING,
            retry_budget=max_retries,
            max_retries=max_retries,
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            "INSERT INTO agent_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._state_to_row(state),
        )
        self._conn.commit()
        return state

    def get(self, run_id: str) -> AgentRunState | None:
        row = self._conn.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_state(row) if row else None

    def update(self, state: AgentRunState) -> None:
        self._conn.execute(
            """UPDATE agent_runs
               SET agent_id=?, status=?, retry_budget=?, max_retries=?,
                   checkpoint_log=?, created_at=?, updated_at=?, error=?
               WHERE run_id=?""",
            (
                state.agent_id,
                state.status.value,
                state.retry_budget,
                state.max_retries,
                json.dumps(state.checkpoint_log),
                state.created_at,
                state.updated_at,
                state.error,
                state.run_id,
            ),
        )
        self._conn.commit()

    def transition(self, run_id: str, new_status: RunStatus) -> AgentRunState:
        state = self.get(run_id)
        if state is None:
            raise KeyError(f"Run {run_id!r} not found")

        allowed = _VALID_TRANSITIONS[state.status]

        if new_status == RunStatus.RUNNING and state.status == RunStatus.FAILED:
            if state.retry_budget <= 0:
                raise ValueError(f"Cannot retry run {run_id!r}: retry_budget exhausted")
            state.retry_budget -= 1
        elif new_status not in allowed:
            raise ValueError(
                f"Invalid transition {state.status!r} -> {new_status!r} for run {run_id!r}"
            )

        state.status = new_status
        state.updated_at = time.time()
        self.update(state)
        return state

    def checkpoint(self, run_id: str, step: int, state: dict) -> None:
        run = self.get(run_id)
        if run is None:
            raise KeyError(f"Run {run_id!r} not found")
        run.checkpoint_log.append({"step": step, "state": state, "ts": time.time()})
        run.updated_at = time.time()
        self.update(run)

    def retry(self, run_id: str) -> AgentRunState:
        state = self.get(run_id)
        if state is None:
            raise KeyError(f"Run {run_id!r} not found")
        if state.status != RunStatus.FAILED:
            raise ValueError(f"retry() requires FAILED status, got {state.status!r}")
        if state.retry_budget <= 0:
            raise RuntimeError(f"Run {run_id!r} has no remaining retry budget")
        return self.transition(run_id, RunStatus.RUNNING)

    def list_by_status(self, status: RunStatus) -> list[AgentRunState]:
        rows = self._conn.execute(
            "SELECT * FROM agent_runs WHERE status = ?", (status.value,)
        ).fetchall()
        return [self._row_to_state(r) for r in rows]

    def list_all(self) -> list[AgentRunState]:
        rows = self._conn.execute("SELECT * FROM agent_runs").fetchall()
        return [self._row_to_state(r) for r in rows]

    def delete(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))
        self._conn.commit()
