"""
execution_monitor.py – Monitors agent execution for anomalies and budget exhaustion.
Stdlib-only.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    event_type: str
    payload: dict
    timestamp: float


@dataclass(frozen=True)
class BudgetConfig:
    max_steps: int = 50
    max_time_s: float = 300.0
    max_cost: float = 1.0


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class ExecutionMonitor:
    """Records execution events and checks budget constraints."""

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self._config: BudgetConfig = config if config is not None else BudgetConfig()
        self._events: list[ExecutionEvent] = []

    # --- recording --------------------------------------------------------

    def record(self, event_type: str, payload: dict) -> ExecutionEvent:
        event = ExecutionEvent(
            event_id=uuid.uuid4().hex[:8],
            event_type=event_type,
            payload=payload,
            timestamp=time.monotonic(),
        )
        self._events.append(event)
        return event

    # --- metrics ----------------------------------------------------------

    def step_count(self) -> int:
        return len(self._events)

    def elapsed_s(self) -> float:
        if not self._events:
            return 0.0
        return time.monotonic() - self._events[0].timestamp

    def total_cost(self) -> float:
        return sum(e.payload.get("cost", 0.0) for e in self._events)

    # --- budget -----------------------------------------------------------

    def budget_status(self) -> dict:
        steps_ok = self.step_count() <= self._config.max_steps
        time_ok = self.elapsed_s() <= self._config.max_time_s
        cost_ok = self.total_cost() <= self._config.max_cost
        return {
            "steps_ok": steps_ok,
            "time_ok": time_ok,
            "cost_ok": cost_ok,
            "all_ok": steps_ok and time_ok and cost_ok,
        }

    def is_over_budget(self) -> bool:
        return not self.budget_status()["all_ok"]

    # --- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        self._events.clear()

    def export_log(self) -> list[dict]:
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "payload": e.payload,
                "timestamp": e.timestamp,
            }
            for e in self._events
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXECUTION_MONITOR_REGISTRY: dict[str, type] = {"default": ExecutionMonitor}
