"""Workflow scheduler with cron-like interval support."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class WorkflowSchedule:
    name: str
    interval_seconds: float
    handler: Callable[[], None]
    _last_run: float = 0.0

    def due(self, now: float) -> bool:
        return (now - self._last_run) >= self.interval_seconds

    def run(self) -> None:
        self.handler()
        self._last_run = time.monotonic()


@dataclass
class WorkflowScheduler:
    schedules: list[WorkflowSchedule] = field(default_factory=list)

    def add(self, schedule: WorkflowSchedule) -> None:
        self.schedules.append(schedule)

    def tick(self) -> None:
        now = time.monotonic()
        for s in self.schedules:
            if s.due(now):
                s.run()

    def remove(self, name: str) -> None:
        self.schedules = [s for s in self.schedules if s.name != name]


WORKFLOW_SCHEDULER = WorkflowScheduler()
