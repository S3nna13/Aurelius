"""Tests for workflow scheduler."""
from __future__ import annotations

import time

import pytest

from src.workflow.scheduler import WorkflowScheduler, WorkflowSchedule


class TestWorkflowScheduler:
    def test_schedule_due(self):
        s = WorkflowSchedule(name="t", interval_seconds=0, handler=lambda: None)
        assert s.due(time.monotonic())

    def test_schedule_not_due(self):
        s = WorkflowSchedule(name="t", interval_seconds=100, handler=lambda: None)
        s._last_run = time.monotonic()
        assert not s.due(time.monotonic())

    def test_tick_runs_due(self):
        calls = []
        sched = WorkflowSchedule(name="t", interval_seconds=0, handler=lambda: calls.append(1))
        scheduler = WorkflowScheduler(schedules=[sched])
        scheduler.tick()
        assert len(calls) == 1

    def test_add_and_remove(self):
        s = WorkflowSchedule(name="t", interval_seconds=0, handler=lambda: None)
        scheduler = WorkflowScheduler()
        scheduler.add(s)
        assert len(scheduler.schedules) == 1
        scheduler.remove("t")
        assert len(scheduler.schedules) == 0