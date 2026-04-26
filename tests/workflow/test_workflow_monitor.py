"""Tests for workflow_monitor — workflow execution monitoring."""
from __future__ import annotations

import time

from src.workflow.workflow_monitor import (
    WorkflowMonitor,
    WorkflowStatus,
    WorkflowEvent,
    EventType,
    create_event,
)


class TestWorkflowStatus:
    def test_status_transitions(self):
        s = WorkflowStatus.RUNNING
        assert s.value == "running"


class TestWorkflowEvent:
    def test_event_creation(self):
        e = WorkflowEvent(event_type=EventType.STARTED, workflow_id="wf-1", message="started")
        assert e.workflow_id == "wf-1"
        assert e.event_type == EventType.STARTED

    def test_default_timestamp(self):
        e = create_event(EventType.STARTED, "wf-1")
        assert e.timestamp > 0


class TestWorkflowMonitor:
    def test_start_and_complete(self):
        m = WorkflowMonitor()
        m.start_workflow("wf-1", "test workflow")
        assert m.get_status("wf-1") == WorkflowStatus.RUNNING
        m.complete_workflow("wf-1", "completed OK")
        assert m.get_status("wf-1") == WorkflowStatus.COMPLETED

    def test_fail_workflow(self):
        m = WorkflowMonitor()
        m.start_workflow("wf-2", "failing")
        m.fail_workflow("wf-2", "error happened")
        assert m.get_status("wf-2") == WorkflowStatus.FAILED

    def test_event_history(self):
        m = WorkflowMonitor()
        m.start_workflow("wf-3", "history test")
        m.log_event("wf-3", EventType.PROGRESS, "step 1 done")
        m.log_event("wf-3", EventType.PROGRESS, "step 2 done")
        events = m.get_events("wf-3")
        assert len(events) == 3  # start + 2 progress

    def test_unknown_workflow(self):
        m = WorkflowMonitor()
        assert m.get_status("nonexistent") is None
        assert m.get_events("nonexistent") == []

    def test_multiple_workflows(self):
        m = WorkflowMonitor()
        m.start_workflow("a", "first")
        m.start_workflow("b", "second")
        assert m.active_count() == 2
        m.complete_workflow("a", "done")
        assert m.active_count() == 1

    def test_summary(self):
        m = WorkflowMonitor()
        m.start_workflow("w1", "task 1")
        m.complete_workflow("w1", "ok")
        m.start_workflow("w2", "task 2")
        summary = m.summary()
        assert summary["total"] == 2
        assert summary["running"] == 1
        assert summary["completed"] == 1
