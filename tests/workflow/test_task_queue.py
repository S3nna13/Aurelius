"""Tests for task queue."""

from __future__ import annotations

from src.workflow.task_queue import Task, TaskQueue


class TestTaskQueue:
    def test_task_without_deps_is_ready(self):
        tq = TaskQueue()
        tq.add(Task(id="a"))
        assert len(tq.ready()) == 1

    def test_task_with_unmet_dep_not_ready(self):
        tq = TaskQueue()
        tq.add(Task(id="a", deps=["b"]))
        assert len(tq.ready()) == 0

    def test_task_becomes_ready_after_dep_completed(self):
        tq = TaskQueue()
        tq.add(Task(id="a", deps=["b"]))
        tq.add(Task(id="b"))
        tq.complete("b")
        assert len(tq.ready()) == 1

    def test_pending_count(self):
        tq = TaskQueue()
        tq.add(Task(id="a"))
        tq.add(Task(id="b"))
        assert tq.pending_count() == 2

    def test_complete_removes_from_pending(self):
        tq = TaskQueue()
        tq.add(Task(id="a"))
        tq.complete("a")
        assert tq.pending_count() == 0
