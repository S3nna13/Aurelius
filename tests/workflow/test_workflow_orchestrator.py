"""Tests for workflow_orchestrator — dependency-aware step execution."""

from __future__ import annotations

import pytest

from src.workflow.workflow_orchestrator import (
    WorkflowOrchestrator,
    WORKFLOW_ORCHESTRATOR_REGISTRY,
)


class TestWorkflowOrchestrator:
    def test_linear_workflow(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda: 1)
        o.register_step("b", lambda a: a + 1, depends_on=["a"])
        o.register_step("c", lambda b: b * 2, depends_on=["b"])
        results = o.execute()
        assert results == {"a": 1, "b": 2, "c": 4}

    def test_branched_workflow(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda: 10)
        o.register_step("b", lambda a: a + 1, depends_on=["a"])
        o.register_step("c", lambda a: a + 2, depends_on=["a"])
        o.register_step("d", lambda b, c: b + c, depends_on=["b", "c"])
        results = o.execute()
        assert results["a"] == 10
        assert results["b"] == 11
        assert results["c"] == 12
        assert results["d"] == 23

    def test_circular_dependency_rejected(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda b: 1, depends_on=["b"])
        o.register_step("b", lambda a: 2, depends_on=["a"])
        with pytest.raises(ValueError, match="circular dependency"):
            o.execute()

    def test_missing_dependency_rejected(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda b: 1, depends_on=["b"])
        with pytest.raises(ValueError, match="missing dependency"):
            o.execute()

    def test_max_steps_enforced(self):
        o = WorkflowOrchestrator(max_steps=2)
        o.register_step("a", lambda: 1)
        o.register_step("b", lambda a: 2, depends_on=["a"])
        with pytest.raises(ValueError, match="max_steps"):
            o.register_step("c", lambda b: 3, depends_on=["b"])


class TestWorkflowOrchestratorValidation:
    def test_duplicate_step_name_rejected(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda: 1)
        with pytest.raises(ValueError, match="duplicate step name"):
            o.register_step("a", lambda: 2)

    def test_get_execution_order(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda: 1)
        o.register_step("b", lambda a: 2, depends_on=["a"])
        o.register_step("c", lambda a: 3, depends_on=["a"])
        order = o.get_execution_order()
        assert order[0] == "a"
        assert order.index("b") > order.index("a")
        assert order.index("c") > order.index("a")

    def test_reset(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda: 1)
        o.execute()
        o.reset()
        assert o._results == {}

    def test_execute_with_inputs(self):
        o = WorkflowOrchestrator()
        o.register_step("a", lambda x: x * 2)
        results = o.execute(inputs={"x": 5})
        assert results["a"] == 10

    def test_registry_has_default(self):
        assert WORKFLOW_ORCHESTRATOR_REGISTRY["default"] is WorkflowOrchestrator
