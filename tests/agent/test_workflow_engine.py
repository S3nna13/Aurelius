"""Unit tests for src/agent/workflow_engine.py."""

from __future__ import annotations

import pytest

from src.agent.workflow_engine import (
    DEFAULT_WORKFLOW_EXECUTOR,
    WORKFLOW_EXECUTOR_REGISTRY,
    WorkflowCheckpoint,
    WorkflowDAG,
    WorkflowError,
    WorkflowExecutor,
    WorkflowNode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> WorkflowExecutor:
    """Return a fresh executor to avoid test pollution."""
    return WorkflowExecutor()


def _fresh_dag() -> WorkflowDAG:
    """Return an empty DAG for testing."""
    return WorkflowDAG(workflow_id="test")


# ---------------------------------------------------------------------------
# 1. test_build_dag_and_validate
# ---------------------------------------------------------------------------


def test_build_dag_and_validate():
    dag = _fresh_dag()
    node_a = WorkflowNode(node_id="a", action="noop")
    node_b = WorkflowNode(node_id="b", action="noop", depends_on=["a"])
    dag.add_node(node_a)
    dag.add_node(node_b)
    errors = dag.validate()
    assert errors == []


# ---------------------------------------------------------------------------
# 2. test_detect_cycle_in_dag
# ---------------------------------------------------------------------------


def test_detect_cycle_in_dag():
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="noop"))
    dag.add_node(WorkflowNode(node_id="b", action="noop", depends_on=["a"]))
    dag.add_node(WorkflowNode(node_id="c", action="noop", depends_on=["b"]))
    # Introduce a cycle by mutating the already-registered node.
    dag.nodes["a"].depends_on.append("c")
    errors = dag.validate()
    assert any("cycle" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 3. test_topological_order_correct
# ---------------------------------------------------------------------------


def test_topological_order_correct():
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="noop"))
    dag.add_node(WorkflowNode(node_id="b", action="noop", depends_on=["a"]))
    dag.add_node(WorkflowNode(node_id="c", action="noop", depends_on=["b"]))
    order = dag.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")


# ---------------------------------------------------------------------------
# 4. test_execute_simple_sequential_workflow
# ---------------------------------------------------------------------------


def test_execute_simple_sequential_workflow():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "hello"}))
    dag.add_node(WorkflowNode(node_id="b", action="bash", payload={"cmd": "ls"}))
    result = exe.execute(dag)
    assert result["a"] == "hello"
    assert result["b"] == "executed: ls"


# ---------------------------------------------------------------------------
# 5. test_execute_workflow_with_dependencies
# ---------------------------------------------------------------------------


def test_execute_workflow_with_dependencies():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "step1"}))
    dag.add_node(
        WorkflowNode(
            node_id="b",
            action="prompt",
            payload={"text": "step2"},
            depends_on=["a"],
        )
    )
    result = exe.execute(dag)
    assert result["a"] == "step1"
    assert result["b"] == "step2"


# ---------------------------------------------------------------------------
# 6. test_execute_workflow_with_human_gate
# ---------------------------------------------------------------------------


def test_execute_workflow_with_human_gate():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(
        WorkflowNode(
            node_id="gate",
            action="human_gate",
            payload={"question": "Approve?"},
        )
    )
    result = exe.execute(dag)
    assert result["gate"] == "Approve? (awaiting approval)"


# ---------------------------------------------------------------------------
# 7. test_execute_loop_node_stops_at_max_iterations
# ---------------------------------------------------------------------------


def test_execute_loop_node_stops_at_max_iterations():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(
        WorkflowNode(
            node_id="loop",
            action="loop",
            payload={"sub_node": "body"},
            max_iterations=3,
        )
    )
    result = exe.execute(dag)
    assert result["loop"]["iterations"] == 3


# ---------------------------------------------------------------------------
# 8. test_fail_fast_on_node_failure
# ---------------------------------------------------------------------------


def test_fail_fast_on_node_failure():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="ok", action="noop"))
    dag.add_node(
        WorkflowNode(
            node_id="fail",
            action="noop",
            payload={"fail": True},
            depends_on=["ok"],
        )
    )
    dag.add_node(WorkflowNode(node_id="never", action="noop", depends_on=["fail"]))
    with pytest.raises(WorkflowError, match="failed"):
        exe.execute(dag)


# ---------------------------------------------------------------------------
# 9. test_checkpoint_creation
# ---------------------------------------------------------------------------


def test_checkpoint_creation():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "hello"}))
    dag.add_node(WorkflowNode(node_id="b", action="bash", payload={"cmd": "ls"}))
    checkpoints = exe.execute_with_checkpoints(dag)
    assert len(checkpoints) == 2
    assert checkpoints[0].node_id == "a"
    assert checkpoints[0].status == "success"
    assert checkpoints[1].node_id == "b"
    assert checkpoints[1].status == "success"


# ---------------------------------------------------------------------------
# 10. test_resume_from_checkpoint_skips_completed_nodes
# ---------------------------------------------------------------------------


def test_resume_from_checkpoint_skips_completed_nodes():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "hello"}))
    dag.add_node(WorkflowNode(node_id="b", action="bash", payload={"cmd": "ls"}))
    checkpoints = exe.execute_with_checkpoints(dag)
    result = exe.resume_from_checkpoint(dag, checkpoints)
    assert result["a"] == "hello"
    assert result["b"] == "executed: ls"


# ---------------------------------------------------------------------------
# 11. test_resume_re_runs_failed_nodes
# ---------------------------------------------------------------------------


def test_resume_re_runs_failed_nodes():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "hello"}))
    dag.add_node(WorkflowNode(node_id="b", action="noop", payload={"fail": True}))
    checkpoints = exe.execute_with_checkpoints(dag)
    assert checkpoints[-1].status == "failure"

    # Build a fixed DAG and resume
    dag2 = _fresh_dag()
    dag2.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "hello"}))
    dag2.add_node(WorkflowNode(node_id="b", action="bash", payload={"cmd": "ls"}))
    result = exe.resume_from_checkpoint(dag2, checkpoints)
    assert result["a"] == "hello"
    assert result["b"] == "executed: ls"


# ---------------------------------------------------------------------------
# 12. test_empty_dag_returns_empty_result
# ---------------------------------------------------------------------------


def test_empty_dag_returns_empty_result():
    exe = _fresh()
    dag = _fresh_dag()
    result = exe.execute(dag)
    assert result == {}


# ---------------------------------------------------------------------------
# 13. test_missing_dependency_raises_workflow_error
# ---------------------------------------------------------------------------


def test_missing_dependency_raises_in_validate():
    dag = _fresh_dag()
    node = WorkflowNode(node_id="a", action="noop", depends_on=["missing"])
    dag.add_node(node)
    errors = dag.validate()
    assert any("missing" in err for err in errors)


# ---------------------------------------------------------------------------
# 14. test_duplicate_node_id_raises_workflow_error
# ---------------------------------------------------------------------------


def test_duplicate_node_id_raises_workflow_error():
    dag = _fresh_dag()
    node = WorkflowNode(node_id="a", action="noop")
    dag.add_node(node)
    with pytest.raises(WorkflowError, match="Duplicate node_id"):
        dag.add_node(node)


# ---------------------------------------------------------------------------
# 15. test_registry_singleton
# ---------------------------------------------------------------------------


def test_registry_singleton():
    assert "default" in WORKFLOW_EXECUTOR_REGISTRY
    assert WORKFLOW_EXECUTOR_REGISTRY["default"] is DEFAULT_WORKFLOW_EXECUTOR


# ---------------------------------------------------------------------------
# 16. test_context_passed_through_execution
# ---------------------------------------------------------------------------


def test_context_passed_through_execution():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="noop"))
    result = exe.execute(dag, context={"key": "value"})
    assert result["key"] == "value"


# ---------------------------------------------------------------------------
# 17. test_to_dict_roundtrip
# ---------------------------------------------------------------------------


def test_to_dict_roundtrip():
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "hello"}))
    d = dag.to_dict()
    assert d["workflow_id"] == "test"
    assert d["nodes"]["a"]["action"] == "prompt"
    assert d["nodes"]["a"]["payload"]["text"] == "hello"


# ---------------------------------------------------------------------------
# 18. test_topological_order_raises_on_cycle
# ---------------------------------------------------------------------------


def test_topological_order_raises_on_cycle():
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="noop"))
    dag.add_node(WorkflowNode(node_id="b", action="noop", depends_on=["a"]))
    dag.add_node(WorkflowNode(node_id="c", action="noop", depends_on=["b"]))
    # Introduce a cycle by mutating the already-registered node.
    dag.nodes["a"].depends_on.append("c")
    with pytest.raises(WorkflowError, match="Invalid DAG"):
        dag.topological_order()


# ---------------------------------------------------------------------------
# 19. test_loop_node_zero_iterations_when_done
# ---------------------------------------------------------------------------


def test_loop_node_zero_iterations_when_done():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(
        WorkflowNode(
            node_id="loop",
            action="loop",
            payload={"sub_node": "body", "condition": "done"},
            max_iterations=5,
        )
    )
    result = exe.execute(dag)
    assert result["loop"]["iterations"] == 0


# ---------------------------------------------------------------------------
# 20. test_checkpoint_state_captured
# ---------------------------------------------------------------------------


def test_checkpoint_state_captured():
    exe = _fresh()
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="prompt", payload={"text": "hello"}))
    dag.add_node(
        WorkflowNode(
            node_id="b",
            action="prompt",
            payload={"text": "world"},
            depends_on=["a"],
        )
    )
    checkpoints = exe.execute_with_checkpoints(dag)
    assert checkpoints[1].state["a"] == "hello"
    assert checkpoints[1].state["b"] == "world"


# ---------------------------------------------------------------------------
# 21. test_workflow_error_is_exception
# ---------------------------------------------------------------------------


def test_workflow_error_is_exception():
    assert issubclass(WorkflowError, Exception)


# ---------------------------------------------------------------------------
# 22. test_dag_multiple_dependencies
# ---------------------------------------------------------------------------


def test_dag_multiple_dependencies():
    dag = _fresh_dag()
    dag.add_node(WorkflowNode(node_id="a", action="noop"))
    dag.add_node(WorkflowNode(node_id="b", action="noop"))
    dag.add_node(WorkflowNode(node_id="c", action="noop", depends_on=["a", "b"]))
    order = dag.topological_order()
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("c")
