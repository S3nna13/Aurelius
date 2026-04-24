import pytest

from src.workflow.dag_executor import (
    DAG_EXECUTOR_REGISTRY,
    DAGExecutor,
    DAGNode,
    ExecutionResult,
    NodeStatus,
)


def test_registry_has_default():
    assert DAG_EXECUTOR_REGISTRY["default"] is DAGExecutor


def test_node_status_values():
    assert NodeStatus.PENDING.value == "pending"
    assert NodeStatus.RUNNING.value == "running"
    assert NodeStatus.COMPLETED.value == "completed"
    assert NodeStatus.FAILED.value == "failed"
    assert NodeStatus.SKIPPED.value == "skipped"


def test_execution_result_is_frozen():
    r = ExecutionResult(node_id="a", status=NodeStatus.COMPLETED, result=1, duration_ms=0.0)
    with pytest.raises(Exception):
        r.result = 2  # type: ignore[misc]


def test_dag_node_defaults():
    node = DAGNode(node_id="x", fn=lambda: 1)
    assert node.dependencies == []
    assert node.status == NodeStatus.PENDING
    assert node.result is None
    assert node.error == ""


def test_add_node_basic():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    assert "a" in d._nodes


def test_add_node_duplicate_raises():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    with pytest.raises(ValueError):
        d.add_node("a", lambda: 2)


def test_topological_sort_linear():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    d.add_node("b", lambda **kw: 2, dependencies=["a"])
    d.add_node("c", lambda **kw: 3, dependencies=["b"])
    order = d._topological_sort()
    assert order.index("a") < order.index("b") < order.index("c")


def test_topological_sort_diamond():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    d.add_node("b", lambda **kw: 2, dependencies=["a"])
    d.add_node("c", lambda **kw: 3, dependencies=["a"])
    d.add_node("e", lambda **kw: 4, dependencies=["b", "c"])
    order = d._topological_sort()
    assert order[0] == "a"
    assert order[-1] == "e"


def test_cycle_detection_raises():
    d = DAGExecutor()
    d.add_node("a", lambda **kw: 1, dependencies=["b"])
    d.add_node("b", lambda **kw: 2, dependencies=["a"])
    with pytest.raises(ValueError):
        d._topological_sort()


def test_unknown_dependency_raises():
    d = DAGExecutor()
    d.add_node("a", lambda **kw: 1, dependencies=["missing"])
    with pytest.raises(ValueError):
        d._topological_sort()


def test_execute_single_node():
    d = DAGExecutor()
    d.add_node("a", lambda: 42)
    results = d.execute()
    assert results["a"].status == NodeStatus.COMPLETED
    assert results["a"].result == 42


def test_execute_passes_dep_results_as_kwargs():
    d = DAGExecutor()
    d.add_node("a", lambda: 10)
    d.add_node("b", lambda **kw: kw["a"] + 5, dependencies=["a"])
    results = d.execute()
    assert results["b"].result == 15


def test_execute_diamond_order():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    d.add_node("b", lambda **kw: kw["a"] + 1, dependencies=["a"])
    d.add_node("c", lambda **kw: kw["a"] + 2, dependencies=["a"])
    d.add_node("e", lambda **kw: kw["b"] + kw["c"], dependencies=["b", "c"])
    results = d.execute()
    assert results["e"].result == (1 + 1) + (1 + 2)


def test_failed_node_marks_failed():
    def boom():
        raise RuntimeError("bad")

    d = DAGExecutor()
    d.add_node("a", boom)
    results = d.execute()
    assert results["a"].status == NodeStatus.FAILED
    assert "bad" in results["a"].error


def test_failure_propagates_skip_to_dependent():
    def boom():
        raise RuntimeError("x")

    d = DAGExecutor()
    d.add_node("a", boom)
    d.add_node("b", lambda **kw: 1, dependencies=["a"])
    results = d.execute()
    assert results["a"].status == NodeStatus.FAILED
    assert results["b"].status == NodeStatus.SKIPPED


def test_independent_node_runs_despite_failure():
    def boom():
        raise RuntimeError("x")

    d = DAGExecutor()
    d.add_node("a", boom)
    d.add_node("b", lambda: 99)
    results = d.execute()
    assert results["a"].status == NodeStatus.FAILED
    assert results["b"].status == NodeStatus.COMPLETED
    assert results["b"].result == 99


def test_skip_propagation_chain():
    def boom():
        raise RuntimeError("x")

    d = DAGExecutor()
    d.add_node("a", boom)
    d.add_node("b", lambda **kw: 1, dependencies=["a"])
    d.add_node("c", lambda **kw: 2, dependencies=["b"])
    results = d.execute()
    assert results["c"].status == NodeStatus.SKIPPED


def test_execution_summary_counts():
    def boom():
        raise RuntimeError("x")

    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    d.add_node("b", boom)
    d.add_node("c", lambda **kw: 2, dependencies=["b"])
    results = d.execute()
    summary = d.execution_summary(results)
    assert summary["total"] == 3
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["skipped"] == 1
    assert summary["total_duration_ms"] >= 0.0


def test_execution_summary_all_completed():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    d.add_node("b", lambda: 2)
    results = d.execute()
    summary = d.execution_summary(results)
    assert summary["completed"] == 2
    assert summary["failed"] == 0


def test_reset_restores_pending():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    d.execute()
    assert d._nodes["a"].status == NodeStatus.COMPLETED
    d.reset()
    assert d._nodes["a"].status == NodeStatus.PENDING
    assert d._nodes["a"].result is None
    assert d._nodes["a"].error == ""


def test_reset_clears_error():
    def boom():
        raise RuntimeError("x")

    d = DAGExecutor()
    d.add_node("a", boom)
    d.execute()
    assert d._nodes["a"].error
    d.reset()
    assert d._nodes["a"].error == ""


def test_execute_with_inputs_for_root():
    d = DAGExecutor()
    d.add_node("a", lambda **kw: kw["x"] * 2)
    results = d.execute(inputs={"x": 5})
    assert results["a"].result == 10


def test_empty_executor_returns_empty():
    d = DAGExecutor()
    assert d.execute() == {}


def test_execution_result_duration_non_negative():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    results = d.execute()
    assert results["a"].duration_ms >= 0.0


def test_summary_on_empty():
    d = DAGExecutor()
    summary = d.execution_summary({})
    assert summary["total"] == 0
    assert summary["completed"] == 0


def test_self_loop_is_cycle():
    d = DAGExecutor()
    d.add_node("a", lambda **kw: 1, dependencies=["a"])
    with pytest.raises(ValueError):
        d._topological_sort()


def test_multiple_roots():
    d = DAGExecutor()
    d.add_node("a", lambda: 1)
    d.add_node("b", lambda: 2)
    d.add_node("c", lambda **kw: kw["a"] + kw["b"], dependencies=["a", "b"])
    results = d.execute()
    assert results["c"].result == 3


def test_re_execute_after_reset():
    counter = {"n": 0}

    def bump():
        counter["n"] += 1
        return counter["n"]

    d = DAGExecutor()
    d.add_node("a", bump)
    d.execute()
    d.reset()
    results = d.execute()
    assert results["a"].result == 2
