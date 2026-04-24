"""Tests for src/agent/workflow_graph.py"""

from __future__ import annotations

import time

import pytest

from src.agent.workflow_graph import (
    AGENT_REGISTRY,
    NodeStatus,
    WorkflowGraph,
    WorkflowNode,
    WorkflowResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fn(return_value):
    def fn(state):
        return return_value
    return fn


def make_error_fn(msg: str):
    def fn(state):
        raise RuntimeError(msg)
    return fn


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_workflow_graph():
    assert "workflow_graph" in AGENT_REGISTRY
    assert AGENT_REGISTRY["workflow_graph"] is WorkflowGraph


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------

def test_add_node_basic():
    g = WorkflowGraph()
    g.add_node("a", make_fn(1))
    assert "a" in g._nodes


def test_add_node_duplicate_raises():
    g = WorkflowGraph()
    g.add_node("a", make_fn(1))
    with pytest.raises(ValueError, match="already registered"):
        g.add_node("a", make_fn(2))


def test_add_node_with_deps():
    g = WorkflowGraph()
    g.add_node("a", make_fn(1))
    g.add_node("b", make_fn(2), deps=["a"])
    assert g._nodes["b"].deps == ["a"]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_no_cycle():
    g = WorkflowGraph()
    g.add_node("a", make_fn(1))
    g.add_node("b", make_fn(2), deps=["a"])
    g.add_node("c", make_fn(3), deps=["b"])
    errors = g.validate()
    assert errors == []


def test_validate_detects_cycle():
    g = WorkflowGraph()
    g.add_node("a", make_fn(1), deps=["b"])
    g.add_node("b", make_fn(2), deps=["a"])
    errors = g.validate()
    assert len(errors) > 0
    assert any("Cycle" in e for e in errors)


def test_validate_undefined_dep():
    g = WorkflowGraph()
    g.add_node("a", make_fn(1), deps=["missing"])
    errors = g.validate()
    assert any("undefined" in e for e in errors)


# ---------------------------------------------------------------------------
# _topological_sort
# ---------------------------------------------------------------------------

def test_topo_sort_respects_order():
    g = WorkflowGraph()
    g.add_node("a", make_fn(1))
    g.add_node("b", make_fn(2), deps=["a"])
    g.add_node("c", make_fn(3), deps=["b"])
    order = g._topological_sort()
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


# ---------------------------------------------------------------------------
# run_sequential
# ---------------------------------------------------------------------------

def test_run_sequential_basic():
    g = WorkflowGraph()
    g.add_node("a", make_fn({"x": 1}))
    g.add_node("b", make_fn({"y": 2}), deps=["a"])
    results = g.run_sequential({})
    assert results["a"].error is None
    assert results["b"].error is None


def test_run_sequential_state_propagation():
    """Node b receives state updated by node a."""
    received = {}

    def fn_b(state):
        received.update(state)
        return {}

    g = WorkflowGraph()
    g.add_node("a", make_fn({"key": "value"}))
    g.add_node("b", fn_b, deps=["a"])
    g.run_sequential({})
    assert received.get("key") == "value"


def test_run_sequential_captures_exception():
    g = WorkflowGraph()
    g.add_node("boom", make_error_fn("kaboom"))
    results = g.run_sequential({})
    assert results["boom"].error == "kaboom"
    assert results["boom"].output is None


def test_run_sequential_duration_recorded():
    g = WorkflowGraph()
    g.add_node("a", make_fn(None))
    results = g.run_sequential({})
    assert results["a"].duration_ms >= 0.0


# ---------------------------------------------------------------------------
# run_parallel
# ---------------------------------------------------------------------------

def test_run_parallel_basic():
    g = WorkflowGraph()
    g.add_node("a", make_fn({"pa": 1}))
    g.add_node("b", make_fn({"pb": 2}))
    results = g.run_parallel({})
    assert results["a"].error is None
    assert results["b"].error is None


def test_run_parallel_respects_deps():
    """Node c must run after a and b."""
    order: list[str] = []

    def fn(name):
        def inner(state):
            order.append(name)
        return inner

    g = WorkflowGraph()
    g.add_node("a", fn("a"))
    g.add_node("b", fn("b"))
    g.add_node("c", fn("c"), deps=["a", "b"])
    g.run_parallel({})
    assert order.index("c") > order.index("a")
    assert order.index("c") > order.index("b")


def test_run_parallel_captures_exception():
    g = WorkflowGraph()
    g.add_node("bad", make_error_fn("parallel-boom"))
    results = g.run_parallel({})
    assert results["bad"].error == "parallel-boom"


# ---------------------------------------------------------------------------
# node_merge
# ---------------------------------------------------------------------------

def test_node_merge_combines_dicts():
    results = {
        "a": WorkflowResult("a", {"x": 1}, None, 0.0),
        "b": WorkflowResult("b", {"y": 2}, None, 0.0),
    }
    g = WorkflowGraph()
    merged = g.node_merge(results)
    assert merged == {"x": 1, "y": 2}


def test_node_merge_skips_errors():
    results = {
        "a": WorkflowResult("a", {"x": 1}, None, 0.0),
        "b": WorkflowResult("b", None, "oops", 0.0),
    }
    g = WorkflowGraph()
    merged = g.node_merge(results)
    assert "x" in merged
    assert len(merged) == 1


def test_node_merge_scalar_output():
    results = {"a": WorkflowResult("a", 42, None, 0.0)}
    g = WorkflowGraph()
    merged = g.node_merge(results)
    assert merged["a"] == 42


def test_node_merge_empty():
    g = WorkflowGraph()
    assert g.node_merge({}) == {}
