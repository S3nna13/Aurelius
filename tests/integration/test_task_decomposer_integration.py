"""Integration tests for the task decomposer via the public agent surface."""

from __future__ import annotations

import json

import pytest


def test_task_decomposer_exposed_on_agent_surface():
    import src.agent as agent

    assert hasattr(agent, "TaskDecomposer")
    assert hasattr(agent, "TaskDAG")
    assert hasattr(agent, "SubTask")
    assert hasattr(agent, "TaskDecompositionError")
    assert "task_decompose" in agent.AGENT_LOOP_REGISTRY
    assert agent.AGENT_LOOP_REGISTRY["task_decompose"] is agent.TaskDecomposer


def test_prior_agent_entries_intact():
    import src.agent as agent

    # Parsers and prior loops still registered.
    assert "xml" in agent.TOOL_CALL_PARSER_REGISTRY
    assert "json" in agent.TOOL_CALL_PARSER_REGISTRY
    for key in ("react", "safe_dispatch", "beam_plan"):
        assert key in agent.AGENT_LOOP_REGISTRY, key
    # Prior names still exported.
    for name in (
        "ReActLoop",
        "ToolRegistryDispatcher",
        "BeamPlanner",
        "RecoveringDispatcher",
        "RepoContextPacker",
        "UnifiedDiffGenerator",
        "ShellCommandPlanner",
        "CodeExecutionSandbox",
    ):
        assert name in agent.__all__, name


def test_end_to_end_decompose():
    from src.agent import TaskDecomposer

    payload = [
        {"id": "plan", "description": "plan it", "depends_on": []},
        {"id": "code", "description": "write code", "depends_on": ["plan"],
         "estimated_complexity": "high", "tool_hints": ["editor"]},
        {"id": "test", "description": "run tests", "depends_on": ["plan"]},
        {"id": "review", "description": "review", "depends_on": ["code", "test"]},
    ]
    text = json.dumps(payload)

    dec = TaskDecomposer(lambda task: text)
    dag = dec.decompose("ship feature X")

    order = dec.topological_sort(dag)
    groups = dec.parallelizable_groups(dag)

    assert order[0] == "plan"
    assert order[-1] == "review"
    assert groups == [["plan"], ["code", "test"], ["review"]]
    assert dag.roots == ["plan"]
    assert dag.leaves == ["review"]

    code = next(t for t in dag.tasks if t.id == "code")
    assert code.estimated_complexity == "high"
    assert code.tool_hints == ["editor"]


def test_end_to_end_cycle_rejected():
    from src.agent import TaskDecomposer, TaskDecompositionError

    payload = [
        {"id": "a", "description": "a", "depends_on": ["b"]},
        {"id": "b", "description": "b", "depends_on": ["a"]},
    ]
    dec = TaskDecomposer(lambda _t: json.dumps(payload))
    with pytest.raises(TaskDecompositionError):
        dec.decompose("x")
