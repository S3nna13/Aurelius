"""Unit tests for :mod:`src.agent.task_decomposer`."""

from __future__ import annotations

import json

import pytest

from src.agent.task_decomposer import (
    SubTask,
    TaskDAG,
    TaskDecomposer,
    TaskDecompositionError,
)


def _fake(payload: list[dict]):
    text = json.dumps(payload)

    def _gen(_task: str) -> str:
        return text

    return _gen


SINGLE = [{"id": "a", "description": "do a", "depends_on": []}]

CHAIN = [
    {"id": "A", "description": "a", "depends_on": []},
    {"id": "B", "description": "b", "depends_on": ["A"]},
    {"id": "C", "description": "c", "depends_on": ["B"]},
]

FORK_JOIN = [
    {"id": "A", "description": "a", "depends_on": []},
    {"id": "B", "description": "b", "depends_on": ["A"]},
    {"id": "C", "description": "c", "depends_on": ["A"]},
    {"id": "D", "description": "d", "depends_on": ["B", "C"]},
]


def test_single_task_decomposition():
    dec = TaskDecomposer(_fake(SINGLE))
    dag = dec.decompose("x")
    assert [t.id for t in dag.tasks] == ["a"]
    assert dag.roots == ["a"]
    assert dag.leaves == ["a"]


def test_chain_linear():
    dec = TaskDecomposer(_fake(CHAIN))
    dag = dec.decompose("x")
    order = dec.topological_sort(dag)
    assert order == ["A", "B", "C"]
    assert dag.roots == ["A"]
    assert dag.leaves == ["C"]


def test_fork_join():
    dec = TaskDecomposer(_fake(FORK_JOIN))
    dag = dec.decompose("x")
    order = dec.topological_sort(dag)
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


def test_parallelizable_groups_identifies_parallel():
    dec = TaskDecomposer(_fake(FORK_JOIN))
    dag = dec.decompose("x")
    groups = dec.parallelizable_groups(dag)
    assert groups == [["A"], ["B", "C"], ["D"]]


def test_topological_sort_is_valid_ordering():
    dec = TaskDecomposer(_fake(FORK_JOIN))
    dag = dec.decompose("x")
    order = dec.topological_sort(dag)
    seen: set[str] = set()
    by_id = {t.id: t for t in dag.tasks}
    for tid in order:
        for d in by_id[tid].depends_on:
            assert d in seen, f"{tid} appeared before dep {d}"
        seen.add(tid)


def test_cycle_detection_raises():
    payload = [
        {"id": "A", "description": "a", "depends_on": ["B"]},
        {"id": "B", "description": "b", "depends_on": ["A"]},
    ]
    dec = TaskDecomposer(_fake(payload))
    with pytest.raises(TaskDecompositionError, match="cycle"):
        dec.decompose("x")


def test_missing_dependency_raises():
    payload = [{"id": "A", "description": "a", "depends_on": ["ghost"]}]
    dec = TaskDecomposer(_fake(payload))
    with pytest.raises(TaskDecompositionError, match="unknown id"):
        dec.decompose("x")


def test_self_loop_raises():
    payload = [{"id": "A", "description": "a", "depends_on": ["A"]}]
    dec = TaskDecomposer(_fake(payload))
    with pytest.raises(TaskDecompositionError, match="self-loop"):
        dec.decompose("x")


def test_max_tasks_exceeded_raises():
    payload = [{"id": f"t{i}", "description": "x", "depends_on": []} for i in range(6)]
    dec = TaskDecomposer(_fake(payload), max_tasks=5)
    with pytest.raises(TaskDecompositionError, match="max_tasks"):
        dec.decompose("x")


def test_max_depth_exceeded_raises():
    # 5-long chain violates max_depth=3
    payload = [
        {"id": "A", "description": "a", "depends_on": []},
        {"id": "B", "description": "b", "depends_on": ["A"]},
        {"id": "C", "description": "c", "depends_on": ["B"]},
        {"id": "D", "description": "d", "depends_on": ["C"]},
        {"id": "E", "description": "e", "depends_on": ["D"]},
    ]
    dec = TaskDecomposer(_fake(payload), max_depth=3)
    with pytest.raises(TaskDecompositionError, match="depth"):
        dec.decompose("x")


def test_determinism():
    dec1 = TaskDecomposer(_fake(FORK_JOIN))
    dec2 = TaskDecomposer(_fake(FORK_JOIN))
    g1 = dec1.parallelizable_groups(dec1.decompose("x"))
    g2 = dec2.parallelizable_groups(dec2.decompose("x"))
    assert g1 == g2


def test_empty_task_raises():
    dec = TaskDecomposer(_fake(SINGLE))
    with pytest.raises(TaskDecompositionError, match="non-empty"):
        dec.decompose("")
    with pytest.raises(TaskDecompositionError, match="non-empty"):
        dec.decompose("   ")


def test_malformed_json_raises():
    def _gen(_t: str) -> str:
        return "{not json"

    dec = TaskDecomposer(_gen)
    with pytest.raises(TaskDecompositionError, match="malformed JSON"):
        dec.decompose("x")


def test_non_array_top_level_raises():
    def _gen(_t: str) -> str:
        return json.dumps({"id": "A"})

    dec = TaskDecomposer(_gen)
    with pytest.raises(TaskDecompositionError, match="JSON array"):
        dec.decompose("x")


def test_duplicate_ids_raises():
    payload = [
        {"id": "A", "description": "a", "depends_on": []},
        {"id": "A", "description": "b", "depends_on": []},
    ]
    dec = TaskDecomposer(_fake(payload))
    with pytest.raises(TaskDecompositionError, match="duplicate"):
        dec.decompose("x")


def test_roots_and_leaves_on_fork_join():
    dec = TaskDecomposer(_fake(FORK_JOIN))
    dag = dec.decompose("x")
    assert dag.roots == ["A"]
    assert dag.leaves == ["D"]


def test_tool_hints_and_complexity_round_trip():
    payload = [
        {
            "id": "A",
            "description": "a",
            "depends_on": [],
            "estimated_complexity": "high",
            "tool_hints": ["bash", "python"],
        }
    ]
    dec = TaskDecomposer(_fake(payload))
    dag = dec.decompose("x")
    assert dag.tasks[0].estimated_complexity == "high"
    assert dag.tasks[0].tool_hints == ["bash", "python"]


def test_invalid_complexity_raises():
    payload = [
        {
            "id": "A",
            "description": "a",
            "depends_on": [],
            "estimated_complexity": "epic",
        }
    ]
    dec = TaskDecomposer(_fake(payload))
    with pytest.raises(TaskDecompositionError, match="estimated_complexity"):
        dec.decompose("x")


def test_dataclasses_constructible_directly():
    t = SubTask(id="a", description="do", depends_on=[])
    dag = TaskDAG(tasks=[t], roots=["a"], leaves=["a"])
    assert dag.tasks[0].estimated_complexity == "medium"
    assert dag.tasks[0].tool_hints == []
