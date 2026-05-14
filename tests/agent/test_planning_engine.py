"""Comprehensive tests for agent.planning_engine."""

from __future__ import annotations

import pytest

from agent.planning_engine import (
    PLANNING_ENGINE_REGISTRY,
    Plan,
    PlanExecutor,
    PlanningEngine,
    PlanResult,
    PlanValidationError,
    TaskNode,
    TaskStatus,
)

# ------------------------------------------------------------------ TaskNode


def test_tasknode_defaults():
    n = TaskNode(id="a", description="do a")
    assert n.status == TaskStatus.PENDING
    assert n.dependencies == []
    assert n.children == []
    assert n.max_retries == 2
    assert n.retry_count == 0
    assert n.is_leaf() is True
    assert n.is_atomic() is True


def test_tasknode_with_children():
    child = TaskNode(id="c", description="child")
    parent = TaskNode(id="p", description="parent", children=[child])
    assert parent.is_leaf() is False
    assert parent.is_atomic() is False


def test_tasknode_reset():
    n = TaskNode(
        id="a",
        description="do a",
        status=TaskStatus.FAILED,
        retry_count=2,
        observation="obs",
        error="err",
    )
    n.reset()
    assert n.status == TaskStatus.PENDING
    assert n.retry_count == 0
    assert n.observation == ""
    assert n.error == ""


def test_tasknode_reset_recursive():
    child = TaskNode(id="c", description="child", status=TaskStatus.FAILED, retry_count=1)
    parent = TaskNode(id="p", description="parent", children=[child])
    parent.reset()
    assert child.status == TaskStatus.PENDING
    assert child.retry_count == 0


def test_tasknode_roundtrip_dict():
    n = TaskNode(
        id="a",
        description="do a",
        status=TaskStatus.COMPLETED,
        dependencies=["b"],
        children=[TaskNode(id="c", description="child")],
        max_retries=3,
        retry_count=1,
        observation="ok",
        error="",
        metadata={"key": "val"},
    )
    d = n.to_dict()
    restored = TaskNode.from_dict(d)
    assert restored.id == n.id
    assert restored.description == n.description
    assert restored.status == n.status
    assert restored.dependencies == n.dependencies
    assert len(restored.children) == 1
    assert restored.children[0].id == "c"
    assert restored.max_retries == n.max_retries
    assert restored.retry_count == n.retry_count
    assert restored.observation == n.observation
    assert restored.error == n.error
    assert restored.metadata == n.metadata


# ------------------------------------------------------------------ Plan


def test_plan_create():
    p = Plan.create("build a thing")
    assert p.root.description == "build a thing"
    assert p.metadata["goal"] == "build a thing"
    assert p.metadata["version"] == 1
    assert "created_at" in p.metadata


def test_plan_all_nodes():
    child = TaskNode(id="c", description="child")
    root = TaskNode(id="r", description="root", children=[child])
    p = Plan(root=root)
    ids = {n.id for n in p.all_nodes()}
    assert ids == {"r", "c"}


def test_plan_node_by_id():
    child = TaskNode(id="c", description="child")
    root = TaskNode(id="r", description="root", children=[child])
    p = Plan(root=root)
    assert p.node_by_id("c") is child
    assert p.node_by_id("missing") is None


def test_plan_leaf_nodes():
    leaf1 = TaskNode(id="l1", description="leaf1")
    leaf2 = TaskNode(id="l2", description="leaf2")
    mid = TaskNode(id="m", description="mid", children=[leaf1, leaf2])
    root = TaskNode(id="r", description="root", children=[mid])
    p = Plan(root=root)
    leaves = p.leaf_nodes()
    assert {n.id for n in leaves} == {"l1", "l2"}


def test_plan_bump_version():
    p = Plan.create("goal")
    v1 = p.metadata["version"]
    t1 = p.metadata["created_at"]
    p.bump_version()
    assert p.metadata["version"] == v1 + 1
    assert p.metadata["created_at"] >= t1


def test_plan_roundtrip_dict():
    root = TaskNode(
        id="r",
        description="root",
        children=[TaskNode(id="c", description="child")],
    )
    p = Plan(root=root, metadata={"goal": "g"})
    d = p.to_dict()
    restored = Plan.from_dict(d)
    assert restored.root.id == "r"
    assert restored.root.children[0].id == "c"
    assert restored.metadata["goal"] == "g"


# ------------------------------------------------------------------ PlanningEngine


def test_engine_build_plan_with_decompose_fn():
    def decompose(goal: str) -> Plan:
        return Plan.create(goal, root=TaskNode(id="r", description=goal))

    engine = PlanningEngine(decompose_fn=decompose)
    plan = engine.build_plan("test goal")
    assert plan.root.description == "test goal"


def test_engine_build_plan_without_decompose_fn():
    engine = PlanningEngine()
    plan = engine.build_plan("test goal")
    assert plan.root.description == "test goal"
    assert plan.root.id == "root"


def test_engine_build_plan_empty_goal_raises():
    engine = PlanningEngine()
    with pytest.raises(ValueError, match="non-empty"):
        engine.build_plan("")
    with pytest.raises(ValueError, match="non-empty"):
        engine.build_plan("   ")


def test_engine_build_plan_bad_decompose_type_raises():
    engine = PlanningEngine(decompose_fn=lambda g: "not a plan")
    with pytest.raises(TypeError, match="Plan instance"):
        engine.build_plan("x")


def test_validate_plan_duplicate_ids():
    root = TaskNode(
        id="dup", description="root", children=[TaskNode(id="dup", description="child")]
    )
    plan = Plan(root=root)
    engine = PlanningEngine()
    with pytest.raises(PlanValidationError, match="duplicate"):
        engine.validate_plan(plan)


def test_validate_plan_self_loop():
    root = TaskNode(
        id="r", description="root", children=[TaskNode(id="a", description="a", dependencies=["a"])]
    )
    plan = Plan(root=root)
    engine = PlanningEngine()
    with pytest.raises(PlanValidationError, match="self-loop"):
        engine.validate_plan(plan)


def test_validate_plan_unknown_dependency():
    root = TaskNode(
        id="r",
        description="root",
        children=[TaskNode(id="a", description="a", dependencies=["ghost"])],
    )
    plan = Plan(root=root)
    engine = PlanningEngine()
    with pytest.raises(PlanValidationError, match="unknown id"):
        engine.validate_plan(plan)


def test_validate_plan_max_nodes_exceeded():
    children = [TaskNode(id=f"c{i}", description=f"child {i}") for i in range(6)]
    root = TaskNode(id="r", description="root", children=children)
    plan = Plan(root=root)
    engine = PlanningEngine(max_nodes=5)
    with pytest.raises(PlanValidationError, match="max_nodes"):
        engine.validate_plan(plan)


def test_validate_plan_max_depth_exceeded():
    # depth 4 chain
    n4 = TaskNode(id="d", description="d")
    n3 = TaskNode(id="c", description="c", children=[n4])
    n2 = TaskNode(id="b", description="b", children=[n3])
    n1 = TaskNode(id="a", description="a", children=[n2])
    root = TaskNode(id="r", description="root", children=[n1])
    plan = Plan(root=root)
    engine = PlanningEngine(max_depth=3)
    with pytest.raises(PlanValidationError, match="max_depth"):
        engine.validate_plan(plan)


def test_validate_plan_cycle_in_dependencies():
    a = TaskNode(id="a", description="a", dependencies=["b"])
    b = TaskNode(id="b", description="b", dependencies=["a"])
    root = TaskNode(id="r", description="root", children=[a, b])
    plan = Plan(root=root)
    engine = PlanningEngine()
    with pytest.raises(PlanValidationError, match="cycle"):
        engine.validate_plan(plan)


def test_topological_order_linear():
    a = TaskNode(id="a", description="a")
    b = TaskNode(id="b", description="b", dependencies=["a"])
    c = TaskNode(id="c", description="c", dependencies=["b"])
    root = TaskNode(id="r", description="root", children=[a, b, c])
    plan = Plan(root=root)
    engine = PlanningEngine()
    order = engine.topological_order(plan)
    ids = [n.id for n in order]
    assert ids == ["a", "b", "c"]


def test_topological_order_fork_join():
    a = TaskNode(id="a", description="a")
    b = TaskNode(id="b", description="b", dependencies=["a"])
    c = TaskNode(id="c", description="c", dependencies=["a"])
    d = TaskNode(id="d", description="d", dependencies=["b", "c"])
    root = TaskNode(id="r", description="root", children=[a, b, c, d])
    plan = Plan(root=root)
    engine = PlanningEngine()
    order = engine.topological_order(plan)
    ids = [n.id for n in order]
    assert ids.index("a") < ids.index("b")
    assert ids.index("a") < ids.index("c")
    assert ids.index("b") < ids.index("d")
    assert ids.index("c") < ids.index("d")


def test_topological_order_cycle_raises():
    a = TaskNode(id="a", description="a", dependencies=["b"])
    b = TaskNode(id="b", description="b", dependencies=["a"])
    root = TaskNode(id="r", description="root", children=[a, b])
    plan = Plan(root=root)
    engine = PlanningEngine()
    with pytest.raises(PlanValidationError, match="cycle"):
        engine.topological_order(plan)


def test_revise_plan_with_replanner():
    def replanner(plan: Plan, failed_id: str) -> Plan:
        new_root = TaskNode(id="new_root", description="revised")
        return Plan(root=new_root, metadata=dict(plan.metadata))

    plan = Plan.create("goal", root=TaskNode(id="a", description="a"))
    engine = PlanningEngine()
    revised = engine.revise_plan(plan, "a", replanner=replanner)
    assert revised.root.id == "new_root"
    assert revised.metadata["version"] == 2


def test_revise_plan_without_replanner():
    child = TaskNode(id="c", description="c", status=TaskStatus.FAILED, retry_count=1)
    root = TaskNode(id="r", description="r", children=[child])
    plan = Plan.create("goal", root=root)
    engine = PlanningEngine()
    revised = engine.revise_plan(plan, "c")
    assert revised is plan
    assert revised.metadata["version"] == 2
    assert child.status == TaskStatus.PENDING
    assert child.retry_count == 0


def test_revise_plan_missing_node_raises():
    plan = Plan.create("goal")
    engine = PlanningEngine()
    with pytest.raises(ValueError, match="not found"):
        engine.revise_plan(plan, "ghost")


def test_revise_plan_bad_replanner_type():
    plan = Plan.create("goal", root=TaskNode(id="a", description="a"))
    engine = PlanningEngine()
    with pytest.raises(TypeError, match="Plan instance"):
        engine.revise_plan(plan, "a", replanner=lambda p, fid: "nope")


# ------------------------------------------------------------------ PlanExecutor


def test_executor_happy_path():
    def actor(node: TaskNode) -> str:
        return f"done:{node.id}"

    a = TaskNode(id="a", description="a")
    b = TaskNode(id="b", description="b", dependencies=["a"])
    root = TaskNode(id="r", description="root", children=[a, b])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor)
    result = executor.execute(plan)
    assert result.succeeded is True
    assert result.completed_nodes == ["a", "b"]
    assert result.failed_nodes == []
    assert result.observations == {"a": "done:a", "b": "done:b"}


def test_executor_failure_no_fallback():
    def actor(node: TaskNode) -> str:
        if node.id == "b":
            raise RuntimeError("boom")
        return "ok"

    a = TaskNode(id="a", description="a")
    b = TaskNode(id="b", description="b", dependencies=["a"])
    root = TaskNode(id="r", description="root", children=[a, b])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor)
    result = executor.execute(plan)
    assert result.succeeded is False
    assert result.completed_nodes == ["a"]
    assert result.failed_nodes == ["b"]
    assert "boom" in result.errors["b"]


def test_executor_fallback():
    def actor(node: TaskNode) -> str:
        raise RuntimeError("actor fail")

    def fallback(node: TaskNode) -> str:
        return f"fallback:{node.id}"

    a = TaskNode(id="a", description="a")
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor, fallback=fallback)
    result = executor.execute(plan)
    assert result.succeeded is True
    assert result.completed_nodes == ["a"]
    assert result.observations["a"] == "fallback:a"


def test_executor_fallback_failure():
    def actor(node: TaskNode) -> str:
        raise RuntimeError("actor fail")

    def fallback(node: TaskNode) -> str:
        raise RuntimeError("fallback fail")

    a = TaskNode(id="a", description="a")
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor, fallback=fallback)
    result = executor.execute(plan)
    assert result.succeeded is False
    assert "actor fail" in result.errors["a"]
    assert "fallback fail" in result.errors["a"]


def test_executor_retry_then_success():
    calls = []

    def actor(node: TaskNode) -> str:
        calls.append(node.id)
        if len(calls) < 2:
            raise RuntimeError("transient")
        return "ok"

    a = TaskNode(id="a", description="a", max_retries=2)
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor)
    result = executor.execute(plan)
    assert result.succeeded is True
    assert result.completed_nodes == ["a"]
    assert len(calls) == 2


def test_executor_retry_exhausted():
    def actor(node: TaskNode) -> str:
        raise RuntimeError("persistent")

    a = TaskNode(id="a", description="a", max_retries=1)
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor)
    result = executor.execute(plan)
    assert result.succeeded is False
    assert result.failed_nodes == ["a"]


def test_executor_skip_on_dependency_failure():
    def actor(node: TaskNode) -> str:
        if node.id == "a":
            raise RuntimeError("a fails")
        return "ok"

    a = TaskNode(id="a", description="a")
    b = TaskNode(id="b", description="b", dependencies=["a"])
    root = TaskNode(id="r", description="root", children=[a, b])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor)
    result = executor.execute(plan)
    assert result.succeeded is False
    assert result.completed_nodes == []
    assert result.failed_nodes == ["a"]
    assert b.status == TaskStatus.SKIPPED


def test_executor_on_step_callback():
    log = []

    def actor(node: TaskNode) -> str:
        return "ok"

    def on_step(node: TaskNode, event: str) -> None:
        log.append((node.id, event))

    a = TaskNode(id="a", description="a")
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor, on_step=on_step)
    executor.execute(plan)
    assert log == [("a", "completed")]


def test_executor_on_step_with_failure():
    log = []

    def actor(node: TaskNode) -> str:
        raise RuntimeError("fail")

    def on_step(node: TaskNode, event: str) -> None:
        log.append((node.id, event))

    a = TaskNode(id="a", description="a", max_retries=0)
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=actor, on_step=on_step)
    executor.execute(plan)
    assert log == [("a", "failed")]


def test_executor_non_callable_actor_raises():
    with pytest.raises(TypeError, match="callable"):
        PlanExecutor(actor="not callable")  # type: ignore[arg-type]


def test_executor_result_types():
    a = TaskNode(id="a", description="a")
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=lambda n: "ok")
    result = executor.execute(plan)
    assert isinstance(result, PlanResult)
    assert isinstance(result.plan, Plan)
    assert isinstance(result.succeeded, bool)
    assert isinstance(result.completed_nodes, list)
    assert isinstance(result.failed_nodes, list)
    assert isinstance(result.observations, dict)
    assert isinstance(result.errors, dict)
    assert isinstance(result.execution_log, list)


def test_executor_execution_log_entries():
    a = TaskNode(id="a", description="a")
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=lambda n: "ok")
    result = executor.execute(plan)
    assert len(result.execution_log) == 1
    entry = result.execution_log[0]
    assert entry["node_id"] == "a"
    assert entry["action"] == "execute"
    assert entry["status"] == "completed"


def test_executor_handles_none_observation():
    a = TaskNode(id="a", description="a")
    root = TaskNode(id="r", description="root", children=[a])
    plan = Plan(root=root)
    executor = PlanExecutor(actor=lambda n: None)  # type: ignore[return-value]
    result = executor.execute(plan)
    assert result.observations["a"] == ""


# ------------------------------------------------------------------ Registry


def test_registry_has_defaults():
    assert PLANNING_ENGINE_REGISTRY["default"] is PlanningEngine
    assert PLANNING_ENGINE_REGISTRY["executor"] is PlanExecutor
