"""Hierarchical Planning Engine for the Aurelius agent surface.

Decomposes high-level goals into hierarchical task trees (root -> subtasks ->
atomic actions), validates dependency DAGs, supports plan revision when
subtasks fail, and provides a PlanExecutor that runs the plan step-by-step
with retry and fallback.

Pure Python, stdlib only. No async.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """A node in a hierarchical task tree."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    children: list[TaskNode] = field(default_factory=list)
    max_retries: int = 2
    retry_count: int = 0
    observation: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return not self.children

    def is_atomic(self) -> bool:
        return self.is_leaf()

    def reset(self) -> None:
        self.status = TaskStatus.PENDING
        self.retry_count = 0
        self.observation = ""
        self.error = ""
        for child in self.children:
            child.reset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "children": [c.to_dict() for c in self.children],
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "observation": self.observation,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskNode:
        node = cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            dependencies=list(data.get("dependencies", [])),
            max_retries=data.get("max_retries", 2),
            retry_count=data.get("retry_count", 0),
            observation=data.get("observation", ""),
            error=data.get("error", ""),
            metadata=dict(data.get("metadata", {})),
        )
        node.children = [cls.from_dict(c) for c in data.get("children", [])]
        return node


@dataclass
class Plan:
    """A hierarchical plan with a root TaskNode and metadata."""

    root: TaskNode
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, goal: str, root: TaskNode | None = None) -> Plan:
        meta = {
            "goal": goal,
            "created_at": time.time(),
            "version": 1,
        }
        return cls(root=root or TaskNode(id="root", description=goal), metadata=meta)

    def bump_version(self) -> None:
        self.metadata["version"] = self.metadata.get("version", 1) + 1
        self.metadata["created_at"] = time.time()

    def all_nodes(self) -> list[TaskNode]:
        result: list[TaskNode] = []
        queue: deque[TaskNode] = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node)
            queue.extend(node.children)
        return result

    def node_by_id(self, node_id: str) -> TaskNode | None:
        for node in self.all_nodes():
            if node.id == node_id:
                return node
        return None

    def leaf_nodes(self) -> list[TaskNode]:
        return [n for n in self.all_nodes() if n.is_leaf()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        return cls(
            root=TaskNode.from_dict(data["root"]),
            metadata=dict(data.get("metadata", {})),
        )


class PlanValidationError(ValueError):
    """Raised when a plan's dependency graph is invalid."""


class PlanningEngine:
    """Decompose goals into hierarchical plans and validate dependency DAGs."""

    def __init__(
        self,
        decompose_fn: Callable[[str], Plan] | None = None,
        max_depth: int = 4,
        max_nodes: int = 64,
    ) -> None:
        self.decompose_fn = decompose_fn
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    # ------------------------------------------------------------------ core

    def build_plan(self, goal: str) -> Plan:
        """Build a plan from a goal, optionally using a decompose_fn."""
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")

        if self.decompose_fn is not None:
            plan = self.decompose_fn(goal)
            if not isinstance(plan, Plan):
                raise TypeError("decompose_fn must return a Plan instance")
        else:
            plan = Plan.create(goal)

        self.validate_plan(plan)
        return plan

    def validate_plan(self, plan: Plan) -> None:
        """Raise PlanValidationError if the plan is structurally invalid."""
        nodes = plan.all_nodes()
        if len(nodes) > self.max_nodes:
            raise PlanValidationError(
                f"plan has {len(nodes)} nodes, exceeds max_nodes={self.max_nodes}"
            )

        ids = {n.id for n in nodes}
        if len(ids) != len(nodes):
            raise PlanValidationError("duplicate node ids detected")

        # Validate dependencies on leaf nodes only (subtree execution order
        # is handled by the tree structure; explicit deps are for cross-subtree).
        for node in nodes:
            for dep_id in node.dependencies:
                if dep_id == node.id:
                    raise PlanValidationError(f"self-loop detected on node {node.id!r}")
                if dep_id not in ids:
                    raise PlanValidationError(
                        f"node {node.id!r} depends on unknown id {dep_id!r}"
                    )

        depth = self._max_depth(plan.root)
        if depth > self.max_depth:
            raise PlanValidationError(
                f"plan depth {depth} exceeds max_depth={self.max_depth}"
            )

        # Validate no cycles in the explicit dependency DAG across all nodes.
        self._validate_dependency_dag(nodes)

    def topological_order(self, plan: Plan) -> list[TaskNode]:
        """Return leaf nodes in a valid dependency-respecting order.

        Non-leaf nodes are not executable actions; they are structural.
        """
        leaves = plan.leaf_nodes()
        by_id = {n.id: n for n in leaves}
        indegree: dict[str, int] = {n.id: len(n.dependencies) for n in leaves}
        dependents: dict[str, list[str]] = {n.id: [] for n in leaves}
        for n in leaves:
            for dep in n.dependencies:
                if dep in dependents:
                    dependents[dep].append(n.id)

        order: list[TaskNode] = []
        queue: deque[str] = deque(
            n.id for n in leaves if indegree[n.id] == 0
        )
        while queue:
            tid = queue.popleft()
            order.append(by_id[tid])
            for dep in dependents[tid]:
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(leaves):
            raise PlanValidationError("cycle detected in leaf dependency DAG")
        return order

    def revise_plan(
        self,
        plan: Plan,
        failed_node_id: str,
        replanner: Callable[[Plan, str], Plan] | None = None,
    ) -> Plan:
        """Replan the subtree rooted at the failed node.

        If ``replanner`` is provided, it receives the current plan and the
        failed node id and must return a new Plan. Otherwise the failed node's
        subtree is reset to PENDING for re-execution.
        """
        failed = plan.node_by_id(failed_node_id)
        if failed is None:
            raise ValueError(f"node {failed_node_id!r} not found in plan")

        if replanner is not None:
            new_plan = replanner(plan, failed_node_id)
            if not isinstance(new_plan, Plan):
                raise TypeError("replanner must return a Plan instance")
            new_plan.bump_version()
            self.validate_plan(new_plan)
            return new_plan

        # Default revision: reset the failed node and its descendants.
        failed.reset()
        plan.bump_version()
        return plan

    # ------------------------------------------------------------- internals

    def _max_depth(self, node: TaskNode, current: int = 1) -> int:
        if not node.children:
            return current
        return max(self._max_depth(c, current + 1) for c in node.children)

    def _validate_dependency_dag(self, nodes: list[TaskNode]) -> None:
        by_id = {n.id: n for n in nodes}
        indegree: dict[str, int] = {n.id: len(n.dependencies) for n in nodes}
        dependents: dict[str, list[str]] = {n.id: [] for n in nodes}
        for n in nodes:
            for dep in n.dependencies:
                if dep in dependents:
                    dependents[dep].append(n.id)

        queue: deque[str] = deque(n.id for n in nodes if indegree[n.id] == 0)
        visited = 0
        while queue:
            tid = queue.popleft()
            visited += 1
            for dep in dependents[tid]:
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)

        if visited != len(nodes):
            raise PlanValidationError("cycle detected in dependency DAG")
        _ = by_id


@dataclass
class PlanResult:
    """Result of executing a Plan."""

    plan: Plan
    succeeded: bool
    completed_nodes: list[str]
    failed_nodes: list[str]
    observations: dict[str, str]
    errors: dict[str, str]
    execution_log: list[dict[str, Any]]


class PlanExecutor:
    """Execute a Plan step-by-step using an actor callable.

    Handles retries, fallback, and dependency ordering.
    """

    def __init__(
        self,
        actor: Callable[[TaskNode], str],
        fallback: Callable[[TaskNode], str] | None = None,
        max_retries: int = 2,
        on_step: Callable[[TaskNode, str], None] | None = None,
    ) -> None:
        if not callable(actor):
            raise TypeError("actor must be callable")
        self.actor = actor
        self.fallback = fallback
        self.max_retries = max_retries
        self.on_step = on_step

    def execute(self, plan: Plan) -> PlanResult:
        """Run the plan and return a PlanResult."""
        engine = PlanningEngine()
        engine.validate_plan(plan)

        execution_order = engine.topological_order(plan)
        completed: list[str] = []
        failed: list[str] = []
        observations: dict[str, str] = {}
        errors: dict[str, str] = {}
        log: list[dict[str, Any]] = []

        for node in execution_order:
            # Skip if any dependency failed.
            if any(dep in failed for dep in node.dependencies):
                node.status = TaskStatus.SKIPPED
                log.append(
                    {
                        "node_id": node.id,
                        "action": "skip",
                        "reason": "dependency_failed",
                    }
                )
                continue

            result = self._execute_node(node)
            log.append(
                {
                    "node_id": node.id,
                    "action": result["action"],
                    "status": result["status"],
                    "observation": result.get("observation", ""),
                    "error": result.get("error", ""),
                    "retries": result.get("retries", 0),
                }
            )

            if result["status"] == "completed":
                completed.append(node.id)
                observations[node.id] = result.get("observation", "")
            else:
                failed.append(node.id)
                errors[node.id] = result.get("error", "")

        succeeded = len(failed) == 0 and len(completed) == len(execution_order)
        return PlanResult(
            plan=plan,
            succeeded=succeeded,
            completed_nodes=completed,
            failed_nodes=failed,
            observations=observations,
            errors=errors,
            execution_log=log,
        )

    def _execute_node(self, node: TaskNode) -> dict[str, Any]:
        node.status = TaskStatus.IN_PROGRESS
        retries = 0
        max_retries = node.max_retries if node.max_retries is not None else self.max_retries

        while retries <= max_retries:
            try:
                observation = self.actor(node)
                node.status = TaskStatus.COMPLETED
                node.observation = observation if observation is not None else ""
                node.retry_count = retries
                if self.on_step:
                    self.on_step(node, "completed")
                return {
                    "action": "execute",
                    "status": "completed",
                    "observation": node.observation,
                    "retries": retries,
                }
            except Exception as exc:  # noqa: BLE001
                retries += 1
                node.retry_count = retries
                if retries > max_retries:
                    node.status = TaskStatus.FAILED
                    node.error = f"{type(exc).__name__}: {exc}"
                    if self.fallback is not None:
                        try:
                            fallback_obs = self.fallback(node)
                            node.status = TaskStatus.COMPLETED
                            node.observation = (
                                fallback_obs if fallback_obs is not None else ""
                            )
                            if self.on_step:
                                self.on_step(node, "fallback")
                            return {
                                "action": "fallback",
                                "status": "completed",
                                "observation": node.observation,
                                "retries": retries,
                            }
                        except Exception as fb_exc:  # noqa: BLE001
                            node.error += f" | fallback failed: {type(fb_exc).__name__}: {fb_exc}"
                    if self.on_step:
                        self.on_step(node, "failed")
                    return {
                        "action": "execute",
                        "status": "failed",
                        "error": node.error,
                        "retries": retries,
                    }

        # Unreachable, but keeps type checker happy.
        return {"action": "execute", "status": "failed", "error": node.error, "retries": retries}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PLANNING_ENGINE_REGISTRY: dict[str, Any] = {
    "default": PlanningEngine,
    "executor": PlanExecutor,
}
