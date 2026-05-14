import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    node_id: str
    fn: Callable
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    node_id: str
    status: NodeStatus
    result: Any
    duration_ms: float
    error: str = ""


class DAGExecutor:
    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}

    def add_node(self, node_id: str, fn: Callable, dependencies: list[str] | None = None) -> None:
        if node_id in self._nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        self._nodes[node_id] = DAGNode(
            node_id=node_id, fn=fn, dependencies=list(dependencies or [])
        )

    def _topological_sort(self) -> list[str]:
        indegree: dict[str, int] = {nid: 0 for nid in self._nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise ValueError(f"unknown dependency: {dep}")
                adj[dep].append(node.node_id)
                indegree[node.node_id] += 1

        queue: deque[str] = deque([nid for nid, d in indegree.items() if d == 0])
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for nxt in adj[nid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self._nodes):
            raise ValueError("cycle detected in DAG")
        return order

    def execute(self, inputs: dict | None = None) -> dict[str, ExecutionResult]:
        inputs = inputs or {}
        order = self._topological_sort()
        results: dict[str, ExecutionResult] = {}

        for nid in order:
            node = self._nodes[nid]
            failed_dep = next(
                (
                    d
                    for d in node.dependencies
                    if results.get(d) and results[d].status != NodeStatus.COMPLETED
                ),
                None,
            )
            if failed_dep is not None:
                node.status = NodeStatus.SKIPPED
                results[nid] = ExecutionResult(
                    node_id=nid,
                    status=NodeStatus.SKIPPED,
                    result=None,
                    duration_ms=0.0,
                    error=f"dependency {failed_dep} not completed",
                )
                continue

            kwargs = {d: results[d].result for d in node.dependencies}
            if not node.dependencies and inputs:
                kwargs = dict(inputs)

            node.status = NodeStatus.RUNNING
            start = time.monotonic()
            try:
                out = node.fn(**kwargs) if kwargs else node.fn()
                duration = (time.monotonic() - start) * 1000.0
                node.status = NodeStatus.COMPLETED
                node.result = out
                results[nid] = ExecutionResult(
                    node_id=nid,
                    status=NodeStatus.COMPLETED,
                    result=out,
                    duration_ms=duration,
                )
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000.0
                node.status = NodeStatus.FAILED
                node.error = str(exc)
                results[nid] = ExecutionResult(
                    node_id=nid,
                    status=NodeStatus.FAILED,
                    result=None,
                    duration_ms=duration,
                    error=str(exc),
                )

        return results

    def execution_summary(self, results: dict[str, ExecutionResult]) -> dict:
        completed = sum(1 for r in results.values() if r.status == NodeStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == NodeStatus.FAILED)
        skipped = sum(1 for r in results.values() if r.status == NodeStatus.SKIPPED)
        total_duration = sum(r.duration_ms for r in results.values())
        return {
            "total": len(results),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_duration_ms": total_duration,
        }

    def reset(self) -> None:
        for node in self._nodes.values():
            node.status = NodeStatus.PENDING
            node.result = None
            node.error = ""


DAG_EXECUTOR_REGISTRY: dict[str, type[DAGExecutor]] = {"default": DAGExecutor}
