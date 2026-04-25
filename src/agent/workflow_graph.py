"""DAG-based workflow graph for the Aurelius agent surface.

Supports both sequential (topo-sorted) and parallel (ThreadPoolExecutor)
execution of callable nodes, with DFS cycle detection via WHITE/GRAY/BLACK
colouring.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Enumerations & data classes
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    SKIPPED   = "SKIPPED"


@dataclass
class WorkflowNode:
    node_id: str
    fn: Callable[..., Any]
    deps: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING


@dataclass
class WorkflowResult:
    node_id: str
    output: Any
    error: str | None
    duration_ms: float


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class WorkflowGraph:
    """DAG workflow with sequential and parallel execution modes."""

    def __init__(self) -> None:
        self._nodes: dict[str, WorkflowNode] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        fn: Callable[..., Any],
        deps: list[str] | None = None,
    ) -> None:
        """Register a node.  Raises ValueError on duplicate node_id."""
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already registered.")
        self._nodes[node_id] = WorkflowNode(
            node_id=node_id,
            fn=fn,
            deps=list(deps or []),
        )

    # ------------------------------------------------------------------
    # Validation & topology
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """DFS cycle detection with WHITE/GRAY/BLACK colouring.

        Returns a list of error strings (empty → no cycles).
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {nid: WHITE for nid in self._nodes}
        errors: list[str] = []

        # Check for references to undefined nodes
        for node in self._nodes.values():
            for dep in node.deps:
                if dep not in self._nodes:
                    errors.append(
                        f"Node '{node.node_id}' depends on undefined node '{dep}'."
                    )

        def dfs(nid: str) -> None:
            colour[nid] = GRAY
            for dep in self._nodes[nid].deps:
                if dep not in self._nodes:
                    continue  # already reported above
                if colour[dep] == GRAY:
                    errors.append(
                        f"Cycle detected: '{dep}' is an ancestor of '{nid}'."
                    )
                elif colour[dep] == WHITE:
                    dfs(dep)
            colour[nid] = BLACK

        for nid in list(self._nodes):
            if colour[nid] == WHITE:
                dfs(nid)

        return errors

    def _topological_sort(self) -> list[str]:
        """Return node IDs in a valid execution order (deps before dependents)."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for dep in self._nodes[nid].deps:
                if dep in self._nodes:
                    visit(dep)
            order.append(nid)

        for nid in self._nodes:
            visit(nid)

        return order

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _run_node(
        self,
        node: WorkflowNode,
        state: dict,
    ) -> WorkflowResult:
        """Execute a single node, catching all exceptions."""
        node.status = NodeStatus.RUNNING
        t0 = time.perf_counter()
        try:
            output = node.fn(state)
            node.status = NodeStatus.COMPLETED
            return WorkflowResult(
                node_id=node.node_id,
                output=output,
                error=None,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            node.status = NodeStatus.FAILED
            return WorkflowResult(
                node_id=node.node_id,
                output=None,
                error=str(exc),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )

    # ------------------------------------------------------------------
    # Sequential execution
    # ------------------------------------------------------------------

    def run_sequential(self, state: dict) -> dict[str, WorkflowResult]:
        """Execute all nodes in topological order, passing cumulative state."""
        order = self._topological_sort()
        results: dict[str, WorkflowResult] = {}

        for nid in order:
            node = self._nodes[nid]
            result = self._run_node(node, state)
            results[nid] = result
            # Fold output into state so downstream nodes can see it
            if result.error is None and result.output is not None:
                if isinstance(result.output, dict):
                    state.update(result.output)
                else:
                    state[nid] = result.output

        return results

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    def run_parallel(
        self,
        state: dict,
        max_workers: int = 4,
    ) -> dict[str, WorkflowResult]:
        """Execute nodes in parallel using ThreadPoolExecutor.

        Nodes are submitted as soon as all their dependencies are COMPLETED.
        The loop repeats until every node is done or no progress can be made.
        """
        # Reset statuses
        for node in self._nodes.values():
            node.status = NodeStatus.PENDING

        results: dict[str, WorkflowResult] = {}
        in_flight: dict[str, Future] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # Collect completed futures
                done_ids = [
                    nid for nid, fut in in_flight.items() if fut.done()
                ]
                for nid in done_ids:
                    result = in_flight.pop(nid).result()
                    results[nid] = result
                    node = self._nodes[nid]
                    if result.error is None and result.output is not None:
                        if isinstance(result.output, dict):
                            state.update(result.output)
                        else:
                            state[nid] = result.output

                # Determine which nodes are ready to run
                submitted = False
                for nid, node in self._nodes.items():
                    if node.status != NodeStatus.PENDING:
                        continue
                    if nid in in_flight:
                        continue
                    deps_ok = all(
                        self._nodes[dep].status == NodeStatus.COMPLETED
                        for dep in node.deps
                        if dep in self._nodes
                    )
                    if deps_ok:
                        fut = executor.submit(self._run_node, node, dict(state))
                        in_flight[nid] = fut
                        submitted = True

                # Check termination
                all_done = all(
                    n.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)
                    for n in self._nodes.values()
                ) and not in_flight

                if all_done:
                    break

                # Stuck: no progress and nothing in flight
                if not submitted and not in_flight:
                    # Mark remaining pending nodes as SKIPPED
                    for nid, node in self._nodes.items():
                        if node.status == NodeStatus.PENDING:
                            node.status = NodeStatus.SKIPPED
                            results[nid] = WorkflowResult(
                                node_id=nid,
                                output=None,
                                error="Skipped: dependency failed or cycle.",
                                duration_ms=0.0,
                            )
                    break

                if in_flight:
                    # Wait for at least one future to complete before re-checking
                    next(as_completed(in_flight.values()), None)

        return results

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    def node_merge(self, results: dict[str, WorkflowResult]) -> dict:
        """Merge all successful node outputs into a single dict."""
        merged: dict = {}
        for result in results.values():
            if result.error is None and result.output is not None:
                if isinstance(result.output, dict):
                    merged.update(result.output)
                else:
                    merged[result.node_id] = result.output
        return merged


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, Any] = {
    "workflow_graph": WorkflowGraph,
}
