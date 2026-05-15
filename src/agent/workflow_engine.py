"""DAG workflow executor inspired by Archon's workflow engine.

Workflows are directed acyclic graphs of nodes. Nodes are executed
sequentially based on dependencies. Supports loops, human gates, and
checkpoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WorkflowError(Exception):
    """Raised when a workflow operation fails."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WorkflowCheckpoint:
    """Snapshot of node execution state."""

    node_id: str
    state: dict[str, Any]
    timestamp: float
    status: str  # "success", "failure", "pending"


@dataclass
class WorkflowNode:
    """A single node in a workflow DAG."""

    node_id: str
    action: str  # "prompt", "bash", "human_gate", "loop", "noop"
    depends_on: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDAG:
    """Directed acyclic graph of workflow nodes."""

    workflow_id: str
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)

    def add_node(self, node: WorkflowNode) -> None:
        """Register a node after validating uniqueness.

        Dependencies are validated lazily in :meth:`validate` to allow
        nodes to be added in any order.
        """
        if node.node_id in self.nodes:
            raise WorkflowError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node

    def validate(self) -> list[str]:
        """Check for missing dependencies and cycles using Kahn's algorithm.

        Returns a list of error strings (empty = valid).
        """
        errors: list[str] = []

        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    errors.append(f"Node '{node.node_id}' depends on undefined node '{dep}'.")

        in_degree = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep in self.nodes:
                    in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            nid = queue.pop(0)
            visited += 1
            for node in self.nodes.values():
                if nid in node.depends_on:
                    in_degree[node.node_id] -= 1
                    if in_degree[node.node_id] == 0:
                        queue.append(node.node_id)

        if visited != len(self.nodes):
            errors.append("Cycle detected in workflow DAG.")

        return errors

    def topological_order(self) -> list[str]:
        """Return node IDs in a valid dependency order.

        Raises WorkflowError if the DAG is invalid.
        """
        errors = self.validate()
        if errors:
            raise WorkflowError(f"Invalid DAG: {'; '.join(errors)}")

        in_degree = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep in self.nodes:
                    in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for node in self.nodes.values():
                if nid in node.depends_on:
                    in_degree[node.node_id] -= 1
                    if in_degree[node.node_id] == 0:
                        queue.append(node.node_id)

        return order

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DAG to a plain dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "nodes": {
                nid: {
                    "node_id": node.node_id,
                    "action": node.action,
                    "depends_on": list(node.depends_on),
                    "payload": dict(node.payload),
                    "max_iterations": node.max_iterations,
                    "metadata": dict(node.metadata),
                }
                for nid, node in self.nodes.items()
            },
        }


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


@dataclass
class WorkflowExecutor:
    """Executes WorkflowDAG instances sequentially."""

    def execute(
        self,
        dag: WorkflowDAG,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run nodes in topological order and return the shared state dict.

        If a node fails, execution stops immediately (fail-fast).
        """
        state: dict[str, Any] = dict(context) if context else {}
        order = dag.topological_order()

        for nid in order:
            node = dag.nodes[nid]
            output = self._run_node(node)
            state[nid] = output

        return state

    def execute_with_checkpoints(
        self,
        dag: WorkflowDAG,
        context: dict[str, Any] | None = None,
    ) -> list[WorkflowCheckpoint]:
        """Same as ``execute`` but returns a checkpoint per node."""
        state: dict[str, Any] = dict(context) if context else {}
        order = dag.topological_order()
        checkpoints: list[WorkflowCheckpoint] = []

        for nid in order:
            node = dag.nodes[nid]
            try:
                output = self._run_node(node)
                state[nid] = output
                status = "success"
            except WorkflowError as exc:
                state[nid] = str(exc)
                status = "failure"
                checkpoints.append(
                    WorkflowCheckpoint(
                        node_id=nid,
                        state=dict(state),
                        timestamp=time.time(),
                        status=status,
                    )
                )
                break

            checkpoints.append(
                WorkflowCheckpoint(
                    node_id=nid,
                    state=dict(state),
                    timestamp=time.time(),
                    status=status,
                )
            )

        return checkpoints

    def resume_from_checkpoint(
        self,
        dag: WorkflowDAG,
        checkpoints: list[WorkflowCheckpoint],
    ) -> dict[str, Any]:
        """Skip nodes whose checkpoint status is ``success``; re-run others."""
        checkpoint_map = {cp.node_id: cp for cp in checkpoints}
        order = dag.topological_order()
        state: dict[str, Any] = dict(checkpoints[-1].state) if checkpoints else {}

        for nid in order:
            cp = checkpoint_map.get(nid)
            if cp and cp.status == "success":
                continue

            node = dag.nodes[nid]
            output = self._run_node(node)
            state[nid] = output

        return state

    @staticmethod
    def _run_node(node: WorkflowNode) -> Any:
        """Execute a single node and return its output."""
        if node.payload.get("fail"):
            raise WorkflowError(f"Node {node.node_id} failed")

        action = node.action
        payload = node.payload

        if action == "prompt":
            return payload.get("text", "")

        if action == "bash":
            cmd = payload.get("cmd", "")
            return f"executed: {cmd}"

        if action == "human_gate":
            return f"{payload.get('question', '')} (awaiting approval)"

        if action == "loop":
            iterations = 0
            sub_node = payload.get("sub_node", "")
            condition = payload.get("condition", "")
            outputs: list[str] = []
            while iterations < node.max_iterations and condition != "done":
                outputs.append(f"executed: {sub_node}")
                iterations += 1
            return {"iterations": iterations, "outputs": outputs}

        if action == "noop":
            return ""

        raise WorkflowError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

DEFAULT_WORKFLOW_EXECUTOR = WorkflowExecutor()
WORKFLOW_EXECUTOR_REGISTRY: dict[str, WorkflowExecutor] = {
    "default": DEFAULT_WORKFLOW_EXECUTOR,
}

__all__ = [
    "DEFAULT_WORKFLOW_EXECUTOR",
    "WORKFLOW_EXECUTOR_REGISTRY",
    "WorkflowCheckpoint",
    "WorkflowDAG",
    "WorkflowError",
    "WorkflowExecutor",
    "WorkflowNode",
]
