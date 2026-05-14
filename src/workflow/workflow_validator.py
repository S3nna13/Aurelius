"""Workflow DAG validator — cycle, orphan, and dependency checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


class WorkflowValidationError(ValueError):
    """Raised when a workflow DAG fails validation."""


@dataclass(frozen=True)
class ValidationReport:
    """Result of workflow validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


class WorkflowValidator:
    """Validate a workflow DAG for structural correctness."""

    @staticmethod
    def validate(nodes: dict[str, Any] | list[Any]) -> ValidationReport:
        """Validate a DAG and return a report."""
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(nodes, (dict, list)):
            errors.append("nodes must be a dict or list")
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings)

        node_list: list[Any]
        if isinstance(nodes, dict):
            node_list = list(nodes.values())
        else:
            node_list = list(nodes)

        if not node_list:
            return ValidationReport(is_valid=True, errors=[], warnings=[])

        # Collect node IDs and detect duplicates
        seen_ids: set[str] = set()
        node_ids: set[str] = set()
        for node in node_list:
            if not hasattr(node, "node_id"):
                errors.append(f"node missing node_id: {type(node).__name__}")
                continue
            nid = str(node.node_id)
            if nid in seen_ids:
                errors.append(f"duplicate node_id: {nid}")
            seen_ids.add(nid)
            node_ids.add(nid)

        if errors:
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings)

        # Build adjacency and indegree
        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        indegree: dict[str, int] = {nid: 0 for nid in node_ids}
        dep_lookup: dict[str, list[str]] = {}

        for node in node_list:
            nid = str(node.node_id)
            deps: list[str] = []
            if hasattr(node, "dependencies"):
                raw = node.dependencies
                if isinstance(raw, (list, tuple)):
                    deps = [str(d) for d in raw]
                elif raw is not None:
                    errors.append(f"node {nid}: dependencies must be a list")
                    continue
            dep_lookup[nid] = deps
            for dep in deps:
                if dep not in node_ids:
                    errors.append(f"node {nid}: missing dependency {dep}")
                else:
                    adj[dep].append(nid)
                    indegree[nid] += 1

        if errors:
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings)

        # Cycle detection via Kahn's algorithm
        queue = deque([nid for nid, deg in indegree.items() if deg == 0])
        visited = 0
        topo: list[str] = []
        while queue:
            cur = queue.popleft()
            visited += 1
            topo.append(cur)
            for nxt in adj[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if visited != len(node_ids):
            # Find nodes in cycle
            remaining = {nid for nid, deg in indegree.items() if deg > 0}
            errors.append(f"cycle detected involving nodes: {sorted(remaining)}")
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings)

        # Orphan detection: nodes not reachable from any root via reverse traversal
        roots = [nid for nid, deps in dep_lookup.items() if not deps]
        if not roots and len(node_ids) > 1:
            warnings.append("no root nodes (all nodes have dependencies)")

        reachable: set[str] = set()
        rev_adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for nid, deps in dep_lookup.items():
            for dep in deps:
                rev_adj[nid].append(dep)

        # BFS from roots forward through dependency graph
        fwd_adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for nid, deps in dep_lookup.items():
            for dep in deps:
                fwd_adj[dep].append(nid)

        bfs = deque(roots)
        while bfs:
            cur = bfs.popleft()
            if cur in reachable:
                continue
            reachable.add(cur)
            for nxt in fwd_adj[cur]:
                if nxt not in reachable:
                    bfs.append(nxt)

        orphans = node_ids - reachable
        if orphans:
            warnings.append(f"orphaned nodes (unreachable from roots): {sorted(orphans)}")

        return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_and_raise(nodes: dict[str, Any] | list[Any]) -> None:
        """Validate and raise WorkflowValidationError on failure."""
        report = WorkflowValidator.validate(nodes)
        if not report.is_valid:
            raise WorkflowValidationError("; ".join(report.errors))
