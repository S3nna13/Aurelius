"""
memory_graph.py — Knowledge graph for memory associations.

Part of the Aurelius memory subsystem. Stdlib-only, no external dependencies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class MemoryNode:
    """A node in the memory knowledge graph."""

    node_id: str
    content: str
    node_type: str = "concept"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEdge:
    """A directed, weighted edge between two nodes."""

    src: str
    dst: str
    relation: str
    weight: float = 1.0


class MemoryGraph:
    """Directed knowledge graph for storing and traversing memory associations."""

    def __init__(self) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        # adjacency: src_id -> list[MemoryEdge]
        self._adj: dict[str, list[MemoryEdge]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: MemoryNode) -> None:
        """Add a node to the graph. Overwrites if node_id already exists."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []

    def add_edge(self, edge: MemoryEdge) -> None:
        """Add a directed edge. Raises ValueError if src or dst not in graph."""
        if edge.src not in self._nodes:
            raise ValueError(f"Source node '{edge.src}' not found in graph.")
        if edge.dst not in self._nodes:
            raise ValueError(f"Destination node '{edge.dst}' not found in graph.")
        self._adj[edge.src].append(edge)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges that reference it (as src or dst)."""
        if node_id not in self._nodes:
            return
        del self._nodes[node_id]
        del self._adj[node_id]
        # Remove all edges where this node is dst
        for src in self._adj:
            self._adj[src] = [e for e in self._adj[src] if e.dst != node_id]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Return the MemoryNode for node_id, or None if not found."""
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str, relation: str | None = None) -> list[MemoryNode]:
        """Return neighbor nodes reachable from node_id via outgoing edges.

        Parameters
        ----------
        node_id:
            The starting node.
        relation:
            If provided, only edges with this relation type are considered.

        Returns
        -------
        list[MemoryNode]
            Neighbor nodes (dst side), preserving insertion order.
        """
        edges = self._adj.get(node_id, [])
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        result: list[MemoryNode] = []
        for edge in edges:
            node = self._nodes.get(edge.dst)
            if node is not None:
                result.append(node)
        return result

    def shortest_path(self, src: str, dst: str) -> list[str]:
        """BFS shortest path from src to dst.

        Returns
        -------
        list[str]
            Ordered list of node_ids from src to dst (inclusive).
            Empty list if either node is missing or dst is unreachable.
        """
        if src not in self._nodes or dst not in self._nodes:
            return []
        if src == dst:
            return [src]

        visited: set[str] = {src}
        # Queue stores (current_node_id, path_so_far)
        queue: deque[tuple[str, list[str]]] = deque([(src, [src])])

        while queue:
            current, path = queue.popleft()
            for edge in self._adj.get(current, []):
                neighbour = edge.dst
                if neighbour == dst:
                    return path + [dst]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, path + [neighbour]))
        return []

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        return sum(len(edges) for edges in self._adj.values())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MEMORY_GRAPH_REGISTRY: dict[str, type] = {
    "default": MemoryGraph,
}
