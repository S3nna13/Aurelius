"""Topology manager: network topology for federated learning nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class NodeRole(StrEnum):
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    AGGREGATOR = "AGGREGATOR"
    OBSERVER = "OBSERVER"


@dataclass
class FedNode:
    node_id: str
    role: NodeRole
    address: str
    available: bool = True
    metadata: dict = field(default_factory=dict)


class TopologyManager:
    """Manages the federated learning network topology."""

    def __init__(self) -> None:
        self._nodes: dict[str, FedNode] = {}

    def add_node(self, node: FedNode) -> None:
        """Register a node. Raises ValueError if node_id already exists."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists.")
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node by id. Returns True if removed, False if not found."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            return True
        return False

    def get_node(self, node_id: str) -> FedNode | None:
        """Return the node with the given id, or None."""
        return self._nodes.get(node_id)

    def nodes_by_role(self, role: NodeRole) -> list[FedNode]:
        """Return all nodes with the given role, sorted by node_id."""
        return sorted(
            (n for n in self._nodes.values() if n.role == role),
            key=lambda n: n.node_id,
        )

    def available_clients(self) -> list[FedNode]:
        """Return available CLIENT nodes, sorted by node_id."""
        return sorted(
            (n for n in self._nodes.values() if n.role == NodeRole.CLIENT and n.available),
            key=lambda n: n.node_id,
        )

    def set_availability(self, node_id: str, available: bool) -> bool:
        """Set availability flag for a node. Returns False if node not found."""
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.available = available
        return True

    def node_count(self) -> int:
        """Return total number of registered nodes."""
        return len(self._nodes)

    def topology_summary(self) -> dict:
        """Return a summary of the current topology."""
        by_role: dict[str, int] = {}
        for role in NodeRole:
            count = sum(1 for n in self._nodes.values() if n.role == role)
            by_role[role.value] = count
        return {
            "total": len(self._nodes),
            "by_role": by_role,
            "available_clients": len(self.available_clients()),
        }


TOPOLOGY_MANAGER_REGISTRY: dict[str, type] = {"default": TopologyManager}
