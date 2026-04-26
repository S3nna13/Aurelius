"""Tests for src/federation/topology_manager.py."""

from __future__ import annotations

import pytest

from src.federation.topology_manager import (
    TOPOLOGY_MANAGER_REGISTRY,
    FedNode,
    NodeRole,
    TopologyManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(
    node_id: str,
    role: NodeRole = NodeRole.CLIENT,
    address: str = "127.0.0.1:5000",
    available: bool = True,
    metadata: dict | None = None,
) -> FedNode:
    return FedNode(
        node_id=node_id,
        role=role,
        address=address,
        available=available,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# FedNode dataclass
# ---------------------------------------------------------------------------


class TestFedNode:
    def test_required_fields_stored(self):
        node = make_node("n1", NodeRole.SERVER, "10.0.0.1:8080")
        assert node.node_id == "n1"
        assert node.role == NodeRole.SERVER
        assert node.address == "10.0.0.1:8080"

    def test_available_default_true(self):
        node = FedNode(node_id="n1", role=NodeRole.CLIENT, address="addr")
        assert node.available is True

    def test_metadata_default_empty_dict(self):
        node = FedNode(node_id="n1", role=NodeRole.CLIENT, address="addr")
        assert node.metadata == {}

    def test_metadata_independent_per_instance(self):
        n1 = FedNode(node_id="n1", role=NodeRole.CLIENT, address="a")
        n2 = FedNode(node_id="n2", role=NodeRole.CLIENT, address="b")
        n1.metadata["key"] = "value"
        assert "key" not in n2.metadata

    def test_node_role_values(self):
        assert NodeRole.SERVER.value == "SERVER"
        assert NodeRole.CLIENT.value == "CLIENT"
        assert NodeRole.AGGREGATOR.value == "AGGREGATOR"
        assert NodeRole.OBSERVER.value == "OBSERVER"


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_add_node_increases_count(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        assert tm.node_count() == 1

    def test_add_multiple_nodes(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        tm.add_node(make_node("n2"))
        assert tm.node_count() == 2

    def test_add_duplicate_raises_value_error(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        with pytest.raises(ValueError):
            tm.add_node(make_node("n1"))

    def test_add_duplicate_message_contains_node_id(self):
        tm = TopologyManager()
        tm.add_node(make_node("dup_node"))
        with pytest.raises(ValueError, match="dup_node"):
            tm.add_node(make_node("dup_node"))

    def test_add_node_different_roles(self):
        tm = TopologyManager()
        tm.add_node(make_node("s1", NodeRole.SERVER))
        tm.add_node(make_node("a1", NodeRole.AGGREGATOR))
        assert tm.node_count() == 2


# ---------------------------------------------------------------------------
# remove_node
# ---------------------------------------------------------------------------


class TestRemoveNode:
    def test_remove_existing_returns_true(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        assert tm.remove_node("n1") is True

    def test_remove_existing_decreases_count(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        tm.remove_node("n1")
        assert tm.node_count() == 0

    def test_remove_missing_returns_false(self):
        tm = TopologyManager()
        assert tm.remove_node("ghost") is False

    def test_remove_then_re_add_succeeds(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        tm.remove_node("n1")
        tm.add_node(make_node("n1"))
        assert tm.node_count() == 1


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


class TestGetNode:
    def test_get_existing_node(self):
        tm = TopologyManager()
        node = make_node("n1")
        tm.add_node(node)
        result = tm.get_node("n1")
        assert result is node

    def test_get_missing_returns_none(self):
        tm = TopologyManager()
        assert tm.get_node("missing") is None

    def test_get_after_remove_returns_none(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        tm.remove_node("n1")
        assert tm.get_node("n1") is None


# ---------------------------------------------------------------------------
# nodes_by_role
# ---------------------------------------------------------------------------


class TestNodesByRole:
    def test_nodes_by_role_returns_correct_role(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT))
        tm.add_node(make_node("s1", NodeRole.SERVER))
        clients = tm.nodes_by_role(NodeRole.CLIENT)
        assert all(n.role == NodeRole.CLIENT for n in clients)

    def test_nodes_by_role_sorted_by_node_id(self):
        tm = TopologyManager()
        tm.add_node(make_node("c3", NodeRole.CLIENT))
        tm.add_node(make_node("c1", NodeRole.CLIENT))
        tm.add_node(make_node("c2", NodeRole.CLIENT))
        ids = [n.node_id for n in tm.nodes_by_role(NodeRole.CLIENT)]
        assert ids == ["c1", "c2", "c3"]

    def test_nodes_by_role_empty_when_none(self):
        tm = TopologyManager()
        tm.add_node(make_node("s1", NodeRole.SERVER))
        assert tm.nodes_by_role(NodeRole.AGGREGATOR) == []

    def test_nodes_by_role_excludes_other_roles(self):
        tm = TopologyManager()
        tm.add_node(make_node("s1", NodeRole.SERVER))
        tm.add_node(make_node("c1", NodeRole.CLIENT))
        servers = tm.nodes_by_role(NodeRole.SERVER)
        assert len(servers) == 1
        assert servers[0].node_id == "s1"


# ---------------------------------------------------------------------------
# available_clients
# ---------------------------------------------------------------------------


class TestAvailableClients:
    def test_available_clients_only_available_clients(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT, available=True))
        tm.add_node(make_node("c2", NodeRole.CLIENT, available=False))
        result = tm.available_clients()
        assert len(result) == 1
        assert result[0].node_id == "c1"

    def test_available_clients_excludes_servers(self):
        tm = TopologyManager()
        tm.add_node(make_node("s1", NodeRole.SERVER, available=True))
        tm.add_node(make_node("c1", NodeRole.CLIENT, available=True))
        result = tm.available_clients()
        assert all(n.role == NodeRole.CLIENT for n in result)

    def test_available_clients_sorted_by_node_id(self):
        tm = TopologyManager()
        tm.add_node(make_node("c3", NodeRole.CLIENT))
        tm.add_node(make_node("c1", NodeRole.CLIENT))
        tm.add_node(make_node("c2", NodeRole.CLIENT))
        ids = [n.node_id for n in tm.available_clients()]
        assert ids == ["c1", "c2", "c3"]

    def test_available_clients_empty_when_all_unavailable(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT, available=False))
        assert tm.available_clients() == []


# ---------------------------------------------------------------------------
# set_availability
# ---------------------------------------------------------------------------


class TestSetAvailability:
    def test_set_availability_to_false(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT, available=True))
        result = tm.set_availability("c1", False)
        assert result is True
        assert tm.get_node("c1").available is False

    def test_set_availability_to_true(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT, available=False))
        tm.set_availability("c1", True)
        assert tm.get_node("c1").available is True

    def test_set_availability_missing_returns_false(self):
        tm = TopologyManager()
        assert tm.set_availability("ghost", False) is False

    def test_set_availability_affects_available_clients(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT, available=True))
        assert len(tm.available_clients()) == 1
        tm.set_availability("c1", False)
        assert len(tm.available_clients()) == 0


# ---------------------------------------------------------------------------
# node_count
# ---------------------------------------------------------------------------


class TestNodeCount:
    def test_initial_count_zero(self):
        tm = TopologyManager()
        assert tm.node_count() == 0

    def test_count_after_add(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        tm.add_node(make_node("n2"))
        assert tm.node_count() == 2

    def test_count_after_remove(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        tm.add_node(make_node("n2"))
        tm.remove_node("n1")
        assert tm.node_count() == 1


# ---------------------------------------------------------------------------
# topology_summary
# ---------------------------------------------------------------------------


class TestTopologySummary:
    def test_summary_has_required_keys(self):
        tm = TopologyManager()
        summary = tm.topology_summary()
        assert "total" in summary
        assert "by_role" in summary
        assert "available_clients" in summary

    def test_summary_total_count(self):
        tm = TopologyManager()
        tm.add_node(make_node("n1"))
        tm.add_node(make_node("n2"))
        assert tm.topology_summary()["total"] == 2

    def test_summary_by_role_contains_all_roles(self):
        tm = TopologyManager()
        summary = tm.topology_summary()
        for role in NodeRole:
            assert role.value in summary["by_role"]

    def test_summary_by_role_counts_correct(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT))
        tm.add_node(make_node("c2", NodeRole.CLIENT))
        tm.add_node(make_node("s1", NodeRole.SERVER))
        summary = tm.topology_summary()
        assert summary["by_role"][NodeRole.CLIENT.value] == 2
        assert summary["by_role"][NodeRole.SERVER.value] == 1

    def test_summary_available_clients_count(self):
        tm = TopologyManager()
        tm.add_node(make_node("c1", NodeRole.CLIENT, available=True))
        tm.add_node(make_node("c2", NodeRole.CLIENT, available=False))
        summary = tm.topology_summary()
        assert summary["available_clients"] == 1

    def test_summary_empty_topology(self):
        tm = TopologyManager()
        summary = tm.topology_summary()
        assert summary["total"] == 0
        assert summary["available_clients"] == 0
        for role in NodeRole:
            assert summary["by_role"][role.value] == 0


# ---------------------------------------------------------------------------
# TOPOLOGY_MANAGER_REGISTRY
# ---------------------------------------------------------------------------


class TestTopologyManagerRegistry:
    def test_registry_exists(self):
        assert TOPOLOGY_MANAGER_REGISTRY is not None

    def test_registry_has_default(self):
        assert "default" in TOPOLOGY_MANAGER_REGISTRY

    def test_registry_default_is_class(self):
        assert TOPOLOGY_MANAGER_REGISTRY["default"] is TopologyManager

    def test_registry_default_is_callable(self):
        cls = TOPOLOGY_MANAGER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, TopologyManager)
