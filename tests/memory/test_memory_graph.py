"""
Tests for src/memory/memory_graph.py

Coverage: MemoryNode, MemoryEdge, MemoryGraph, MEMORY_GRAPH_REGISTRY.
At least 28 test functions.
"""

import pytest

from src.memory.memory_graph import (
    MEMORY_GRAPH_REGISTRY,
    MemoryEdge,
    MemoryGraph,
    MemoryNode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph():
    return MemoryGraph()


@pytest.fixture()
def populated_graph():
    g = MemoryGraph()
    g.add_node(MemoryNode("a", "Alpha", "concept"))
    g.add_node(MemoryNode("b", "Beta", "fact"))
    g.add_node(MemoryNode("c", "Gamma", "concept"))
    g.add_node(MemoryNode("d", "Delta", "fact"))
    g.add_edge(MemoryEdge("a", "b", "relates_to"))
    g.add_edge(MemoryEdge("b", "c", "leads_to"))
    g.add_edge(MemoryEdge("a", "c", "relates_to"))
    g.add_edge(MemoryEdge("c", "d", "causes"))
    return g


# ---------------------------------------------------------------------------
# MemoryNode dataclass
# ---------------------------------------------------------------------------


class TestMemoryNode:
    def test_node_creation_defaults(self):
        n = MemoryNode("n1", "hello")
        assert n.node_id == "n1"
        assert n.content == "hello"
        assert n.node_type == "concept"
        assert n.metadata == {}

    def test_node_creation_explicit(self):
        n = MemoryNode("n2", "world", node_type="fact", metadata={"src": "wiki"})
        assert n.node_type == "fact"
        assert n.metadata == {"src": "wiki"}

    def test_node_metadata_isolation(self):
        """Separate instances should not share the same metadata dict."""
        n1 = MemoryNode("x", "foo")
        n2 = MemoryNode("y", "bar")
        n1.metadata["key"] = "val"
        assert "key" not in n2.metadata


# ---------------------------------------------------------------------------
# MemoryEdge dataclass
# ---------------------------------------------------------------------------


class TestMemoryEdge:
    def test_edge_creation_defaults(self):
        e = MemoryEdge("a", "b", "likes")
        assert e.src == "a"
        assert e.dst == "b"
        assert e.relation == "likes"
        assert e.weight == 1.0

    def test_edge_custom_weight(self):
        e = MemoryEdge("a", "b", "likes", weight=0.5)
        assert e.weight == 0.5

    def test_edge_frozen(self):
        e = MemoryEdge("a", "b", "rel")
        with pytest.raises((AttributeError, TypeError)):
            e.weight = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryGraph: add / get
# ---------------------------------------------------------------------------


class TestAddGetNode:
    def test_add_and_get_node(self, graph):
        n = MemoryNode("n1", "content")
        graph.add_node(n)
        assert graph.get_node("n1") is n

    def test_get_missing_node_returns_none(self, graph):
        assert graph.get_node("missing") is None

    def test_overwrite_node(self, graph):
        n1 = MemoryNode("id", "first")
        n2 = MemoryNode("id", "second")
        graph.add_node(n1)
        graph.add_node(n2)
        assert graph.get_node("id").content == "second"

    def test_node_count_zero_initially(self, graph):
        assert graph.node_count() == 0

    def test_node_count_after_adds(self, graph):
        graph.add_node(MemoryNode("a", "A"))
        graph.add_node(MemoryNode("b", "B"))
        assert graph.node_count() == 2


# ---------------------------------------------------------------------------
# MemoryGraph: add_edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    def test_add_edge_valid(self, graph):
        graph.add_node(MemoryNode("a", "A"))
        graph.add_node(MemoryNode("b", "B"))
        graph.add_edge(MemoryEdge("a", "b", "links"))
        assert graph.edge_count() == 1

    def test_add_edge_unknown_src_raises(self, graph):
        graph.add_node(MemoryNode("b", "B"))
        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge(MemoryEdge("ghost", "b", "rel"))

    def test_add_edge_unknown_dst_raises(self, graph):
        graph.add_node(MemoryNode("a", "A"))
        with pytest.raises(ValueError, match="Destination node"):
            graph.add_edge(MemoryEdge("a", "ghost", "rel"))

    def test_edge_count_zero_initially(self, graph):
        assert graph.edge_count() == 0

    def test_edge_count_multiple(self, populated_graph):
        assert populated_graph.edge_count() == 4


# ---------------------------------------------------------------------------
# MemoryGraph: neighbors
# ---------------------------------------------------------------------------


class TestNeighbors:
    def test_neighbors_no_filter(self, populated_graph):
        result = populated_graph.neighbors("a")
        ids = {n.node_id for n in result}
        assert ids == {"b", "c"}

    def test_neighbors_with_relation_filter(self, populated_graph):
        result = populated_graph.neighbors("a", relation="relates_to")
        ids = {n.node_id for n in result}
        assert ids == {"b", "c"}

    def test_neighbors_filter_excludes_other_relations(self, populated_graph):
        result = populated_graph.neighbors("c", relation="relates_to")
        assert result == []

    def test_neighbors_filter_causes(self, populated_graph):
        result = populated_graph.neighbors("c", relation="causes")
        assert len(result) == 1
        assert result[0].node_id == "d"

    def test_neighbors_leaf_node(self, populated_graph):
        # "d" has no outgoing edges
        result = populated_graph.neighbors("d")
        assert result == []

    def test_neighbors_missing_node_returns_empty(self, populated_graph):
        result = populated_graph.neighbors("z")
        assert result == []


# ---------------------------------------------------------------------------
# MemoryGraph: shortest_path
# ---------------------------------------------------------------------------


class TestShortestPath:
    def test_shortest_path_direct(self, populated_graph):
        path = populated_graph.shortest_path("a", "b")
        assert path == ["a", "b"]

    def test_shortest_path_two_hops(self, populated_graph):
        # a->b->c exists but a->c directly also exists; BFS may return either
        path = populated_graph.shortest_path("a", "d")
        assert path[0] == "a"
        assert path[-1] == "d"
        assert len(path) >= 2

    def test_shortest_path_same_node(self, populated_graph):
        path = populated_graph.shortest_path("a", "a")
        assert path == ["a"]

    def test_shortest_path_unreachable(self, populated_graph):
        # Add isolated node
        populated_graph.add_node(MemoryNode("iso", "Isolated"))
        path = populated_graph.shortest_path("a", "iso")
        assert path == []

    def test_shortest_path_missing_src(self, populated_graph):
        path = populated_graph.shortest_path("zzz", "a")
        assert path == []

    def test_shortest_path_missing_dst(self, populated_graph):
        path = populated_graph.shortest_path("a", "zzz")
        assert path == []

    def test_shortest_path_via_intermediate(self):
        g = MemoryGraph()
        for nid in ["x", "y", "z"]:
            g.add_node(MemoryNode(nid, nid))
        g.add_edge(MemoryEdge("x", "y", "r"))
        g.add_edge(MemoryEdge("y", "z", "r"))
        path = g.shortest_path("x", "z")
        assert path == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# MemoryGraph: remove_node
# ---------------------------------------------------------------------------


class TestRemoveNode:
    def test_remove_node_decrements_count(self, populated_graph):
        before = populated_graph.node_count()
        populated_graph.remove_node("d")
        assert populated_graph.node_count() == before - 1

    def test_remove_node_get_returns_none(self, populated_graph):
        populated_graph.remove_node("b")
        assert populated_graph.get_node("b") is None

    def test_remove_node_removes_outgoing_edges(self, populated_graph):
        populated_graph.remove_node("b")
        # edge b->c should be gone; neighbors of b should not include c anymore
        # b is gone, so edge_count should decrease
        # a->b edge and b->c edge both removed
        assert populated_graph.edge_count() == 2  # a->c and c->d remain

    def test_remove_node_removes_incoming_edges(self, populated_graph):
        # removing "c" removes a->c, b->c, and c->d
        populated_graph.remove_node("c")
        assert populated_graph.edge_count() == 1  # only a->b remains

    def test_remove_node_missing_no_error(self, graph):
        # Should not raise
        graph.remove_node("nonexistent")

    def test_remove_node_then_add_same_id(self, populated_graph):
        populated_graph.remove_node("d")
        new_d = MemoryNode("d", "New Delta")
        populated_graph.add_node(new_d)
        assert populated_graph.get_node("d").content == "New Delta"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_exists(self):
        assert isinstance(MEMORY_GRAPH_REGISTRY, dict)

    def test_registry_has_default(self):
        assert "default" in MEMORY_GRAPH_REGISTRY

    def test_registry_default_is_memory_graph(self):
        assert MEMORY_GRAPH_REGISTRY["default"] is MemoryGraph

    def test_registry_default_instantiates(self):
        cls = MEMORY_GRAPH_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, MemoryGraph)
