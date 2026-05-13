"""Tests: plugins/memory/memory_graph.py — Knowledge graph for memory associations."""

from __future__ import annotations

import pytest

from plugins.memory.memory_graph import (
    MEMORY_GRAPH_REGISTRY,
    MemoryEdge,
    MemoryGraph,
    MemoryNode,
)


@pytest.fixture
def graph():
    """Empty graph."""
    return MemoryGraph()


@pytest.fixture
def populated_graph():
    """Graph with 4 nodes and 3 edges."""
    g = MemoryGraph()
    g.add_node(MemoryNode("n1", "concept A"))
    g.add_node(MemoryNode("n2", "concept B"))
    g.add_node(MemoryNode("n3", "concept C"))
    g.add_node(MemoryNode("n4", "concept D"))
    g.add_edge(MemoryEdge("n1", "n2", "relates_to"))
    g.add_edge(MemoryEdge("n2", "n3", "precedes"))
    # NOTE: no direct n1->n3 edge; shortest path MUST go n1->n2->n3
    return g


class TestMemoryNode:
    def test_node_creation(self):
        node = MemoryNode("n1", "content", "type", {"key": "val"})
        assert node.node_id == "n1"
        assert node.content == "content"
        assert node.node_type == "type"
        assert node.metadata == {"key": "val"}

    def test_node_default_type(self):
        node = MemoryNode("n2", "content")
        assert node.node_type == "concept"

    def test_node_default_metadata(self):
        node = MemoryNode("n3", "content")
        assert node.metadata == {}


class TestMemoryEdge:
    def test_edge_creation(self):
        edge = MemoryEdge("a", "b", "rel", 0.5)
        assert edge.src == "a"
        assert edge.dst == "b"
        assert edge.relation == "rel"
        assert edge.weight == 0.5

    def test_edge_default_weight(self):
        edge = MemoryEdge("x", "y", "links")
        assert edge.weight == 1.0


class TestMemoryGraphMutation:
    def test_add_node(self, graph):
        node = MemoryNode("n1", "test")
        graph.add_node(node)
        assert graph.get_node("n1") is node

    def test_add_node_overwrites(self, graph):
        node1 = MemoryNode("n1", "first")
        node2 = MemoryNode("n1", "second")
        graph.add_node(node1)
        graph.add_node(node2)
        assert graph.get_node("n1") is node2

    def test_add_edge_success(self, graph):
        graph.add_node(MemoryNode("a", "A"))
        graph.add_node(MemoryNode("b", "B"))
        graph.add_edge(MemoryEdge("a", "b", "relates"))
        neighbors = graph.neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0].node_id == "b"

    def test_add_edge_src_not_found(self, graph):
        graph.add_node(MemoryNode("b", "B"))
        with pytest.raises(ValueError, match="Source node 'a' not found"):
            graph.add_edge(MemoryEdge("a", "b", "rel"))

    def test_add_edge_dst_not_found(self, graph):
        graph.add_node(MemoryNode("a", "A"))
        with pytest.raises(ValueError, match="Destination node 'b' not found"):
            graph.add_edge(MemoryEdge("a", "b", "rel"))

    def test_remove_node(self, graph):
        graph.add_node(MemoryNode("n1", "test"))
        graph.add_node(MemoryNode("n2", "other"))
        graph.add_edge(MemoryEdge("n1", "n2", "links"))
        graph.remove_node("n1")
        assert graph.get_node("n1") is None
        assert graph.get_node("n2") is not None
        # Edge removed because n1 was dst
        assert graph.neighbors("n2") == []

    def test_remove_nonexistent_node(self, graph):
        graph.remove_node("ghost")  # Should not raise


class TestMemoryGraphQuery:
    def test_get_node_found(self, graph):
        node = MemoryNode("n1", "content")
        graph.add_node(node)
        assert graph.get_node("n1") is node

    def test_get_node_not_found(self, graph):
        assert graph.get_node("ghost") is None

    def test_neighbors_no_edges(self, graph):
        graph.add_node(MemoryNode("n1", "solo"))
        assert graph.neighbors("n1") == []

    def test_neighbors_by_relation(self, populated_graph):
        n1_neighbors = populated_graph.neighbors("n1", relation="relates_to")
        assert len(n1_neighbors) == 1
        assert n1_neighbors[0].node_id == "n2"

    def test_neighbors_wrong_relation(self, populated_graph):
        n1_neighbors = populated_graph.neighbors("n1", relation="precedes")
        assert n1_neighbors == []


class TestShortestPath:
    def test_path_exists(self, populated_graph):
        path = populated_graph.shortest_path("n1", "n3")
        assert path == ["n1", "n2", "n3"]

    def test_path_not_reachable(self, populated_graph):
        path = populated_graph.shortest_path("n4", "n1")
        assert path == []

    def test_path_src_not_found(self, graph):
        path = graph.shortest_path("ghost", "n1")
        assert path == []

    def test_path_dst_not_found(self, graph):
        graph.add_node(MemoryNode("a", "A"))
        path = graph.shortest_path("a", "ghost")
        assert path == []

    def test_path_same_node(self, graph):
        graph.add_node(MemoryNode("a", "A"))
        path = graph.shortest_path("a", "a")
        assert path == ["a"]


class TestMemoryGraphCounts:
    def test_node_count_empty(self, graph):
        assert graph.node_count() == 0

    def test_node_count(self, populated_graph):
        assert populated_graph.node_count() == 4

    def test_edge_count_empty(self, graph):
        assert graph.edge_count() == 0

    def test_edge_count(self, populated_graph):
        assert populated_graph.edge_count() == 2  # n1->n2, n2->n3 (no direct n1->n3)


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in MEMORY_GRAPH_REGISTRY
        assert MEMORY_GRAPH_REGISTRY["default"] is MemoryGraph

    def test_registry_default_is_instance(self):
        inst = MEMORY_GRAPH_REGISTRY["default"]()
        assert isinstance(inst, MemoryGraph)
