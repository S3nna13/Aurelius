"""Tests for circuit_analyzer module."""

from src.interpretability.circuit_analyzer import (
    Circuit,
    CircuitAnalyzer,
    CircuitEdge,
    CircuitNode,
)

# ── CircuitNode ─────────────────────────────────────────────────────────────────


def test_circuit_node_auto_generates_id():
    node = CircuitNode(layer=0, head=None, component_type="mlp")
    assert node.node_id is not None
    assert isinstance(node.node_id, str)
    assert len(node.node_id) == 8


def test_circuit_node_ids_unique():
    node_a = CircuitNode(layer=0, head=None, component_type="mlp")
    node_b = CircuitNode(layer=0, head=None, component_type="mlp")
    assert node_a.node_id != node_b.node_id


def test_circuit_node_fields():
    node = CircuitNode(layer=2, head=3, component_type="attn", importance=0.7)
    assert node.layer == 2
    assert node.head == 3
    assert node.component_type == "attn"
    assert node.importance == 0.7


def test_circuit_node_default_importance():
    node = CircuitNode(layer=0, head=None, component_type="mlp")
    assert node.importance == 0.0


def test_circuit_node_head_none():
    node = CircuitNode(layer=0, head=None, component_type="mlp")
    assert node.head is None


def test_circuit_node_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(CircuitNode)


# ── CircuitEdge ─────────────────────────────────────────────────────────────────


def test_circuit_edge_fields():
    edge = CircuitEdge(from_node="aabbccdd", to_node="11223344", weight=0.5)
    assert edge.from_node == "aabbccdd"
    assert edge.to_node == "11223344"
    assert edge.weight == 0.5


def test_circuit_edge_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(CircuitEdge)


# ── Circuit ─────────────────────────────────────────────────────────────────────


def test_circuit_fields():
    c = Circuit(name="test", nodes=[], edges=[], task="sentiment")
    assert c.name == "test"
    assert c.nodes == []
    assert c.edges == []
    assert c.task == "sentiment"


def test_circuit_default_task():
    c = Circuit(name="test", nodes=[], edges=[])
    assert c.task == ""


def test_circuit_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(Circuit)


# ── CircuitAnalyzer ─────────────────────────────────────────────────────────────


def make_node_specs(n=3):
    return [
        {
            "layer": i,
            "head": i if i % 2 == 0 else None,
            "component_type": "attn" if i % 2 == 0 else "mlp",
            "importance": 0.3 * (i + 1),
        }
        for i in range(n)
    ]


def make_edge_specs(n_nodes=3):
    return (
        [
            {"from": 0, "to": 1, "weight": 0.5},
            {"from": 1, "to": 2, "weight": 0.8},
        ]
        if n_nodes >= 3
        else []
    )


def test_build_circuit_returns_circuit():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(), make_edge_specs())
    assert isinstance(c, Circuit)


def test_build_circuit_node_count():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(4), [])
    assert len(c.nodes) == 4


def test_build_circuit_edge_count():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(3), make_edge_specs(3))
    assert len(c.edges) == 2


def test_build_circuit_node_ids_assigned():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(3), [])
    for node in c.nodes:
        assert len(node.node_id) == 8


def test_build_circuit_edge_uses_node_ids():
    ca = CircuitAnalyzer()
    specs = make_node_specs(3)
    c = ca.build_circuit("test", specs, make_edge_specs(3))
    node_ids = {n.node_id for n in c.nodes}
    for edge in c.edges:
        assert edge.from_node in node_ids
        assert edge.to_node in node_ids


def test_build_circuit_name():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("my_circuit", [], [])
    assert c.name == "my_circuit"


def test_build_circuit_empty():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("empty", [], [])
    assert len(c.nodes) == 0
    assert len(c.edges) == 0


# ── ablation_score ──────────────────────────────────────────────────────────────


def test_ablation_score_no_ablation():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(3), [])
    score = ca.ablation_score(c, [])
    assert score == 1.0


def test_ablation_score_all_ablated():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(3), [])
    all_ids = [n.node_id for n in c.nodes]
    score = ca.ablation_score(c, all_ids)
    assert score == 0.0


def test_ablation_score_partial():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(4), [])
    ablated = [c.nodes[0].node_id]  # ablate 1 of 4
    score = ca.ablation_score(c, ablated)
    assert abs(score - 0.75) < 1e-9


def test_ablation_score_empty_circuit():
    ca = CircuitAnalyzer()
    c = Circuit(name="empty", nodes=[], edges=[])
    score = ca.ablation_score(c, [])
    assert score == 1.0


# ── composition_score ───────────────────────────────────────────────────────────


def test_composition_score_same_importance():
    ca = CircuitAnalyzer()
    a = CircuitNode(layer=0, head=None, component_type="mlp", importance=0.5)
    b = CircuitNode(layer=1, head=None, component_type="mlp", importance=0.5)
    assert ca.composition_score(a, b) == 0.0


def test_composition_score_different_importance():
    ca = CircuitAnalyzer()
    a = CircuitNode(layer=0, head=None, component_type="mlp", importance=0.2)
    b = CircuitNode(layer=1, head=None, component_type="mlp", importance=0.8)
    score = ca.composition_score(a, b)
    assert score > 0.0


def test_composition_score_formula():
    ca = CircuitAnalyzer()
    a = CircuitNode(layer=0, head=None, component_type="mlp", importance=0.2)
    b = CircuitNode(layer=1, head=None, component_type="mlp", importance=0.8)
    expected = abs(0.2 - 0.8) / max(0.2, 0.8, 1e-8)
    assert abs(ca.composition_score(a, b) - expected) < 1e-9


def test_composition_score_zero_importances():
    ca = CircuitAnalyzer()
    a = CircuitNode(layer=0, head=None, component_type="mlp", importance=0.0)
    b = CircuitNode(layer=1, head=None, component_type="mlp", importance=0.0)
    assert ca.composition_score(a, b) == 0.0


# ── subgraph ────────────────────────────────────────────────────────────────────


def test_subgraph_filters_by_min_importance():
    ca = CircuitAnalyzer()
    node_specs = [
        {"layer": 0, "head": None, "component_type": "mlp", "importance": 0.3},
        {"layer": 1, "head": None, "component_type": "mlp", "importance": 0.7},
        {"layer": 2, "head": None, "component_type": "mlp", "importance": 0.5},
    ]
    c = ca.build_circuit("test", node_specs, [])
    sub = ca.subgraph(c, min_importance=0.5)
    assert len(sub.nodes) == 2
    for n in sub.nodes:
        assert n.importance >= 0.5


def test_subgraph_returns_circuit():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(3), [])
    sub = ca.subgraph(c)
    assert isinstance(sub, Circuit)


def test_subgraph_edges_only_between_remaining_nodes():
    ca = CircuitAnalyzer()
    node_specs = [
        {"layer": 0, "head": None, "component_type": "mlp", "importance": 0.3},
        {"layer": 1, "head": None, "component_type": "mlp", "importance": 0.7},
        {"layer": 2, "head": None, "component_type": "mlp", "importance": 0.8},
    ]
    # Edge from node0 (low imp) to node1 (high), and node1 to node2
    edge_specs = [
        {"from": 0, "to": 1, "weight": 0.5},
        {"from": 1, "to": 2, "weight": 0.6},
    ]
    c = ca.build_circuit("test", node_specs, edge_specs)
    sub = ca.subgraph(c, min_importance=0.5)
    kept_ids = {n.node_id for n in sub.nodes}
    for edge in sub.edges:
        assert edge.from_node in kept_ids
        assert edge.to_node in kept_ids


def test_subgraph_all_below_threshold():
    ca = CircuitAnalyzer()
    node_specs = [
        {"layer": 0, "head": None, "component_type": "mlp", "importance": 0.1},
        {"layer": 1, "head": None, "component_type": "mlp", "importance": 0.2},
    ]
    c = ca.build_circuit("test", node_specs, [])
    sub = ca.subgraph(c, min_importance=0.5)
    assert len(sub.nodes) == 0


def test_subgraph_preserves_name():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("named_circuit", make_node_specs(2), [])
    sub = ca.subgraph(c)
    assert sub.name == "named_circuit"


# ── critical_path ───────────────────────────────────────────────────────────────


def test_critical_path_sorted_desc():
    ca = CircuitAnalyzer()
    node_specs = [
        {"layer": 0, "head": None, "component_type": "mlp", "importance": 0.3},
        {"layer": 1, "head": None, "component_type": "mlp", "importance": 0.9},
        {"layer": 2, "head": None, "component_type": "mlp", "importance": 0.5},
    ]
    c = ca.build_circuit("test", node_specs, [])
    path = ca.critical_path(c)
    importances = [n.importance for n in path]
    assert importances == sorted(importances, reverse=True)


def test_critical_path_returns_list():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(3), [])
    path = ca.critical_path(c)
    assert isinstance(path, list)


def test_critical_path_all_nodes_present():
    ca = CircuitAnalyzer()
    c = ca.build_circuit("test", make_node_specs(3), [])
    path = ca.critical_path(c)
    assert len(path) == len(c.nodes)


def test_critical_path_empty_circuit():
    ca = CircuitAnalyzer()
    c = Circuit(name="empty", nodes=[], edges=[])
    path = ca.critical_path(c)
    assert path == []


def test_critical_path_first_is_most_important():
    ca = CircuitAnalyzer()
    node_specs = [
        {"layer": 0, "head": None, "component_type": "mlp", "importance": 0.1},
        {"layer": 1, "head": None, "component_type": "mlp", "importance": 0.95},
        {"layer": 2, "head": None, "component_type": "mlp", "importance": 0.4},
    ]
    c = ca.build_circuit("test", node_specs, [])
    path = ca.critical_path(c)
    assert path[0].importance == 0.95
