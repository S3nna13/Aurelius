"""Circuit analysis: subgraph discovery, ablation study, composition scores."""

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CircuitNode:
    layer: int
    head: Optional[int]
    component_type: str
    importance: float = 0.0
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class CircuitEdge:
    from_node: str
    to_node: str
    weight: float


@dataclass
class Circuit:
    name: str
    nodes: list[CircuitNode]
    edges: list[CircuitEdge]
    task: str = ""


class CircuitAnalyzer:
    def __init__(self) -> None:
        pass

    def build_circuit(
        self,
        name: str,
        node_specs: list[dict],
        edge_specs: list[dict],
        task: str = "",
    ) -> Circuit:
        nodes = []
        for spec in node_specs:
            node = CircuitNode(
                layer=spec["layer"],
                head=spec.get("head", None),
                component_type=spec["component_type"],
                importance=spec.get("importance", 0.0),
            )
            nodes.append(node)

        edges = []
        for espec in edge_specs:
            from_node = nodes[espec["from"]].node_id
            to_node = nodes[espec["to"]].node_id
            edge = CircuitEdge(
                from_node=from_node,
                to_node=to_node,
                weight=espec["weight"],
            )
            edges.append(edge)

        return Circuit(name=name, nodes=nodes, edges=edges, task=task)

    def ablation_score(self, circuit: Circuit, ablated_nodes: list[str]) -> float:
        if not circuit.nodes:
            return 1.0
        ablated_set = set(ablated_nodes)
        not_ablated = sum(1 for n in circuit.nodes if n.node_id not in ablated_set)
        return not_ablated / len(circuit.nodes)

    def composition_score(self, node_a: CircuitNode, node_b: CircuitNode) -> float:
        ia = node_a.importance
        ib = node_b.importance
        if ia == ib:
            return 0.0
        denom = max(ia, ib, 1e-8)
        return abs(ia - ib) / denom

    def subgraph(self, circuit: Circuit, min_importance: float = 0.5) -> Circuit:
        kept_nodes = [n for n in circuit.nodes if n.importance >= min_importance]
        kept_ids = {n.node_id for n in kept_nodes}
        kept_edges = [
            e for e in circuit.edges
            if e.from_node in kept_ids and e.to_node in kept_ids
        ]
        return Circuit(
            name=circuit.name,
            nodes=kept_nodes,
            edges=kept_edges,
            task=circuit.task,
        )

    def critical_path(self, circuit: Circuit) -> list[CircuitNode]:
        return sorted(circuit.nodes, key=lambda n: n.importance, reverse=True)
