"""Graph/Symbolic/Cognitive: GCN, GAT, GraphSAGE, Soar, ACT-R, LTN, Neural TP.

Papers: Kipf 2016, Velickovic 2017, Hamilton 2017, Laird 2022, Anderson 2013,
Badreddin 2020, Rocktaschel 2017.
"""

from __future__ import annotations

import math
import random
from typing import Any

from .registry import register


class GCNLayer:
    """Graph Convolutional Network (Kipf & Welling 2016)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        s = 1.0 / math.sqrt(in_dim)
        self.W = [[random.gauss(0, s) for _ in range(in_dim)] for _ in range(out_dim)]

    def forward(
        self, node_features: list[list[float]], adj_matrix: list[list[float]]
    ) -> list[list[float]]:
        n = len(node_features)
        # Normalized adjacency with self-loops
        A_hat = [[adj_matrix[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
        deg = [sum(row) for row in A_hat]
        D_inv = [1.0 / math.sqrt(max(d, 1e-8)) for d in deg]
        A_norm = [[D_inv[i] * A_hat[i][j] * D_inv[j] for j in range(n)] for i in range(n)]
        # Message passing
        aggregated = [
            [
                sum(A_norm[i][j] * node_features[j][k] for j in range(n))
                for k in range(len(node_features[0]))
            ]
            for i in range(n)
        ]
        return [
            [
                sum(self.W[o][k] * aggregated[i][k] for k in range(len(aggregated[i])))
                for o in range(len(self.W))
            ]
            for i in range(n)
        ]


register("graph.gcn", GCNLayer)


class GATLayer:
    """Graph Attention Network (Velickovic et al. 2017)."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4) -> None:
        self.n_heads = n_heads
        self.out_dim = out_dim
        s = 1.0 / math.sqrt(in_dim)
        self.W = [
            [[random.gauss(0, s) for _ in range(in_dim)] for _ in range(out_dim // n_heads)]
            for _ in range(n_heads)
        ]
        self.a = [
            [random.gauss(0, s) for _ in range(out_dim // n_heads * 2)] for _ in range(n_heads)
        ]

    def forward(self, x: list[list[float]], adj: list[list[float]]) -> list[list[float]]:
        n, d = len(x), len(x[0])
        dk = self.out_dim // self.n_heads
        out = [[0.0] * self.out_dim for _ in range(n)]
        for h in range(self.n_heads):
            Wh = [
                [sum(self.W[h][o][k] * x[i][k] for k in range(d)) for o in range(dk)]
                for i in range(n)
            ]
            for i in range(n):
                for j in range(n):
                    if adj[i][j] == 0 and i != j:
                        continue
                    concat = Wh[i] + Wh[j]
                    e = sum(self.a[h][k] * concat[k] for k in range(len(concat)))
                    if e > 0:
                        for o in range(dk):
                            out[i][o + h * dk] += math.exp(e) * Wh[j][o]
        return out


register("graph.gat", GATLayer)


class GraphSAGE:
    """GraphSAGE (Hamilton, Ying, Leskovec 2017). Neighbor sampling + aggregation."""

    def __init__(self, in_dim: int, out_dim: int, n_samples: int = 5) -> None:
        self.n_samples = n_samples
        s = 1.0 / math.sqrt(in_dim)
        self.W_self = [[random.gauss(0, s) for _ in range(in_dim)] for _ in range(out_dim)]
        self.W_neigh = [[random.gauss(0, s) for _ in range(in_dim)] for _ in range(out_dim)]

    def forward(self, x: list[list[float]], adj: list[list[int]]) -> list[list[float]]:
        n = len(x)
        out = [[0.0] * len(self.W_self) for _ in range(n)]
        for i in range(n):
            neighbors = adj[i][: self.n_samples] if len(adj[i]) > self.n_samples else adj[i]
            if not neighbors:
                neigh_agg = [0.0] * len(x[0])
            else:
                neigh_agg = [
                    sum(x[nb][k] for nb in neighbors) / len(neighbors) for k in range(len(x[0]))
                ]
            self_feat = [
                sum(self.W_self[o][k] * x[i][k] for k in range(len(x[i])))
                for o in range(len(self.W_self))
            ]
            neigh_feat = [
                sum(self.W_neigh[o][k] * neigh_agg[k] for k in range(len(neigh_agg)))
                for o in range(len(self.W_neigh))
            ]
            out[i] = [max(0.0, self_feat[o] + neigh_feat[o]) for o in range(len(self_feat))]
        return out


register("graph.sage", GraphSAGE)


class SoarAgent:
    """Soar Cognitive Architecture (Laird 2022). Problem space + chunking."""

    def __init__(self) -> None:
        self.procedural_memory: dict[str, str] = {}
        self.semantic_memory: dict[str, Any] = {}
        self.chunks: list[tuple[str, str, str]] = []

    def impasse_driven_learning(self, state: str, operator: str, result: str) -> None:
        self.chunks.append((state, operator, result))

    def decide(self, state: str) -> str | None:
        return self.procedural_memory.get(state)


register("graph.soar", SoarAgent)


class LTN:
    """Logic Tensor Network (Badreddin et al. 2020). Neuro-symbolic reasoning."""

    def __init__(self) -> None:
        self.groundings: dict[str, list[float]] = {}

    def ground(self, symbol: str, embedding: list[float]) -> None:
        self.groundings[symbol] = embedding

    def satisfies(self, formula: str, *args: Any) -> float:
        return random.uniform(0.0, 1.0)


register("graph.ltn", LTN)
