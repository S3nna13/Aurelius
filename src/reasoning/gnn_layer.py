"""GNN layers for knowledge graph reasoning.

Implements GCN (Kipf & Welling 2017), GAT (Velickovic 2018),
and GraphSAGE (Hamilton 2017) with a stacking wrapper.
License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNConfig:
    in_dim: int = 64
    out_dim: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    layer_type: str = "gcn"


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer (Kipf & Welling 2017)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(adj @ self.linear(x))


class GATLayer(nn.Module):
    """Graph Attention Network layer (Velickovic 2018)."""

    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.dropout = dropout
        self.linear = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        h = self.linear(x).view(n, self.n_heads, self.out_dim)

        e_src = (h * self.attn_src).sum(dim=-1)
        e_dst = (h * self.attn_dst).sum(dim=-1)

        scores = e_src.unsqueeze(1) + e_dst.unsqueeze(0)
        scores = F.leaky_relu(scores, negative_slope=0.2)

        mask = (adj == 0).unsqueeze(-1).expand_as(scores)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = (attn.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=2)
        return out.mean(dim=1)


class GraphSAGELayer(nn.Module):
    """GraphSAGE: sample-and-aggregate (Hamilton 2017)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neighbor_agg = (adj @ x) / deg
        return F.relu(self.linear(torch.cat([x, neighbor_agg], dim=-1)))


def _build_layer(config: GNNConfig) -> nn.Module:
    if config.layer_type == "gcn":
        return GCNLayer(config.in_dim, config.out_dim)
    if config.layer_type == "gat":
        return GATLayer(config.in_dim, config.out_dim,
                        n_heads=config.n_heads, dropout=config.dropout)
    if config.layer_type == "sage":
        return GraphSAGELayer(config.in_dim, config.out_dim)
    raise ValueError(f"unknown layer_type: {config.layer_type!r}")


class GNNStack(nn.Module):
    """Stack of GNN layers with residual connections."""

    def __init__(self, config: GNNConfig, n_layers: int = 3) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.config = config
        self.layers = nn.ModuleList([_build_layer(config) for _ in range(n_layers)])
        self.use_residual = config.in_dim == config.out_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(x, adj)
            if self.use_residual:
                x = x + h
            else:
                x = h
        return x
