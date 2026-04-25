"""GNN layers for graph representation learning (GCN, GAT, GraphSAGE)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNConfig:
    gnn_type: str = "gcn"
    in_dim: int = 64
    out_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    use_residual: bool = True
    activation: str = "relu"

    def __init__(
        self,
        gnn_type: str = "gcn",
        in_dim: int = 64,
        out_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_residual: bool = True,
        activation: str = "relu",
    ) -> None:
        self.gnn_type = gnn_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.activation = activation
        if self.out_dim % self.n_heads != 0:
            raise ValueError("out_dim must be divisible by n_heads")


GNN_PROFILER_REGISTRY: dict[str, object] = {}


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        edge_index = edge_index.to(x.device)
        x = self.linear(x)
        if edge_index.numel() == 0:
            return self.dropout(F.relu(x))
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg = deg.pow(-0.5)
        deg[deg == float("inf")] = 0.0
        norm = deg[row] * deg[col]
        adj = torch.zeros(n, n, device=x.device)
        adj[edge_index[0], edge_index[1]] = norm
        x = adj @ x
        x = F.relu(x)
        return self.dropout(x)


class GATLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        if out_dim % n_heads != 0:
            raise ValueError("out_dim must be divisible by n_heads")
        self.W = nn.Linear(in_dim, n_heads * self.head_dim)
        self.att = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att.data)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(n_heads * self.head_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        device = x.device
        edge_index = edge_index.to(device)
        row, col = edge_index[0], edge_index[1]
        x_proj = self.W(x).view(n, self.n_heads, self.head_dim)
        h_src = x_proj[row]
        h_dst = x_proj[col]
        alpha = torch.cat([h_src, h_dst], dim=-1)
        alpha = (alpha * self.att.view(1, self.n_heads, 2 * self.head_dim)).sum(-1)
        alpha = F.softmax(alpha, dim=0)
        alpha_exp = F.dropout(alpha, p=0.0, training=self.training)
        h_prime = h_src * alpha_exp.unsqueeze(-1)
        out = torch.zeros(n, self.n_heads, self.head_dim, device=device)
        out.scatter_add_(0, col.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim), h_prime)
        out = out.view(n, -1)
        out = self.out_proj(out)
        out = F.elu(out)
        return self.dropout(out), alpha


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        device = x.device
        edge_index = edge_index.to(device)
        row, col = edge_index[0], edge_index[1]
        neigh = torch.zeros_like(x)
        count = torch.zeros(n, device=device)
        neigh.scatter_add_(0, col.unsqueeze(1).expand_as(x), x[row])
        count.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
        count[count == 0] = 1.0
        neigh = neigh / count.unsqueeze(1)
        x_aggr = torch.cat([x, neigh], dim=-1)
        out = self.linear(x_aggr)
        out = F.relu(out)
        return self.dropout(out)


class GNNStack(nn.Module):
    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        in_d = config.in_dim
        for i in range(config.n_layers):
            out_d = config.out_dim
            if config.gnn_type == "gcn":
                layers.append(GCNLayer(in_d, out_d, config.dropout))
            elif config.gnn_type == "gat":
                layers.append(GATLayer(in_d, out_d, config.n_heads, config.dropout))
            else:
                layers.append(GraphSAGELayer(in_d, out_d, config.dropout))
            in_d = out_d
        self.layers = nn.ModuleList(layers)
        self.use_residual = config.use_residual
        self.activation = config.activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            prev = h
            if isinstance(layer, GATLayer):
                h, _ = layer(h, edge_index)
            else:
                h = layer(h, edge_index)
            if self.use_residual and h.shape == prev.shape and i < len(self.layers) - 1:
                h = h + prev
        return h