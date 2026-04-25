"""Tests for GNN layers (GCN, GAT, GraphSAGE)."""
from __future__ import annotations

import torch

from src.reasoning.gnn_layer import (
    GATLayer,
    GCNLayer,
    GNNConfig,
    GNNStack,
    GraphSAGELayer,
)


def test_gcn_output_shape(x_8, edge_index_8):
    layer = GCNLayer(64, 64)
    out = layer(x_8, edge_index_8)
    if not (isinstance(out, torch.Tensor) and out.shape == (8, 64)):
        raise ValueError(f"Expected shape (8, 64), got {out.shape}")


def test_gat_output_shape(x_8, edge_index_8):
    layer = GATLayer(64, 64, n_heads=4)
    out, alpha = layer(x_8, edge_index_8)
    if not (isinstance(out, torch.Tensor) and out.shape == (8, 64)):
        raise ValueError(f"Expected shape (8, 64), got {out.shape}")
    if not (isinstance(alpha, torch.Tensor) and alpha.ndim > 0):
        raise ValueError(f"Expected attention tensor, got {type(alpha)}")


def test_gat_attention_nonnegative(x_8, edge_index_8):
    layer = GATLayer(64, 64, n_heads=4)
    _, alpha = layer(x_8, edge_index_8)
    if (alpha < 0).any():
        raise ValueError("Attention weights must be non-negative")


def test_graphsage_output_shape(x_8, edge_index_8):
    layer = GraphSAGELayer(64, 64)
    out = layer(x_8, edge_index_8)
    if not (isinstance(out, torch.Tensor) and out.shape == (8, 64)):
        raise ValueError(f"Expected shape (8, 64), got {out.shape}")


def test_gnn_stack_forward(x_8, edge_index_8):
    cfg = GNNConfig(in_dim=64, out_dim=64, n_layers=2, gnn_type="gcn")
    stack = GNNStack(cfg)
    out = stack(x_8, edge_index_8)
    if not (isinstance(out, torch.Tensor) and out.shape == (8, 64)):
        raise ValueError(f"Expected shape (8, 64), got {out.shape}")


def test_gnn_stack_residual(x_8, edge_index_8):
    cfg = GNNConfig(in_dim=64, out_dim=64, n_layers=3, use_residual=True)
    stack = GNNStack(cfg)
    out = stack(x_8, edge_index_8)
    if not (isinstance(out, torch.Tensor) and out.shape == (8, 64)):
        raise ValueError(f"Expected shape (8, 64), got {out.shape}")


def test_gnn_edge_n1():
    x = torch.randn(1, 64)
    edge = torch.tensor([[0], [0]], dtype=torch.long)
    layer = GCNLayer(64, 64)
    out = layer(x, edge)
    if not (isinstance(out, torch.Tensor) and out.shape == (1, 64)):
        raise ValueError(f"Expected shape (1, 64), got {out.shape}")


def test_gnn_empty_edges():
    x = torch.randn(4, 64)
    edge = torch.tensor([], dtype=torch.long).reshape(0, 2)
    layer = GCNLayer(64, 64)
    out = layer(x, edge)
    if not (isinstance(out, torch.Tensor) and out.shape == (4, 64)):
        raise ValueError(f"Expected shape (4, 64), got {out.shape}")


def test_gat_valueerror_mismatched():
    caught = False
    try:
        GATLayer(64, 63, n_heads=4)
    except ValueError:
        caught = True
    if not caught:
        raise ValueError("Expected ValueError for mismatched out_dim")


def test_gradient_flow(x_8, edge_index_8):
    layer = GCNLayer(64, 64, dropout=0.0)
    x = x_8.clone().requires_grad_(True)
    out = layer(x, edge_index_8)
    out.sum().backward()
    if x.grad is None:
        raise ValueError("Gradient not flowing to input")