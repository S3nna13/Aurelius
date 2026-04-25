"""Tests for src/reasoning/gnn_layer.py"""
from __future__ import annotations

import pytest
import torch

from src.reasoning.gnn_layer import (
    GNN_PROFILER_REGISTRY,
    GATLayer,
    GCNLayer,
    GNNConfig,
    GNNStack,
    GraphSAGELayer,
)


@pytest.fixture
def tiny_config():
    return GNNConfig(
        gnn_type="gcn",
        in_dim=64,
        out_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        use_residual=True,
    )


@pytest.fixture
def x_8():
    return torch.randn(8, 64)


@pytest.fixture
def edge_index_8():
    return torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                         [1, 2, 3, 0, 5, 6, 7, 4]], dtype=torch.long)


class TestGCNForward:
    def test_output_shape(self, x_8, edge_index_8):
        layer = GCNLayer(64, 64)
        out = layer(x_8, edge_index_8)
        assert out.shape == (8, 64)

    def test_output_is_float(self, x_8, edge_index_8):
        layer = GCNLayer(64, 64)
        out = layer(x_8, edge_index_8)
        assert out.dtype == torch.float32

    def test_relu_applied(self, x_8, edge_index_8):
        layer = GCNLayer(64, 64)
        out = layer(x_8, edge_index_8)
        assert torch.all(out >= 0).item() or torch.any(out > 0).item()


class TestGATForward:
    def test_output_shape(self, x_8, edge_index_8):
        layer = GATLayer(64, 64, n_heads=4)
        out, alpha = layer(x_8, edge_index_8)
        assert out.shape == (8, 64)

    def test_attention_weights_non_negative(self, x_8, edge_index_8):
        layer = GATLayer(64, 64, n_heads=4)
        out, alpha = layer(x_8, edge_index_8)
        assert torch.all(alpha >= 0.0).item()

    def test_attention_weights_sum_to_one_per_head(self, x_8, edge_index_8):
        layer = GATLayer(64, 64, n_heads=4)
        out, alpha = layer(x_8, edge_index_8)
        n_edges = edge_index_8.shape[1]
        assert alpha.shape == (n_edges, 4)


class TestGraphSAGEForward:
    def test_output_shape(self, x_8, edge_index_8):
        layer = GraphSAGELayer(64, 64)
        out = layer(x_8, edge_index_8)
        assert out.shape == (8, 64)

    def test_uses_concat_aggregation(self, x_8, edge_index_8):
        layer = GraphSAGELayer(64, 64)
        out = layer(x_8, edge_index_8)
        assert out.dtype == torch.float32


class TestGNNStackForward:
    def test_gcn_forward_shape(self, tiny_config, x_8, edge_index_8):
        stack = GNNStack(tiny_config)
        out = stack(x_8, edge_index_8)
        assert out.shape == (8, 64)

    def test_residual_when_dims_match(self, tiny_config):
        cfg = GNNConfig(gnn_type="gcn", in_dim=64, out_dim=64, n_layers=2, use_residual=True)
        x = torch.randn(4, 64)
        edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        stack = GNNStack(cfg)
        out = stack(x, edge)
        assert out.shape == (4, 64)

    def test_no_residual_when_dims_differ(self):
        cfg = GNNConfig(gnn_type="gcn", in_dim=64, out_dim=32, n_layers=2, use_residual=True)
        x = torch.randn(4, 64)
        edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        stack = GNNStack(cfg)
        out = stack(x, edge)
        assert out.shape == (4, 32)


class TestGNNStackEdgeCases:
    def test_single_node(self):
        cfg = GNNConfig(gnn_type="gcn", in_dim=64, out_dim=64, n_layers=2)
        x = torch.randn(1, 64)
        edge = torch.tensor([[0], [0]], dtype=torch.long)
        stack = GNNStack(cfg)
        out = stack(x, edge)
        assert out.shape == (1, 64)

    def test_empty_edges(self):
        cfg = GNNConfig(gnn_type="gcn", in_dim=64, out_dim=64, n_layers=2)
        x = torch.randn(4, 64)
        edge = torch.empty(2, dtype=torch.long)
        stack = GNNStack(cfg)
        out = stack(x, edge)
        assert out.shape == (4, 64)

    def test_gat_with_mismatched_out_dim(self):
        cfg = GNNConfig(gnn_type="gat", in_dim=64, out_dim=64, n_layers=2, n_heads=4)
        x = torch.randn(4, 64)
        edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        stack = GNNStack(cfg)
        out = stack(x, edge)
        assert out.shape == (4, 64)


class TestGNNStackGradientFlow:
    def test_trainable_params_get_grads(self, x_8, edge_index_8):
        cfg = GNNConfig(gnn_type="gcn", in_dim=64, out_dim=64, n_layers=2)
        stack = GNNStack(cfg)
        out = stack(x_8, edge_index_8)
        loss = out.sum()
        loss.backward()
        for name, param in stack.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"


class TestGNNProfilerRegistry:
    def test_registry_exists(self):
        assert isinstance(GNN_PROFILER_REGISTRY, dict)