"""Tests for attribution_graphs."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.interpretability.attribution_graphs import (
    AttributionGraph,
    AttributionGraphBuilder,
)


def _tiny_model(in_dim: int = 4, hid: int = 6, out: int = 5) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(in_dim, hid),
        nn.Linear(hid, out),
    )


def _rand_input(batch: int = 2, in_dim: int = 4) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, in_dim)


def test_build_returns_graph_with_expected_nodes():
    m = _tiny_model()
    b = AttributionGraphBuilder(m)
    g = b.build(_rand_input(), target_layer=-1, target_unit=0)
    assert isinstance(g, AttributionGraph)
    # layer 0 has 6 units, layer 1 has 5 units => 11 nodes
    assert len(g.nodes) == 6 + 5
    assert any(n.layer == 0 for n in g.nodes)
    assert any(n.layer == 1 for n in g.nodes)


def test_top_k_per_node_limits_edges():
    m = _tiny_model()
    b = AttributionGraphBuilder(m, top_k_per_node=2)
    g = b.build(_rand_input(), target_layer=1, target_unit=0)
    # 6 src nodes in layer 0, each with top 2 edges => 12 edges
    assert len(g.edges) == 6 * 2


def test_method_input_x_grad_vs_integrated_gradients_differ():
    # Use a nonlinear model so IG path integral differs from input*grad.
    torch.manual_seed(3)
    m = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 5))
    x = _rand_input()
    a1 = AttributionGraphBuilder(m, method="input_x_grad").input_x_grad(x, torch.tensor(0))
    a2 = AttributionGraphBuilder(m, method="integrated_gradients").integrated_gradients(
        x, torch.tensor(0), n_steps=4
    )
    # shapes match, values should differ for a nontrivial nonlinear model
    assert a1.shape == a2.shape
    assert not torch.allclose(a1, a2, atol=1e-6)


def test_ig_converges_to_input_x_grad_for_linear_model():
    # For a purely linear model (no activation), IG == input * grad exactly.
    torch.manual_seed(7)
    m = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    x = torch.randn(1, 3)
    b_ig = AttributionGraphBuilder(m, method="integrated_gradients")
    b_ig_large = b_ig.integrated_gradients(x, torch.tensor(0), n_steps=128)
    b_ixg = AttributionGraphBuilder(m, method="input_x_grad").input_x_grad(x, torch.tensor(0))
    assert torch.allclose(b_ig_large, b_ixg, atol=1e-4)


def test_edge_weights_are_real_valued():
    m = _tiny_model()
    g = AttributionGraphBuilder(m).build(_rand_input())
    for e in g.edges:
        assert isinstance(e.weight, float)
        assert not (e.weight != e.weight)  # not NaN
        assert e.weight != float("inf")


def test_determinism_under_seed():
    torch.manual_seed(42)
    m1 = _tiny_model()
    x = _rand_input()
    torch.manual_seed(123)
    g1 = AttributionGraphBuilder(m1).build(x)
    torch.manual_seed(42)
    m2 = _tiny_model()
    torch.manual_seed(123)
    g2 = AttributionGraphBuilder(m2).build(x)
    assert len(g1.edges) == len(g2.edges)
    for e1, e2 in zip(g1.edges, g2.edges):
        assert e1.weight == pytest.approx(e2.weight, abs=1e-7)


def test_unknown_method_raises():
    m = _tiny_model()
    with pytest.raises(ValueError):
        AttributionGraphBuilder(m, method="bogus")


def test_empty_layer_names_raises():
    m = _tiny_model()
    with pytest.raises(ValueError):
        AttributionGraphBuilder(m, layer_names=[])


def test_target_layer_out_of_range_raises():
    m = _tiny_model()
    b = AttributionGraphBuilder(m)
    with pytest.raises(IndexError):
        b.build(_rand_input(), target_layer=10, target_unit=0)


def test_top_k_greater_than_units_returns_all():
    m = _tiny_model()
    b = AttributionGraphBuilder(m, top_k_per_node=100)
    g = b.build(_rand_input(), target_layer=1, target_unit=0)
    # 6 src * 5 dst (clamped to actual)
    assert len(g.edges) == 6 * 5


def test_graph_is_forward_acyclic():
    m = _tiny_model()
    g = AttributionGraphBuilder(m).build(_rand_input())
    assert g.is_acyclic_forward()
    for e in g.edges:
        assert e.src.layer < e.dst.layer


def test_integration_forward_backward_runs():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))
    # Train a few steps so weights look realistic
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    for _ in range(5):
        x = torch.randn(4, 4)
        y = m(x).sum()
        opt.zero_grad()
        y.backward()
        opt.step()
    g = AttributionGraphBuilder(m, top_k_per_node=3).build(
        torch.randn(2, 4), target_layer=-1, target_unit=1
    )
    assert len(g.nodes) > 0
    assert len(g.edges) > 0


def test_n_steps_nonpositive_raises():
    m = _tiny_model()
    b = AttributionGraphBuilder(m)
    with pytest.raises(ValueError):
        b.integrated_gradients(_rand_input(), torch.tensor(0), n_steps=0)
    with pytest.raises(ValueError):
        b.integrated_gradients(_rand_input(), torch.tensor(0), n_steps=-2)


def test_graph_serialization_roundtrip():
    m = _tiny_model()
    g = AttributionGraphBuilder(m, top_k_per_node=2).build(_rand_input())
    d = g.to_dict()
    g2 = AttributionGraph.from_dict(d)
    assert len(g.nodes) == len(g2.nodes)
    assert len(g.edges) == len(g2.edges)
    for a, b in zip(g.nodes, g2.nodes):
        assert a.layer == b.layer and a.unit == b.unit
        assert a.activation == pytest.approx(b.activation)
    for ea, eb in zip(g.edges, g2.edges):
        assert ea.weight == pytest.approx(eb.weight)
        assert ea.src.layer == eb.src.layer and ea.dst.unit == eb.dst.unit


def test_target_unit_out_of_range_raises():
    m = _tiny_model()
    b = AttributionGraphBuilder(m)
    with pytest.raises(IndexError):
        b.build(_rand_input(), target_layer=-1, target_unit=999)
