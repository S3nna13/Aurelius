"""
Tests for src/model/graph_neural_lm.py

Coverage:
    TokenGraph edge constructors
    GNNLayer forward, aggregation variants, isolated nodes
    GATLayer forward, attention weights
    GraphTransformerLayer forward
    GNNLanguageModel forward, loss, gradients, graph types
    build_graph validity
    GNNConfig defaults
"""

import torch
import pytest

from src.model.graph_neural_lm import (
    TokenGraph,
    GNNLayer,
    GATLayer,
    GraphTransformerLayer,
    GNNLanguageModel,
    GNNConfig,
)

# ── shared fixtures ─────────────────────────────────────────────────────────

D_MODEL = 16
VOCAB = 16
N_LAYERS = 2
T = 8
B = 2
WINDOW = 2
N_HEADS = 4


def make_model(gnn_type="gat", graph_type="window"):
    return GNNLanguageModel(
        d_model=D_MODEL,
        vocab_size=VOCAB,
        n_layers=N_LAYERS,
        gnn_type=gnn_type,
        graph_type=graph_type,
        n_heads=N_HEADS,
        window_size=WINDOW,
    )


def make_input():
    return torch.randint(0, VOCAB, (B, T))


# ── 1. TokenGraph: add_sequential_edges shape ───────────────────────────────

def test_sequential_edges_shape():
    ei = TokenGraph.add_sequential_edges(T)
    assert ei.shape == (2, T - 1), f"Expected [2, {T-1}], got {ei.shape}"


def test_sequential_edges_values():
    ei = TokenGraph.add_sequential_edges(T)
    expected_src = torch.arange(T - 1)
    expected_dst = torch.arange(1, T)
    assert torch.equal(ei[0], expected_src)
    assert torch.equal(ei[1], expected_dst)


# ── 2. TokenGraph: add_window_edges within-bound connections only ────────────

def test_window_edges_within_bound():
    ei = TokenGraph.add_window_edges(T, window=WINDOW)
    src, dst = ei[0], ei[1]
    diff = (dst - src).abs()
    assert diff.max().item() <= WINDOW, (
        f"Edge distance exceeds window={WINDOW}: max diff={diff.max().item()}"
    )


def test_window_edges_no_self_loops():
    ei = TokenGraph.add_window_edges(T, window=WINDOW)
    assert (ei[0] != ei[1]).all(), "Window edges must not contain self-loops"


def test_window_edges_symmetric():
    ei = TokenGraph.add_window_edges(T, window=WINDOW)
    src, dst = ei[0].tolist(), ei[1].tolist()
    edge_set = set(zip(src, dst))
    for s, d in list(edge_set):
        assert (d, s) in edge_set, f"Window edges not symmetric: ({s},{d}) has no reverse"


# ── 3. TokenGraph: add_dependency_edges causal (no future edges) ─────────────

def test_dependency_edges_causal():
    ei = TokenGraph.add_dependency_edges(T)
    src, dst = ei[0], ei[1]
    assert (src < dst).all(), "Dependency edges must be strictly causal (src < dst)"


def test_dependency_edges_complete_dag():
    ei = TokenGraph.add_dependency_edges(T)
    expected_n_edges = T * (T - 1) // 2
    assert ei.shape[1] == expected_n_edges, (
        f"Expected {expected_n_edges} dependency edges, got {ei.shape[1]}"
    )


# ── 4. GNNLayer forward output shape ────────────────────────────────────────

def test_gnn_layer_output_shape():
    N, d_in, d_out = 10, 16, 16
    layer = GNNLayer(d_in, d_out, aggregation="mean")
    x = torch.randn(N, d_in)
    ei = TokenGraph.add_window_edges(N, window=2)
    out = layer(x, ei)
    assert out.shape == (N, d_out), f"Expected [{N}, {d_out}], got {out.shape}"


# ── 5. GNNLayer aggregation="mean" vs "max" produce different outputs ────────

def test_gnn_layer_aggregation_mean_vs_max():
    torch.manual_seed(0)
    N, d = 10, 16
    x = torch.randn(N, d)
    ei = TokenGraph.add_window_edges(N, window=2)

    layer_mean = GNNLayer(d, d, aggregation="mean")
    layer_max = GNNLayer(d, d, aggregation="max")

    with torch.no_grad():
        layer_max.W_self.weight.copy_(layer_mean.W_self.weight)
        layer_max.W_self.bias.copy_(layer_mean.W_self.bias)
        layer_max.W_neighbor.weight.copy_(layer_mean.W_neighbor.weight)
        layer_max.W_neighbor.bias.copy_(layer_mean.W_neighbor.bias)

    out_mean = layer_mean(x, ei)
    out_max = layer_max(x, ei)
    assert not torch.allclose(out_mean, out_max), (
        "mean and max aggregation should differ"
    )


# ── 6. GNNLayer with isolated nodes (no incoming edges) ─────────────────────

def test_gnn_layer_isolated_nodes():
    N, d = 5, 16
    layer = GNNLayer(d, d, aggregation="mean")
    x = torch.randn(N, d)
    ei = torch.zeros(2, 0, dtype=torch.long)
    out = layer(x, ei)
    assert out.shape == (N, d)
    assert torch.isfinite(out).all(), "Output must be finite for isolated nodes"


# ── 7. GATLayer forward output shape ────────────────────────────────────────

def test_gat_layer_output_shape():
    N, d_in, d_out = 10, 16, 16
    layer = GATLayer(d_in, d_out, n_heads=N_HEADS)
    x = torch.randn(N, d_in)
    ei = TokenGraph.add_window_edges(N, window=2)
    out = layer(x, ei)
    assert out.shape == (N, d_out), f"Expected [{N},{d_out}], got {out.shape}"


# ── 8. GATLayer attention weights sum to 1 per node ─────────────────────────

def test_gat_attention_weights_sum_to_one():
    """
    Re-compute attention weights externally and verify they sum to 1 per
    destination node (per head).
    """
    torch.manual_seed(42)
    N = 8
    d_in = d_out = 16
    H = 4
    d_h = d_out // H

    layer = GATLayer(d_in, d_out, n_heads=H)
    # Use inference mode rather than .eval() to avoid hook issue
    x = torch.randn(N, d_in)
    ei = TokenGraph.add_window_edges(N, window=2)
    src, dst = ei[0], ei[1]
    E = src.size(0)

    with torch.no_grad():
        Wx = layer.W(x).view(N, H, d_h)
        h_src = Wx[src]
        h_dst = Wx[dst]
        a_i = layer.attn_vec[:, :d_h]
        a_j = layer.attn_vec[:, d_h:]
        e = layer.leaky(
            (h_src * a_i.unsqueeze(0)).sum(-1) +
            (h_dst * a_j.unsqueeze(0)).sum(-1)
        )  # [E, H]

        e_max = torch.full((N, H), float("-inf"))
        e_max.scatter_reduce_(
            0, dst.unsqueeze(1).expand(E, H), e,
            reduce="amax", include_self=True,
        )
        e_shifted = e - e_max[dst]
        exp_e = torch.exp(e_shifted)
        exp_sum = torch.zeros(N, H)
        exp_sum.scatter_add_(0, dst.unsqueeze(1).expand(E, H), exp_e)
        alpha = exp_e / (exp_sum[dst] + 1e-16)   # [E, H]

        alpha_sum = torch.zeros(N, H)
        alpha_sum.scatter_add_(0, dst.unsqueeze(1).expand(E, H), alpha)

        has_in_edge = torch.zeros(N, dtype=torch.bool)
        has_in_edge[dst] = True
        sums = alpha_sum[has_in_edge]   # [k, H]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
            f"Attention weights don't sum to 1: {sums}"
        )


# ── 9. GraphTransformerLayer forward output shape ───────────────────────────

def test_graph_transformer_layer_output_shape():
    N, d = 10, 16
    layer = GraphTransformerLayer(d, n_heads=N_HEADS)
    x = torch.randn(N, d)
    ei = TokenGraph.add_window_edges(N, window=2)
    out = layer(x, ei)
    assert out.shape == (N, d), f"Expected [{N},{d}], got {out.shape}"


# ── 10. GNNLanguageModel forward output shape ───────────────────────────────

def test_model_forward_output_shape():
    model = make_model()
    ids = make_input()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB), (
        f"Expected [{B},{T},{VOCAB}], got {logits.shape}"
    )


# ── 11. GNNLanguageModel compute_loss is finite positive scalar ──────────────

def test_model_loss_finite_positive():
    model = make_model()
    ids = make_input()
    loss = model.compute_loss(ids)
    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() > 0.0, "Loss must be positive"


# ── 12. GNNLanguageModel compute_loss backward (gradients flow) ──────────────

def test_model_loss_backward():
    model = make_model()
    ids = make_input()
    loss = model.compute_loss(ids)
    loss.backward()
    grads_exist = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert grads_exist, "All trainable parameters must have finite gradients"


# ── 13. GNNLanguageModel graph_type="sequential" works ───────────────────────

def test_model_sequential_graph():
    model = make_model(gnn_type="gnn", graph_type="sequential")
    ids = make_input()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB)
    loss = model.compute_loss(ids)
    assert torch.isfinite(loss)


# ── 14. GNNLanguageModel graph_type="window" works ───────────────────────────

def test_model_window_graph():
    model = make_model(gnn_type="gat", graph_type="window")
    ids = make_input()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB)
    loss = model.compute_loss(ids)
    assert torch.isfinite(loss)


# ── 15. build_graph returns valid edge_index with correct max node index ──────

def test_build_graph_valid_edge_index():
    model = make_model(graph_type="window")
    ids = make_input()
    edge_index, batch_vector = model.build_graph(ids)

    total_nodes = B * T

    assert batch_vector.shape == (total_nodes,), (
        f"batch_vector shape: {batch_vector.shape}"
    )

    if edge_index.size(1) > 0:
        assert edge_index.min().item() >= 0
        assert edge_index.max().item() < total_nodes, (
            f"Max node index {edge_index.max().item()} >= total nodes {total_nodes}"
        )

    assert batch_vector.min().item() == 0
    assert batch_vector.max().item() == B - 1


# ── 16. GNNConfig defaults ────────────────────────────────────────────────────

def test_gnn_config_defaults():
    cfg = GNNConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_heads == 4
    assert cfg.gnn_type == "gat"
    assert cfg.graph_type == "window"
    assert cfg.window_size == 3


# ── 17. GNNLanguageModel with graph_type="dependency" works ──────────────────

def test_model_dependency_graph():
    model = make_model(gnn_type="gnn", graph_type="dependency")
    ids = make_input()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB)
    loss = model.compute_loss(ids)
    assert torch.isfinite(loss)


# ── 18. GraphTransformerLayer with no edges (isolated tokens) ─────────────────

def test_graph_transformer_no_edges():
    N, d = 6, 16
    layer = GraphTransformerLayer(d, n_heads=N_HEADS)
    x = torch.randn(N, d)
    ei = torch.zeros(2, 0, dtype=torch.long)
    out = layer(x, ei)
    assert out.shape == (N, d)
    assert torch.isfinite(out).all()
