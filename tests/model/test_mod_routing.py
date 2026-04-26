"""Tests for src/model/mod_routing.py — MoD with adaptive token routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.mod_routing import (
    MoDLayer,
    MoDRoutingConfig,
    MoDTransformer,
    TokenImportanceRouter,
    analyze_routing_patterns,
    compute_load_balance_loss,
    route_tokens,
    scatter_back,
)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
N_LAYERS = 2
CAPACITY_FACTOR = 0.5
B, T = 2, 8


def make_config(**kwargs) -> MoDRoutingConfig:
    defaults = dict(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        capacity_factor=CAPACITY_FACTOR,
        router_type="learned",
        load_balance_weight=0.01,
        straight_through=True,
    )
    defaults.update(kwargs)
    return MoDRoutingConfig(**defaults)


def make_hidden() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D_MODEL)


def make_proxy_layer() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(D_MODEL, D_MODEL)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mod_routing_config_defaults():
    cfg = MoDRoutingConfig()
    assert cfg.capacity_factor == 0.5
    assert cfg.n_layers == 4
    assert cfg.d_model == 512
    assert cfg.router_type == "learned"
    assert cfg.load_balance_weight == 0.01
    assert cfg.straight_through is True


def test_token_importance_router_learned_shape():
    torch.manual_seed(0)
    router = TokenImportanceRouter(D_MODEL, router_type="learned")
    hidden = make_hidden()
    scores = router(hidden)
    assert scores.shape == (B, T), f"Expected ({B}, {T}), got {scores.shape}"


def test_token_importance_router_random_shape():
    torch.manual_seed(0)
    router = TokenImportanceRouter(D_MODEL, router_type="random")
    hidden = make_hidden()
    scores = router(hidden)
    assert scores.shape == (B, T), f"Expected ({B}, {T}), got {scores.shape}"


def test_route_tokens_selected_count():
    torch.manual_seed(0)
    hidden = make_hidden()
    scores = torch.randn(B, T)
    capacity = int(B * T * CAPACITY_FACTOR)
    selected_hidden, selected_indices, routing_weights = route_tokens(hidden, scores, capacity)
    assert selected_hidden.shape[0] == capacity, (
        f"Expected {capacity} selected tokens, got {selected_hidden.shape[0]}"
    )
    assert selected_indices.shape[0] == capacity
    assert routing_weights.shape[0] == capacity


def test_route_tokens_indices_valid():
    torch.manual_seed(0)
    hidden = make_hidden()
    scores = torch.randn(B, T)
    capacity = int(B * T * CAPACITY_FACTOR)
    _, selected_indices, _ = route_tokens(hidden, scores, capacity)
    BT = B * T
    assert (selected_indices >= 0).all(), "Negative index found"
    assert (selected_indices < BT).all(), f"Index >= {BT} found"


def test_scatter_back_shape():
    torch.manual_seed(0)
    hidden = make_hidden()
    scores = torch.randn(B, T)
    capacity = int(B * T * CAPACITY_FACTOR)
    selected_hidden, selected_indices, routing_weights = route_tokens(hidden, scores, capacity)
    # Use a simple identity transform as processed output
    processed = selected_hidden.detach().clone()
    output = scatter_back(processed, selected_indices, routing_weights, hidden)
    assert output.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {output.shape}"


def test_scatter_back_non_selected_unchanged():
    torch.manual_seed(0)
    hidden = make_hidden()
    scores = torch.randn(B, T)
    capacity = int(B * T * CAPACITY_FACTOR)
    selected_hidden, selected_indices, routing_weights = route_tokens(
        hidden, scores, capacity, straight_through=False
    )

    # Simulate processing (use same hidden as output so we can check non-selected)
    processed = selected_hidden.detach().clone()
    output = scatter_back(
        processed, selected_indices, routing_weights, hidden, straight_through=False
    )

    BT = B * T
    flat_hidden = hidden.reshape(BT, D_MODEL)
    flat_output = output.reshape(BT, D_MODEL)

    # Build mask of non-selected positions
    mask = torch.ones(BT, dtype=torch.bool)
    mask[selected_indices] = False

    assert torch.allclose(flat_output[mask], flat_hidden[mask]), (
        "Non-selected positions should retain original hidden states"
    )


def test_load_balance_loss_scalar():
    torch.manual_seed(0)
    scores = torch.randn(B, T)
    loss = compute_load_balance_loss(scores, CAPACITY_FACTOR)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, "Load balance loss should be non-negative"


def test_mod_layer_output_shape():
    torch.manual_seed(0)
    cfg = make_config()
    layer = make_proxy_layer()
    mod = MoDLayer(layer, cfg)
    hidden = make_hidden()
    output, lb_loss = mod(hidden)
    assert output.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {output.shape}"
    assert lb_loss.shape == (), f"Load balance loss should be scalar, got {lb_loss.shape}"


def test_mod_layer_gradient_flow():
    torch.manual_seed(0)
    cfg = make_config()
    layer = make_proxy_layer()
    mod = MoDLayer(layer, cfg)
    hidden = make_hidden().requires_grad_(True)
    output, lb_loss = mod(hidden)
    loss = output.sum() + lb_loss
    loss.backward()
    assert hidden.grad is not None, "Gradient should flow back to input hidden"
    assert not torch.all(hidden.grad == 0), "Gradient should be non-zero"


def test_mod_transformer_output_shape():
    torch.manual_seed(0)
    cfg = make_config()
    layers = [make_proxy_layer() for _ in range(N_LAYERS)]
    transformer = MoDTransformer(layers, cfg)
    hidden = make_hidden()
    output, total_lb_loss = transformer(hidden)
    assert output.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {output.shape}"
    assert total_lb_loss.shape == (), (
        f"Total load balance loss should be scalar, got {total_lb_loss.shape}"
    )


def test_analyze_routing_patterns_keys():
    torch.manual_seed(0)
    # Create a list of (B, T) boolean routing decision tensors
    decisions = [
        torch.rand(B, T) > 0.5  # random bool tensors
        for _ in range(5)
    ]
    result = analyze_routing_patterns(decisions)
    assert "mean_utilization" in result, "Missing 'mean_utilization' key"
    assert "routing_variance" in result, "Missing 'routing_variance' key"
    assert "consistency" in result, "Missing 'consistency' key"
    # All values should be floats in [0, 1]
    for key, val in result.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1] range"
