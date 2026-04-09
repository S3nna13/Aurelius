"""Tests for src/model/mixture_of_depths_v2.py -- Mixture of Depths v2."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.mixture_of_depths_v2 import (
    MoDv2Config,
    TokenRouter,
    MoDLayerV2,
    select_top_k_tokens,
    compute_router_z_loss,
    compute_capacity_utilization,
    build_mod_v2_model,
)

# Common test dimensions
B, T, D = 2, 8, 64


def test_modv2_config_defaults():
    config = MoDv2Config()
    assert config.capacity_factor == 1.25
    assert config.router_z_loss_coeff == 1e-3
    assert config.router_aux_loss_coeff == 1e-2
    assert config.top_k == 1
    assert config.use_sigmoid_router is False


def test_token_router_output_shapes():
    router = TokenRouter(d_model=D, use_sigmoid=False)
    x = torch.randn(B, T, D)
    routing_weights, routing_logits = router(x)
    assert routing_weights.shape == (B, T, 1), (
        f"Expected routing_weights shape ({B}, {T}, 1), got {routing_weights.shape}"
    )
    assert routing_logits.shape == (B, T, 1), (
        f"Expected routing_logits shape ({B}, {T}, 1), got {routing_logits.shape}"
    )


def test_token_router_sigmoid_weights_in_range():
    router = TokenRouter(d_model=D, use_sigmoid=True)
    x = torch.randn(B, T, D)
    routing_weights, _ = router(x)
    assert routing_weights.shape == (B, T, 1)
    assert (routing_weights >= 0.0).all(), "Sigmoid weights must be >= 0"
    assert (routing_weights <= 1.0).all(), "Sigmoid weights must be <= 1"


def test_select_top_k_tokens_output_shape():
    k = 3
    capacity = 5
    routing_weights = torch.rand(B, T, 1)
    selected_indices, selected_weights = select_top_k_tokens(routing_weights, k, capacity)
    expected_n = min(k, capacity)
    assert selected_indices.shape == (B, expected_n), (
        f"Expected selected_indices shape ({B}, {expected_n}), got {selected_indices.shape}"
    )
    assert selected_weights.shape == (B, expected_n), (
        f"Expected selected_weights shape ({B}, {expected_n}), got {selected_weights.shape}"
    )


def test_select_top_k_tokens_indices_valid_range():
    k = 4
    capacity = 6
    routing_weights = torch.rand(B, T, 1)
    selected_indices, _ = select_top_k_tokens(routing_weights, k, capacity)
    assert (selected_indices >= 0).all(), "Indices must be >= 0"
    assert (selected_indices < T).all(), f"Indices must be < T={T}"


def test_compute_router_z_loss_scalar():
    routing_logits = torch.randn(B, T, 1)
    z_loss = compute_router_z_loss(routing_logits)
    assert isinstance(z_loss, torch.Tensor), "z_loss must be a Tensor"
    assert z_loss.ndim == 0, f"z_loss must be scalar, got ndim={z_loss.ndim}"


def test_compute_router_z_loss_zero_logits_low():
    zero_logits = torch.zeros(B, T, 1)
    large_logits = torch.full((B, T, 1), 10.0)
    z_loss_zero = compute_router_z_loss(zero_logits)
    z_loss_large = compute_router_z_loss(large_logits)
    assert z_loss_zero.item() < z_loss_large.item(), (
        f"Expected z_loss(zero) < z_loss(large), got {z_loss_zero.item()} vs {z_loss_large.item()}"
    )


def test_compute_capacity_utilization_range():
    k = 3
    capacity = 5
    routing_weights = torch.rand(B, T, 1)
    selected_indices, _ = select_top_k_tokens(routing_weights, k, capacity)
    util = compute_capacity_utilization(selected_indices, T)
    assert isinstance(util, float), "Utilization must be a float"
    assert 0.0 <= util <= 1.0, f"Utilization must be in [0, 1], got {util}"


def test_mod_layer_v2_output_shape():
    config = MoDv2Config(top_k=2, capacity_factor=1.25)
    layer = nn.Linear(D, D)
    mod_layer = MoDLayerV2(layer=layer, d_model=D, config=config)
    x = torch.randn(B, T, D)
    x_out, info = mod_layer(x)
    assert x_out.shape == (B, T, D), (
        f"Expected output shape ({B}, {T}, {D}), got {x_out.shape}"
    )


def test_mod_layer_v2_returns_routing_dict():
    config = MoDv2Config(top_k=2)
    layer = nn.Linear(D, D)
    mod_layer = MoDLayerV2(layer=layer, d_model=D, config=config)
    x = torch.randn(B, T, D)
    _, info = mod_layer(x)
    assert isinstance(info, dict), "Second return must be a dict"
    assert "router_z_loss" in info, "Info must contain 'router_z_loss'"
    assert "capacity_utilization" in info, "Info must contain 'capacity_utilization'"
    assert isinstance(info["router_z_loss"], torch.Tensor), "router_z_loss must be a Tensor"
    assert info["router_z_loss"].ndim == 0, "router_z_loss must be scalar"
    assert isinstance(info["capacity_utilization"], float), "capacity_utilization must be a float"


def test_build_mod_v2_model_wraps_correct_layers():
    n_layers = 4
    base_layers = nn.ModuleList([nn.Linear(D, D) for _ in range(n_layers)])
    config = MoDv2Config(top_k=1)
    mod_indices = [1, 3]

    new_layers = build_mod_v2_model(base_layers, D, config, mod_indices)

    assert len(new_layers) == n_layers, f"Expected {n_layers} layers, got {len(new_layers)}"

    for i, layer in enumerate(new_layers):
        if i in mod_indices:
            assert isinstance(layer, MoDLayerV2), (
                f"Layer {i} should be MoDLayerV2, got {type(layer).__name__}"
            )
        else:
            assert isinstance(layer, nn.Linear), (
                f"Layer {i} should be nn.Linear, got {type(layer).__name__}"
            )


def test_mod_layer_v2_top_k_zero_identity():
    """With top_k=0, no tokens are processed -- output should equal input."""
    config = MoDv2Config(top_k=0, capacity_factor=1.25)
    layer = nn.Linear(D, D)
    mod_layer = MoDLayerV2(layer=layer, d_model=D, config=config)
    # set to inference mode (no batchnorm/dropout effects)
    mod_layer.train(False)

    x = torch.randn(B, T, D)
    with torch.no_grad():
        x_out, info = mod_layer(x)

    assert x_out.shape == (B, T, D), "Output shape must match input shape"
    assert torch.allclose(x_out, x, atol=1e-6), (
        "With top_k=0, output must equal input (identity / all shortcut)"
    )
    assert info["capacity_utilization"] == 0.0, (
        f"With top_k=0, capacity_utilization must be 0.0, got {info['capacity_utilization']}"
    )
