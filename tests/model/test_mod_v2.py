"""Tests for src/model/mod_v2.py -- Mixture-of-Depths v2."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.mod_v2 import (
    CapacityTracker,
    MoDv2Config,
    MoDv2Layer,
    MoDv2Transformer,
    RouterV2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def mod_cfg() -> MoDv2Config:
    return MoDv2Config(capacity_factor=0.5, routing_type="top_k")


@pytest.fixture
def router(mod_cfg) -> RouterV2:
    return RouterV2(d_model=64, cfg=mod_cfg)


@pytest.fixture
def simple_sublayer() -> nn.Module:
    """A simple linear sublayer that maps D -> D."""
    return nn.Linear(64, 64, bias=False)


@pytest.fixture
def mod_layer(simple_sublayer, mod_cfg) -> MoDv2Layer:
    return MoDv2Layer(sublayer=simple_sublayer, d_model=64, cfg=mod_cfg)


# ---------------------------------------------------------------------------
# 1. test_router_output_shapes
# ---------------------------------------------------------------------------

def test_router_output_shapes(router):
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D)
    routing_weights, route_indices, aux_loss = router(x)

    capacity = math.ceil(T * router.cfg.capacity_factor)

    assert routing_weights.shape == (B, T, 1), (
        f"Expected routing_weights shape ({B}, {T}, 1), got {routing_weights.shape}"
    )
    assert route_indices.shape == (B, capacity), (
        f"Expected route_indices shape ({B}, {capacity}), got {route_indices.shape}"
    )
    assert aux_loss.ndim == 0, f"aux_loss should be a scalar, got ndim={aux_loss.ndim}"


# ---------------------------------------------------------------------------
# 2. test_router_capacity_fraction
# ---------------------------------------------------------------------------

def test_router_capacity_fraction(mod_cfg):
    T = 20
    capacity_factor = 0.5
    mod_cfg = MoDv2Config(capacity_factor=capacity_factor, routing_type="top_k")
    router = RouterV2(d_model=64, cfg=mod_cfg)

    x = torch.randn(1, T, 64)
    _, route_indices, _ = router(x)

    expected_capacity = math.ceil(T * capacity_factor)
    assert route_indices.shape[1] == expected_capacity, (
        f"Expected {expected_capacity} tokens selected, got {route_indices.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 3. test_mod_layer_output_shape
# ---------------------------------------------------------------------------

def test_mod_layer_output_shape(mod_layer):
    B, T, D = 2, 12, 64
    x = torch.randn(B, T, D)
    output, aux_loss = mod_layer(x)
    assert output.shape == (B, T, D), (
        f"Expected output shape ({B}, {T}, {D}), got {output.shape}"
    )


# ---------------------------------------------------------------------------
# 4. test_mod_layer_returns_aux_loss
# ---------------------------------------------------------------------------

def test_mod_layer_returns_aux_loss(mod_layer):
    x = torch.randn(2, 10, 64, requires_grad=True)
    output, aux_loss = mod_layer(x)

    # Must be a tensor (not a Python float)
    assert isinstance(aux_loss, torch.Tensor), "aux_loss must be a torch.Tensor"
    assert aux_loss.ndim == 0, "aux_loss must be a scalar tensor"

    # Must be differentiable: backward should not raise
    aux_loss.backward()


# ---------------------------------------------------------------------------
# 5. test_mod_layer_unrouted_tokens_unchanged
# ---------------------------------------------------------------------------

def test_mod_layer_unrouted_tokens_unchanged():
    """Tokens not selected by the router must equal the input (pure residual)."""
    B, T, D = 1, 8, 64
    capacity_factor = 0.25  # only 2 of 8 tokens routed
    mod_cfg = MoDv2Config(capacity_factor=capacity_factor, routing_type="top_k",
                          use_aux_loss=False)
    sublayer = nn.Linear(D, D, bias=False)
    # Zero out sublayer weights so routed tokens map to 0 -- unrouted = x
    nn.init.zeros_(sublayer.weight)

    mod_layer = MoDv2Layer(sublayer=sublayer, d_model=D, cfg=mod_cfg)
    mod_layer.eval()

    x = torch.randn(B, T, D)

    with torch.no_grad():
        # Get which tokens are selected
        routing_weights, route_indices, _ = mod_layer.router(x)
        selected_set = set(route_indices[0].tolist())

        output, _ = mod_layer(x)

    # Unrouted tokens must be identical to input
    for t in range(T):
        if t not in selected_set:
            assert torch.allclose(output[0, t], x[0, t], atol=1e-6), (
                f"Unrouted token {t} was modified: "
                f"max diff = {(output[0, t] - x[0, t]).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# 6. test_router_aux_loss_positive
# ---------------------------------------------------------------------------

def test_router_aux_loss_positive(router):
    x = torch.randn(2, 16, 64)
    _, _, aux_loss = router(x)
    assert aux_loss.item() > 0, (
        f"Expected aux_loss > 0, got {aux_loss.item()}"
    )


# ---------------------------------------------------------------------------
# 7. test_capacity_tracker_record_and_summary
# ---------------------------------------------------------------------------

def test_capacity_tracker_record_and_summary():
    n_layers = 3
    tracker = CapacityTracker(n_layers)

    tracker.record(0, n_routed=5, n_total=10)
    tracker.record(1, n_routed=4, n_total=10)
    tracker.record(2, n_routed=6, n_total=10)

    stats = tracker.summary()

    # Expected keys
    for i in range(n_layers):
        assert f"layer_{i}_mean" in stats, f"Missing key 'layer_{i}_mean'"
        assert f"layer_{i}_std" in stats, f"Missing key 'layer_{i}_std'"
    assert "overall_mean" in stats, "Missing key 'overall_mean'"

    assert math.isclose(stats["layer_0_mean"], 0.5, rel_tol=1e-6)
    assert math.isclose(stats["layer_1_mean"], 0.4, rel_tol=1e-6)
    assert math.isclose(stats["layer_2_mean"], 0.6, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 8. test_capacity_tracker_overall_mean
# ---------------------------------------------------------------------------

def test_capacity_tracker_overall_mean():
    tracker = CapacityTracker(n_layers=2)

    # Layer 0: fractions 0.5 and 0.5 -> mean 0.5
    tracker.record(0, 5, 10)
    tracker.record(0, 5, 10)

    # Layer 1: fractions 0.4 and 0.6 -> mean 0.5
    tracker.record(1, 4, 10)
    tracker.record(1, 6, 10)

    stats = tracker.summary()
    # overall_mean = mean of all four fractions: 0.5, 0.5, 0.4, 0.6 = 0.5
    assert math.isclose(stats["overall_mean"], 0.5, rel_tol=1e-6), (
        f"Expected overall_mean=0.5, got {stats['overall_mean']}"
    )


# ---------------------------------------------------------------------------
# 9. test_router_z_loss_present
# ---------------------------------------------------------------------------

def test_router_z_loss_present():
    """aux_loss should change when z_loss_coeff changes (z-loss is active)."""
    d_model = 64
    x = torch.randn(2, 16, d_model)

    cfg_no_z = MoDv2Config(
        capacity_factor=0.5,
        router_aux_loss_coeff=0.01,
        router_z_loss_coeff=0.0,
    )
    cfg_with_z = MoDv2Config(
        capacity_factor=0.5,
        router_aux_loss_coeff=0.01,
        router_z_loss_coeff=1.0,  # large coefficient so difference is clear
    )

    router_no_z = RouterV2(d_model, cfg_no_z)
    router_with_z = RouterV2(d_model, cfg_with_z)

    # Share the same linear weights so only the coefficient differs
    router_with_z.router.weight = router_no_z.router.weight

    with torch.no_grad():
        _, _, loss_no_z = router_no_z(x)
        _, _, loss_with_z = router_with_z(x)

    assert not math.isclose(loss_no_z.item(), loss_with_z.item(), rel_tol=1e-4), (
        f"Expected aux_loss to differ when z_loss_coeff changes, "
        f"but got {loss_no_z.item()} vs {loss_with_z.item()}"
    )


# ---------------------------------------------------------------------------
# 10. test_router_top_p_routing
# ---------------------------------------------------------------------------

def test_router_top_p_routing():
    """top_p routing should not crash and return correct shapes."""
    B, T, D = 2, 16, 64
    capacity_factor = 0.5
    cfg = MoDv2Config(capacity_factor=capacity_factor, routing_type="top_p")
    router = RouterV2(d_model=D, cfg=cfg)

    x = torch.randn(B, T, D)
    routing_weights, route_indices, aux_loss = router(x)

    capacity = math.ceil(T * capacity_factor)

    assert routing_weights.shape == (B, T, 1), (
        f"Expected routing_weights ({B}, {T}, 1), got {routing_weights.shape}"
    )
    assert route_indices.shape == (B, capacity), (
        f"Expected route_indices ({B}, {capacity}), got {route_indices.shape}"
    )
    assert aux_loss.ndim == 0, "aux_loss should be scalar"
