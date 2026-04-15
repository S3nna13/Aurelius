"""Tests for src/model/mixture_of_depths.py.

Uses tiny configs: D_MODEL=16, CAPACITY=0.5, B=2, SEQ=8.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.mixture_of_depths import (
    MoDConfig,
    MoDLayer,
    MoDTransformer,
    compute_mod_aux_loss,
    scatter_back,
    token_router,
)

# ---------------------------------------------------------------------------
# Tiny test constants
# ---------------------------------------------------------------------------

D_MODEL = 16
CAPACITY = 0.5
B = 2
SEQ = 8


# ---------------------------------------------------------------------------
# 1. MoDConfig defaults
# ---------------------------------------------------------------------------


class TestMoDConfigDefaults:
    def test_d_model_default(self):
        cfg = MoDConfig()
        assert cfg.d_model == 64

    def test_capacity_default(self):
        cfg = MoDConfig()
        assert cfg.capacity == 0.125

    def test_aux_loss_coef_default(self):
        cfg = MoDConfig()
        assert cfg.aux_loss_coef == 0.01


# ---------------------------------------------------------------------------
# 2. token_router output shapes: selected (B,k,D), indices (B,k), weights (B,k)
# ---------------------------------------------------------------------------


class TestTokenRouterShapes:
    def test_selected_tokens_shape(self):
        router = nn.Linear(D_MODEL, 1, bias=False)
        x = torch.randn(B, SEQ, D_MODEL)
        k = math.ceil(CAPACITY * SEQ)
        selected, indices, weights = token_router(x, router, CAPACITY)
        assert selected.shape == (B, k, D_MODEL), f"Expected ({B},{k},{D_MODEL}), got {selected.shape}"

    def test_token_indices_shape(self):
        router = nn.Linear(D_MODEL, 1, bias=False)
        x = torch.randn(B, SEQ, D_MODEL)
        k = math.ceil(CAPACITY * SEQ)
        selected, indices, weights = token_router(x, router, CAPACITY)
        assert indices.shape == (B, k), f"Expected ({B},{k}), got {indices.shape}"

    def test_router_weights_shape(self):
        router = nn.Linear(D_MODEL, 1, bias=False)
        x = torch.randn(B, SEQ, D_MODEL)
        k = math.ceil(CAPACITY * SEQ)
        selected, indices, weights = token_router(x, router, CAPACITY)
        assert weights.shape == (B, k), f"Expected ({B},{k}), got {weights.shape}"


# ---------------------------------------------------------------------------
# 3. k = ceil(capacity * T)
# ---------------------------------------------------------------------------


class TestTokenRouterK:
    @pytest.mark.parametrize("capacity,T", [
        (0.5, 8),
        (0.25, 8),
        (0.3, 7),
        (1.0, 6),
        (0.125, 16),
    ])
    def test_k_equals_ceil_capacity_times_T(self, capacity, T):
        router = nn.Linear(D_MODEL, 1, bias=False)
        x = torch.randn(1, T, D_MODEL)
        expected_k = math.ceil(capacity * T)
        selected, indices, weights = token_router(x, router, capacity)
        assert selected.shape[1] == expected_k
        assert indices.shape[1] == expected_k
        assert weights.shape[1] == expected_k


# ---------------------------------------------------------------------------
# 4. weights sum to 1 per batch item (among selected)
# ---------------------------------------------------------------------------


class TestTokenRouterWeights:
    def test_weights_sum_to_one_per_batch_item(self):
        router = nn.Linear(D_MODEL, 1, bias=False)
        x = torch.randn(B, SEQ, D_MODEL)
        selected, indices, weights = token_router(x, router, CAPACITY)
        weight_sums = weights.sum(dim=-1)  # (B,)
        assert torch.allclose(weight_sums, torch.ones(B), atol=1e-5), (
            f"Weights do not sum to 1 per batch item: {weight_sums}"
        )


# ---------------------------------------------------------------------------
# 5. scatter_back output shape (B, T, D)
# ---------------------------------------------------------------------------


class TestScatterBackShape:
    def test_output_shape(self):
        k = math.ceil(CAPACITY * SEQ)
        output = torch.randn(B, k, D_MODEL)
        indices = torch.randint(0, SEQ, (B, k))
        weights = torch.softmax(torch.randn(B, k), dim=-1)
        x = torch.randn(B, SEQ, D_MODEL)
        result = scatter_back(output, indices, weights, x)
        assert result.shape == (B, SEQ, D_MODEL), f"Expected ({B},{SEQ},{D_MODEL}), got {result.shape}"


# ---------------------------------------------------------------------------
# 6. unselected positions preserved in scatter_back
# ---------------------------------------------------------------------------


class TestScatterBackUnselected:
    def test_unselected_positions_preserved(self):
        T = 8
        k = 2  # select only 2 out of 8 tokens
        # Use fixed indices [0, 1] for batch item 0 and [2, 3] for batch item 1
        indices = torch.tensor([[0, 1], [2, 3]])
        output = torch.zeros(B, k, D_MODEL)
        weights = torch.ones(B, k) / k
        x = torch.randn(B, T, D_MODEL)
        result = scatter_back(output, indices, weights, x)

        # Unselected for batch item 0: positions 2..7
        for pos in range(2, T):
            assert torch.allclose(result[0, pos], x[0, pos]), (
                f"Batch 0, position {pos} was modified but should be preserved"
            )
        # Unselected for batch item 1: positions 0,1,4..7
        for pos in list(range(0, 2)) + list(range(4, T)):
            assert torch.allclose(result[1, pos], x[1, pos]), (
                f"Batch 1, position {pos} was modified but should be preserved"
            )


# ---------------------------------------------------------------------------
# 7. compute_mod_aux_loss returns a scalar
# ---------------------------------------------------------------------------


class TestComputeModAuxLoss:
    def test_returns_scalar(self):
        scores = torch.randn(B, SEQ)
        loss = compute_mod_aux_loss(scores, CAPACITY, coef=0.01)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0, f"Expected scalar (0-dim), got ndim={loss.ndim}"

    def test_non_negative(self):
        scores = torch.randn(B, SEQ)
        loss = compute_mod_aux_loss(scores, CAPACITY, coef=0.01)
        assert loss.item() >= 0.0, f"Expected non-negative loss, got {loss.item()}"

    def test_imbalanced_routing_positive(self):
        # All scores very high -> sigmoid ~1.0, deviation from capacity is large
        scores = torch.full((B, SEQ), 100.0)
        loss = compute_mod_aux_loss(scores, CAPACITY, coef=0.01)
        assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 8. MoDLayer output shape (B, T, D)
# ---------------------------------------------------------------------------


class TestMoDLayerOutputShape:
    def test_output_shape(self):
        config = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        layer = MoDLayer(nn.Identity(), config)
        x = torch.randn(B, SEQ, D_MODEL)
        output, aux_loss = layer(x)
        assert output.shape == (B, SEQ, D_MODEL), f"Expected ({B},{SEQ},{D_MODEL}), got {output.shape}"


# ---------------------------------------------------------------------------
# 9. MoDLayer aux_loss is scalar
# ---------------------------------------------------------------------------


class TestMoDLayerAuxLoss:
    def test_aux_loss_is_scalar(self):
        config = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        layer = MoDLayer(nn.Identity(), config)
        x = torch.randn(B, SEQ, D_MODEL)
        output, aux_loss = layer(x)
        assert aux_loss.ndim == 0, f"aux_loss should be 0-dim scalar, got ndim={aux_loss.ndim}"

    def test_aux_loss_non_negative(self):
        config = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        layer = MoDLayer(nn.Identity(), config)
        x = torch.randn(B, SEQ, D_MODEL)
        _, aux_loss = layer(x)
        assert aux_loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 10. Gradient flows through MoDLayer
# ---------------------------------------------------------------------------


class TestMoDLayerGradients:
    def test_gradient_flows(self):
        config = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        inner = nn.Linear(D_MODEL, D_MODEL, bias=False)
        layer = MoDLayer(inner, config)
        x = torch.randn(B, SEQ, D_MODEL, requires_grad=True)
        output, aux_loss = layer(x)
        loss = output.sum() + aux_loss
        loss.backward()

        assert x.grad is not None, "No gradient flowed to input x"
        has_param_grad = any(
            p.grad is not None for p in layer.parameters() if p.requires_grad
        )
        assert has_param_grad, "No gradient flowed to MoDLayer parameters"


# ---------------------------------------------------------------------------
# 11. MoDTransformer output shape
# ---------------------------------------------------------------------------


class TestMoDTransformerOutputShape:
    def test_output_shape(self):
        config = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        layers = [nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(2)]
        transformer = MoDTransformer(layers, config)
        x = torch.randn(B, SEQ, D_MODEL)
        output, total_aux_loss = transformer(x)
        assert output.shape == (B, SEQ, D_MODEL), f"Expected ({B},{SEQ},{D_MODEL}), got {output.shape}"


# ---------------------------------------------------------------------------
# 12. MoDTransformer aux_loss is sum of per-layer losses (2 layers)
# ---------------------------------------------------------------------------


class TestMoDTransformerAuxLossSum:
    def test_aux_loss_sum_of_layers(self):
        """MoDTransformer total_aux_loss equals sum of individual layer aux losses."""
        config = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        inner1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        inner2 = nn.Linear(D_MODEL, D_MODEL, bias=False)

        transformer = MoDTransformer([inner1, inner2], config)
        x = torch.randn(B, SEQ, D_MODEL)

        # Run through transformer and collect total
        output, total_aux_loss = transformer(x)

        # Run each layer independently from x to get their individual losses
        # Note: we test that total = layer1_loss + layer2_loss
        # We do this by running the transformer with a single layer each
        cfg1 = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        layer1 = transformer.mod_layers[0]
        layer2 = transformer.mod_layers[1]

        with torch.no_grad():
            out1, loss1 = layer1(x)
            out2, loss2 = layer2(out1)

        expected_total = (loss1 + loss2).item()
        assert abs(total_aux_loss.item() - expected_total) < 1e-5, (
            f"Expected total aux_loss={expected_total}, got {total_aux_loss.item()}"
        )

    def test_total_aux_loss_is_scalar(self):
        config = MoDConfig(d_model=D_MODEL, capacity=CAPACITY, aux_loss_coef=0.01)
        layers = [nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(2)]
        transformer = MoDTransformer(layers, config)
        x = torch.randn(B, SEQ, D_MODEL)
        _, total_aux_loss = transformer(x)
        assert total_aux_loss.ndim == 0


# ---------------------------------------------------------------------------
# 13. capacity=1.0 routes all tokens
# ---------------------------------------------------------------------------


class TestCapacityOne:
    def test_capacity_one_routes_all_tokens(self):
        """When capacity=1.0, all tokens are selected (k == T)."""
        config = MoDConfig(d_model=D_MODEL, capacity=1.0, aux_loss_coef=0.01)
        router = nn.Linear(D_MODEL, 1, bias=False)
        x = torch.randn(B, SEQ, D_MODEL)
        selected, indices, weights = token_router(x, router, capacity=1.0)
        # All T tokens should be selected
        assert selected.shape == (B, SEQ, D_MODEL), (
            f"With capacity=1.0, expected all {SEQ} tokens selected, got {selected.shape}"
        )
        assert indices.shape == (B, SEQ)
        assert weights.shape == (B, SEQ)

    def test_mod_layer_capacity_one(self):
        """With capacity=1.0, MoDLayer processes all tokens."""
        config = MoDConfig(d_model=D_MODEL, capacity=1.0, aux_loss_coef=0.01)
        layer = MoDLayer(nn.Identity(), config)
        x = torch.randn(B, SEQ, D_MODEL)
        output, aux_loss = layer(x)
        assert output.shape == (B, SEQ, D_MODEL)
        assert aux_loss.ndim == 0
