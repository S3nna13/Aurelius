"""Tests for src/model/mixture_of_depths.py."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.mixture_of_depths import (
    MoDConfig,
    MoDRouter,
    MoDLayer,
    MoDTransformerWrapper,
    mod_aux_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    return AureliusTransformer(tiny_cfg)


@pytest.fixture
def simple_layer():
    """A plain nn.Linear as a stand-in for a transformer sub-layer."""
    return nn.Linear(64, 64, bias=False)


@pytest.fixture
def identity_layer():
    return nn.Identity()


# ---------------------------------------------------------------------------
# 1. MoDConfig defaults
# ---------------------------------------------------------------------------

class TestMoDConfigDefaults:
    def test_capacity_factor_default(self):
        cfg = MoDConfig()
        assert cfg.capacity_factor == 0.5

    def test_aux_loss_coeff_default(self):
        cfg = MoDConfig()
        assert cfg.router_aux_loss_coeff == 0.01


# ---------------------------------------------------------------------------
# 2-5. MoDRouter
# ---------------------------------------------------------------------------

class TestMoDRouter:
    """Tests for MoDRouter output shapes and selection correctness."""

    @pytest.mark.parametrize("B,T,D", [(1, 10, 64), (2, 20, 64), (4, 8, 64)])
    def test_output_shapes(self, B, T, D):
        """Test 2: MoDRouter output shapes for various (B, T, d_model)."""
        router = MoDRouter(D, capacity_factor=0.5)
        x = torch.randn(B, T, D)
        selected_x, indices, router_scores = router(x)

        k = max(1, math.ceil(0.5 * T))
        assert selected_x.shape == (B, k, D), f"selected_x shape mismatch: {selected_x.shape}"
        assert indices.shape == (B, k), f"indices shape mismatch: {indices.shape}"
        assert router_scores.shape == (B, T), f"router_scores shape mismatch: {router_scores.shape}"

    @pytest.mark.parametrize("capacity_factor,T", [(0.5, 10), (0.25, 8), (1.0, 6), (0.3, 7)])
    def test_selects_exact_k_tokens(self, capacity_factor, T):
        """Test 3: MoDRouter selects exactly ceil(capacity_factor * T) tokens."""
        D = 32
        router = MoDRouter(D, capacity_factor=capacity_factor)
        x = torch.randn(1, T, D)
        selected_x, indices, _ = router(x)

        expected_k = max(1, math.ceil(capacity_factor * T))
        assert indices.shape[1] == expected_k, (
            f"Expected k={expected_k} but got {indices.shape[1]} for "
            f"capacity_factor={capacity_factor}, T={T}"
        )

    def test_indices_in_valid_range(self):
        """Test 4: MoDRouter indices are in range [0, T)."""
        B, T, D = 3, 12, 64
        router = MoDRouter(D, capacity_factor=0.5)
        x = torch.randn(B, T, D)
        _, indices, _ = router(x)

        assert (indices >= 0).all(), "Indices contain negative values"
        assert (indices < T).all(), f"Indices exceed T={T}"

    def test_router_scores_shape(self):
        """Test 5: MoDRouter router_scores shape is (B, T)."""
        B, T, D = 2, 15, 64
        router = MoDRouter(D, capacity_factor=0.5)
        x = torch.randn(B, T, D)
        _, _, router_scores = router(x)
        assert router_scores.shape == (B, T)

    def test_minimum_one_token_always_selected(self):
        """Even with tiny capacity_factor and T=1, at least 1 token is selected."""
        router = MoDRouter(64, capacity_factor=0.01)
        x = torch.randn(1, 1, 64)
        selected_x, indices, _ = router(x)
        assert selected_x.shape[1] >= 1
        assert indices.shape[1] >= 1


# ---------------------------------------------------------------------------
# 6-9. MoDLayer
# ---------------------------------------------------------------------------

class TestMoDLayer:
    """Tests for MoDLayer forward output correctness."""

    def test_forward_returns_tuple(self, simple_layer):
        """Test 6: MoDLayer forward returns (Tensor, Tensor) — output and aux_loss."""
        config = MoDConfig(capacity_factor=0.5)
        layer = MoDLayer(simple_layer, d_model=64, config=config)
        x = torch.randn(2, 10, 64)
        result = layer(x)

        assert isinstance(result, tuple), "MoDLayer.forward should return a tuple"
        assert len(result) == 2, "Tuple should have 2 elements: (output, aux_loss)"
        output, aux_loss = result
        assert isinstance(output, torch.Tensor)
        assert isinstance(aux_loss, torch.Tensor)

    def test_output_shape_matches_input(self, simple_layer):
        """Test 7: MoDLayer output shape matches input (B, T, d_model)."""
        config = MoDConfig(capacity_factor=0.5)
        layer = MoDLayer(simple_layer, d_model=64, config=config)
        x = torch.randn(2, 10, 64)
        output, _ = layer(x)
        assert output.shape == x.shape, (
            f"Output shape {output.shape} does not match input {x.shape}"
        )

    def test_aux_loss_is_scalar(self, simple_layer):
        """Test 8: MoDLayer aux_loss is a scalar (0-dim tensor)."""
        config = MoDConfig(capacity_factor=0.5)
        layer = MoDLayer(simple_layer, d_model=64, config=config)
        x = torch.randn(2, 10, 64)
        _, aux_loss = layer(x)
        assert aux_loss.ndim == 0, f"aux_loss should be scalar, got ndim={aux_loss.ndim}"

    def test_aux_loss_non_negative(self, simple_layer):
        """Test 9: MoDLayer aux_loss >= 0."""
        config = MoDConfig(capacity_factor=0.5)
        layer = MoDLayer(simple_layer, d_model=64, config=config)
        x = torch.randn(2, 10, 64)
        _, aux_loss = layer(x)
        assert aux_loss.item() >= 0.0, f"aux_loss should be non-negative, got {aux_loss.item()}"

    @pytest.mark.parametrize("B,T", [(1, 8), (2, 16), (3, 5)])
    def test_various_batch_seq_sizes(self, B, T):
        """MoDLayer works for various batch and sequence sizes."""
        config = MoDConfig(capacity_factor=0.5)
        layer = MoDLayer(nn.Linear(64, 64, bias=False), d_model=64, config=config)
        x = torch.randn(B, T, 64)
        output, aux_loss = layer(x)
        assert output.shape == x.shape
        assert aux_loss.ndim == 0

    def test_identity_layer_passes_selected_through(self, identity_layer):
        """With identity_layer, selected tokens are unchanged."""
        config = MoDConfig(capacity_factor=1.0)
        layer = MoDLayer(identity_layer, d_model=64, config=config)
        x = torch.randn(2, 8, 64)
        output, _ = layer(x)
        # With capacity_factor=1.0 all tokens are selected and identity keeps them same
        assert torch.allclose(output, x, atol=1e-5), "Identity layer should not change tokens"


# ---------------------------------------------------------------------------
# 10-11. mod_aux_loss
# ---------------------------------------------------------------------------

class TestModAuxLoss:
    def test_returns_scalar(self):
        """Test 10: mod_aux_loss returns scalar."""
        scores = torch.randn(2, 10)
        loss = mod_aux_loss(scores, capacity_factor=0.5)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_uniform_routing_near_zero(self):
        """Test 11: mod_aux_loss returns ~0 for uniform routing at target capacity.

        When all router scores are the same, sigmoid(score) is constant and
        mean_prob equals sigmoid(score). Setting score so sigmoid(score)=0.5
        (capacity_factor) should yield loss ~0.
        """
        # logit(0.5) = 0.0, so sigmoid(0.0) = 0.5 exactly
        scores = torch.zeros(4, 20)  # all zeros -> sigmoid = 0.5
        loss = mod_aux_loss(scores, capacity_factor=0.5)
        assert loss.item() < 1e-6, f"Expected near-zero loss, got {loss.item()}"

    def test_imbalanced_routing_positive(self):
        """Clearly imbalanced routing should give a positive loss."""
        # All scores very high -> sigmoid ~1.0, target 0.5 -> loss = (1-0.5)^2 = 0.25
        scores = torch.full((2, 10), 100.0)
        loss = mod_aux_loss(scores, capacity_factor=0.5)
        assert loss.item() > 0.1, f"Expected positive loss for imbalanced routing, got {loss.item()}"

    def test_loss_is_non_negative(self):
        """mod_aux_loss is always non-negative (MSE-based)."""
        scores = torch.randn(3, 15)
        loss = mod_aux_loss(scores, capacity_factor=0.4)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 12-15. MoDTransformerWrapper
# ---------------------------------------------------------------------------

class TestMoDTransformerWrapper:
    def test_wraps_model_forward_returns_3_tuple(self, tiny_model):
        """Test 12: MoDTransformerWrapper forward returns 3-tuple."""
        config = MoDConfig(capacity_factor=0.5)
        wrapper = MoDTransformerWrapper(tiny_model, config)
        input_ids = torch.randint(0, 256, (1, 16))
        result = wrapper(input_ids)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_output_logits_shape_correct(self, tiny_model, tiny_cfg):
        """Test 13: MoDTransformerWrapper output logits shape correct."""
        config = MoDConfig(capacity_factor=0.5)
        wrapper = MoDTransformerWrapper(tiny_model, config)
        B, S = 2, 16
        input_ids = torch.randint(0, tiny_cfg.vocab_size, (B, S))
        loss, logits, pkv = wrapper(input_ids)
        assert logits.shape == (B, S, tiny_cfg.vocab_size), (
            f"Expected logits shape ({B}, {S}, {tiny_cfg.vocab_size}), got {logits.shape}"
        )

    def test_get_routing_stats_returns_dict_with_expected_keys(self, tiny_model):
        """Test 14: get_routing_stats returns dict with expected keys."""
        config = MoDConfig(capacity_factor=0.5)
        wrapper = MoDTransformerWrapper(tiny_model, config)
        input_ids = torch.randint(0, 256, (1, 16))
        wrapper(input_ids)  # populate stats

        stats = wrapper.get_routing_stats()
        assert isinstance(stats, dict)
        assert "tokens_processed_fraction" in stats
        assert "capacity_factor" in stats
        assert "n_mod_layers" in stats
        assert "aux_loss" in stats

    def test_capacity_factor_1_vs_05(self, tiny_cfg):
        """Test 15: capacity_factor=1.0 (all tokens) vs 0.5 (half tokens).

        With capacity_factor=1.0, all tokens are processed by every layer.
        With 0.5, roughly half are skipped. Both should produce valid outputs.
        """
        B, S = 2, 16
        input_ids = torch.randint(0, tiny_cfg.vocab_size, (B, S))

        # capacity_factor = 1.0: all tokens processed
        model_full = AureliusTransformer(tiny_cfg)
        wrapper_full = MoDTransformerWrapper(model_full, MoDConfig(capacity_factor=1.0))
        loss_full, logits_full, _ = wrapper_full(input_ids)
        stats_full = wrapper_full.get_routing_stats()
        assert logits_full.shape == (B, S, tiny_cfg.vocab_size)
        assert abs(stats_full["tokens_processed_fraction"] - 1.0) < 1e-5

        # capacity_factor = 0.5: half tokens processed
        model_half = AureliusTransformer(tiny_cfg)
        wrapper_half = MoDTransformerWrapper(model_half, MoDConfig(capacity_factor=0.5))
        loss_half, logits_half, _ = wrapper_half(input_ids)
        stats_half = wrapper_half.get_routing_stats()
        assert logits_half.shape == (B, S, tiny_cfg.vocab_size)
        expected_fraction = math.ceil(0.5 * S) / S
        assert abs(stats_half["tokens_processed_fraction"] - expected_fraction) < 1e-5

    def test_n_mod_layers_matches_model_layers(self, tiny_model, tiny_cfg):
        """MoDTransformerWrapper creates one MoDLayer per transformer block."""
        config = MoDConfig(capacity_factor=0.5)
        wrapper = MoDTransformerWrapper(tiny_model, config)
        assert len(wrapper.mod_layers) == tiny_cfg.n_layers

    def test_loss_is_scalar_tensor(self, tiny_model):
        """Returned loss is a scalar tensor."""
        config = MoDConfig(capacity_factor=0.5)
        wrapper = MoDTransformerWrapper(tiny_model, config)
        input_ids = torch.randint(0, 256, (1, 8))
        loss, _, _ = wrapper(input_ids)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_routing_stats_capacity_factor_matches_config(self, tiny_model):
        """stats['capacity_factor'] matches the MoDConfig value."""
        config = MoDConfig(capacity_factor=0.3)
        wrapper = MoDTransformerWrapper(tiny_model, config)
        input_ids = torch.randint(0, 256, (1, 10))
        wrapper(input_ids)
        stats = wrapper.get_routing_stats()
        assert stats["capacity_factor"] == 0.3

    def test_gradients_flow_through_wrapper(self, tiny_model):
        """Gradients should flow through the MoDTransformerWrapper."""
        config = MoDConfig(capacity_factor=0.5)
        wrapper = MoDTransformerWrapper(tiny_model, config)
        input_ids = torch.randint(0, 256, (1, 8))
        loss, logits, _ = wrapper(input_ids)
        # Backprop through logits sum
        logits.sum().backward()
        # At least one parameter should have a gradient
        has_grad = any(
            p.grad is not None
            for p in wrapper.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients flowed through MoDTransformerWrapper"
