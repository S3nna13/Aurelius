"""Tests for src/model/stochastic_depth.py.

Uses tiny configurations (D_MODEL=16, B=2, SEQ=4, 4 layers) so every test
runs quickly in CI without a GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.stochastic_depth import (
    LinearStochasticDepth,
    StochasticDepthConfig,
    StochasticDepthLayer,
    StochasticDepthTransformer,
    get_expected_depth,
    stochastic_depth,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 16
B = 2
SEQ = 4
N_LAYERS = 4


def make_layers(n: int = N_LAYERS, d: int = D_MODEL) -> list[nn.Module]:
    return [nn.Linear(d, d) for _ in range(n)]


def make_input() -> torch.Tensor:
    """Return a (B, SEQ, D_MODEL) float tensor."""
    torch.manual_seed(0)
    return torch.randn(B, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


class TestStochasticDepthConfig:
    def test_default_drop_rate(self):
        cfg = StochasticDepthConfig()
        assert cfg.drop_rate == 0.1

    def test_default_mode(self):
        cfg = StochasticDepthConfig()
        assert cfg.mode == "row"

    def test_custom_values(self):
        cfg = StochasticDepthConfig(drop_rate=0.3, mode="batch")
        assert cfg.drop_rate == 0.3
        assert cfg.mode == "batch"


# ---------------------------------------------------------------------------
# 2. stochastic_depth functional API
# ---------------------------------------------------------------------------


class TestStochasticDepthFunction:
    def test_eval_always_applies_layer(self):
        """In inference mode (training=False) the layer is always applied."""
        x = make_input()
        layer = nn.Linear(D_MODEL, D_MODEL)
        layer.train(False)  # put layer weights in deterministic state

        expected = x + layer(x)
        for _ in range(10):
            out = stochastic_depth(x, layer, drop_prob=0.5, training=False)
            assert torch.allclose(out, expected), "inference mode must always apply layer"

    def test_eval_no_scale(self):
        """In inference mode there is no 1/(1-p) scaling."""
        x = make_input()

        def identity(t: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(t)

        out = stochastic_depth(x, identity, drop_prob=0.5, training=False)
        # Expected: x + 1  (no scaling)
        assert torch.allclose(out, x + 1.0)

    def test_training_output_shape_matches_input(self):
        """Output shape == input shape in training mode."""
        x = make_input()
        layer = nn.Linear(D_MODEL, D_MODEL)
        out = stochastic_depth(x, layer, drop_prob=0.1, training=True)
        assert out.shape == x.shape

    def test_drop_prob_zero_always_applies_layer(self):
        """drop_prob=0.0 must always apply layer in training mode."""
        x = make_input()
        layer = nn.Linear(D_MODEL, D_MODEL)
        layer.train(False)  # deterministic weights

        expected = x + layer(x)
        for _ in range(10):
            out = stochastic_depth(x, layer, drop_prob=0.0, training=True)
            assert torch.allclose(out, expected), "drop_prob=0 must always apply layer"

    def test_drop_prob_one_always_drops(self):
        """drop_prob=1.0 in training must always return x unchanged."""
        x = make_input()

        called = []

        def counting_layer(t: torch.Tensor) -> torch.Tensor:
            called.append(1)
            return torch.zeros_like(t)

        for _ in range(20):
            out = stochastic_depth(x, counting_layer, drop_prob=1.0, training=True)
            assert torch.allclose(out, x), "drop_prob=1.0 must return residual only"

        assert len(called) == 0, "layer_fn must not be called when drop_prob=1.0"


# ---------------------------------------------------------------------------
# 3. StochasticDepthLayer
# ---------------------------------------------------------------------------


class TestStochasticDepthLayer:
    def test_output_shape(self):
        x = make_input()
        layer = nn.Linear(D_MODEL, D_MODEL)
        sdl = StochasticDepthLayer(layer, drop_prob=0.1)
        sdl.train(False)
        out = sdl(x)
        assert out.shape == x.shape

    def test_survival_prob_property(self):
        sdl = StochasticDepthLayer(nn.Linear(D_MODEL, D_MODEL), drop_prob=0.3)
        assert abs(sdl.survival_prob - 0.7) < 1e-6

    def test_survival_prob_zero_drop(self):
        sdl = StochasticDepthLayer(nn.Linear(D_MODEL, D_MODEL), drop_prob=0.0)
        assert sdl.survival_prob == 1.0

    def test_training_shape(self):
        x = make_input()
        layer = nn.Linear(D_MODEL, D_MODEL)
        sdl = StochasticDepthLayer(layer, drop_prob=0.2)
        sdl.train(True)
        out = sdl(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 4. LinearStochasticDepth
# ---------------------------------------------------------------------------


class TestLinearStochasticDepth:
    def test_get_drop_rates_length(self):
        rates = LinearStochasticDepth.get_drop_rates(N_LAYERS, max_drop_rate=0.4)
        assert len(rates) == N_LAYERS

    def test_get_drop_rates_first_is_zero(self):
        rates = LinearStochasticDepth.get_drop_rates(N_LAYERS, max_drop_rate=0.4)
        assert rates[0] == 0.0

    def test_get_drop_rates_last_is_max(self):
        max_rate = 0.4
        rates = LinearStochasticDepth.get_drop_rates(N_LAYERS, max_drop_rate=max_rate)
        assert abs(rates[-1] - max_rate) < 1e-6

    def test_get_drop_rates_monotone(self):
        rates = LinearStochasticDepth.get_drop_rates(N_LAYERS, max_drop_rate=0.4)
        for a, b in zip(rates, rates[1:]):
            assert b >= a, "drop rates must be non-decreasing"

    def test_get_drop_rates_single_layer(self):
        rates = LinearStochasticDepth.get_drop_rates(1, max_drop_rate=0.5)
        assert len(rates) == 1
        assert rates[0] == 0.0

    def test_wrap_layers_returns_stochastic_depth_layer_list(self):
        layers = make_layers()
        wrapped = LinearStochasticDepth.wrap_layers(layers, max_drop_rate=0.3)
        assert len(wrapped) == len(layers)
        assert all(isinstance(w, StochasticDepthLayer) for w in wrapped)

    def test_wrap_layers_first_drop_prob_zero(self):
        layers = make_layers()
        wrapped = LinearStochasticDepth.wrap_layers(layers, max_drop_rate=0.4)
        assert wrapped[0].drop_prob == 0.0

    def test_wrap_layers_last_drop_prob_max(self):
        layers = make_layers()
        max_rate = 0.4
        wrapped = LinearStochasticDepth.wrap_layers(layers, max_drop_rate=max_rate)
        assert abs(wrapped[-1].drop_prob - max_rate) < 1e-6


# ---------------------------------------------------------------------------
# 5. StochasticDepthTransformer
# ---------------------------------------------------------------------------


class TestStochasticDepthTransformer:
    def _make_transformer(self, drop_rate: float = 0.1) -> StochasticDepthTransformer:
        layers = make_layers()
        cfg = StochasticDepthConfig(drop_rate=drop_rate)
        return StochasticDepthTransformer(layers, cfg)

    def test_output_shape(self):
        x = make_input()
        model = self._make_transformer()
        model.train(False)
        out = model(x)
        assert out.shape == x.shape

    def test_grad_flows(self):
        """Gradients must flow back through the transformer in training mode."""
        x = make_input().requires_grad_(True)
        model = self._make_transformer(drop_rate=0.0)  # never drop so grads always flow
        model.train(True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "gradients must flow to input"
        assert x.grad.shape == x.shape

    def test_num_wrapped_layers(self):
        model = self._make_transformer()
        assert len(model.wrapped_layers) == N_LAYERS


# ---------------------------------------------------------------------------
# 6. get_expected_depth
# ---------------------------------------------------------------------------


class TestGetExpectedDepth:
    def test_all_zero_drop_rates_equals_n_layers(self):
        n = 6
        rates = [0.0] * n
        result = get_expected_depth(n, rates)
        assert abs(result - n) < 1e-6

    def test_uniform_rates(self):
        rates = [0.5, 0.5, 0.5, 0.5]
        result = get_expected_depth(4, rates)
        assert abs(result - 2.0) < 1e-6

    def test_linear_rates(self):
        rates = LinearStochasticDepth.get_drop_rates(4, max_drop_rate=0.3)
        expected = sum(1.0 - r for r in rates)
        result = get_expected_depth(4, rates)
        assert abs(result - expected) < 1e-6

    def test_single_layer_zero_drop(self):
        result = get_expected_depth(1, [0.0])
        assert abs(result - 1.0) < 1e-6
