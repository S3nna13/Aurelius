"""Tests for stochastic depth (layer dropout) module."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.stochastic_depth import (
    StochasticDepthConfig,
    StochasticDepthLayer,
    StochasticDepthTrainer,
    compute_expected_depth,
    get_layer_drop_rates,
    stochastic_depth_forward,
    wrap_with_stochastic_depth,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config() -> AureliusConfig:
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


def _small_model() -> AureliusTransformer:
    return AureliusTransformer(_small_config())


# ---------------------------------------------------------------------------
# 1. StochasticDepthConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = StochasticDepthConfig()
    assert cfg.drop_rate == 0.1
    assert cfg.schedule == "linear"
    assert cfg.min_drop_rate == 0.0
    assert cfg.max_drop_rate == 0.5


# ---------------------------------------------------------------------------
# 2. get_layer_drop_rates length == n_layers
# ---------------------------------------------------------------------------


def test_get_layer_drop_rates_length():
    cfg = StochasticDepthConfig(drop_rate=0.2, schedule="uniform")
    rates = get_layer_drop_rates(8, cfg)
    assert len(rates) == 8


# ---------------------------------------------------------------------------
# 3. linear schedule: last > first
# ---------------------------------------------------------------------------


def test_get_layer_drop_rates_linear_last_greater():
    cfg = StochasticDepthConfig(drop_rate=0.3, schedule="linear")
    rates = get_layer_drop_rates(6, cfg)
    assert rates[-1] > rates[0], f"Expected last rate > first rate, got {rates}"


# ---------------------------------------------------------------------------
# 4. uniform schedule: all equal
# ---------------------------------------------------------------------------


def test_get_layer_drop_rates_uniform_all_equal():
    cfg = StochasticDepthConfig(drop_rate=0.15, schedule="uniform")
    rates = get_layer_drop_rates(5, cfg)
    assert all(r == rates[0] for r in rates), f"Expected all equal, got {rates}"


# ---------------------------------------------------------------------------
# 5. clamp to max_drop_rate
# ---------------------------------------------------------------------------


def test_get_layer_drop_rates_clamps_to_max():
    cfg = StochasticDepthConfig(
        drop_rate=0.9,
        schedule="uniform",
        max_drop_rate=0.5,
    )
    rates = get_layer_drop_rates(4, cfg)
    assert all(r <= 0.5 for r in rates), f"Expected all rates <= 0.5, got {rates}"


# ---------------------------------------------------------------------------
# 6. stochastic_depth_forward in eval mode always applies layer
# ---------------------------------------------------------------------------


def test_stochastic_depth_forward_eval_always_applies():
    """In eval mode (training=False), layer must always be applied."""
    # Use a layer that clearly transforms the input (multiply by 2)
    layer = nn.Linear(8, 8, bias=False)
    torch.nn.init.constant_(layer.weight, 2.0)

    x = torch.ones(1, 8)
    results = set()
    for _ in range(20):
        out = stochastic_depth_forward(x, layer, drop_rate=0.9, training=False)
        results.add(round(out.sum().item(), 4))

    # All results should be the same (layer always applied in eval)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# 7. stochastic_depth_forward training drop_rate=1.0 always drops
# ---------------------------------------------------------------------------


def test_stochastic_depth_forward_training_drop_rate_1_always_drops():
    layer = nn.Linear(8, 8, bias=False)
    torch.nn.init.constant_(layer.weight, 2.0)

    x = torch.ones(1, 8)
    for _ in range(10):
        out = stochastic_depth_forward(x, layer, drop_rate=1.0, training=True)
        # Should return x unchanged (all ones)
        assert torch.allclose(out, x), f"Expected x returned, got {out}"


# ---------------------------------------------------------------------------
# 8. stochastic_depth_forward training drop_rate=0.0 always applies
# ---------------------------------------------------------------------------


def test_stochastic_depth_forward_training_drop_rate_0_always_applies():
    layer = nn.Linear(8, 8, bias=False)
    torch.nn.init.constant_(layer.weight, 2.0)

    x = torch.ones(1, 8)
    for _ in range(10):
        out = stochastic_depth_forward(x, layer, drop_rate=0.0, training=True)
        # Layer multiplies everything by 2, so output != input
        assert not torch.allclose(out, x), "Expected layer to be applied"


# ---------------------------------------------------------------------------
# 9. StochasticDepthLayer forward output shape
# ---------------------------------------------------------------------------


def test_stochastic_depth_layer_output_shape():
    inner = nn.Linear(16, 16, bias=False)
    sdl = StochasticDepthLayer(inner, drop_rate=0.3)
    sdl.eval()

    x = torch.randn(2, 16)
    out = sdl(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 10. wrap_with_stochastic_depth output length
# ---------------------------------------------------------------------------


def test_wrap_with_stochastic_depth_output_length():
    layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])
    cfg = StochasticDepthConfig(drop_rate=0.2, schedule="linear")
    wrapped = wrap_with_stochastic_depth(layers, cfg)
    assert len(wrapped) == 6


# ---------------------------------------------------------------------------
# 11. compute_expected_depth all zeros -> n_layers
# ---------------------------------------------------------------------------


def test_compute_expected_depth_all_zeros():
    n = 10
    drop_rates = [0.0] * n
    result = compute_expected_depth(n, drop_rates)
    assert result == float(n), f"Expected {float(n)}, got {result}"


# ---------------------------------------------------------------------------
# 12. StochasticDepthTrainer.train_step returns correct keys
# ---------------------------------------------------------------------------


def test_stochastic_depth_trainer_train_step_keys():
    model = _small_model()
    cfg = StochasticDepthConfig(drop_rate=0.1, schedule="linear")
    trainer = StochasticDepthTrainer(model, cfg)

    input_ids = torch.randint(0, 256, (1, 8))
    labels = torch.randint(0, 256, (1, 8))

    result = trainer.train_step(input_ids, labels)

    assert "loss" in result, f"Missing 'loss' key, got {list(result.keys())}"
    assert "expected_depth" in result, f"Missing 'expected_depth' key, got {list(result.keys())}"
    assert isinstance(result["loss"], float)
    assert isinstance(result["expected_depth"], float)
