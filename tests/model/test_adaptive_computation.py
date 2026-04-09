"""Tests for src/model/adaptive_computation.py."""

import torch
import pytest

from src.model.adaptive_computation import (
    ACTConfig,
    HaltingUnit,
    ACTLayer,
    act_ponder_cost,
    ACTTransformer,
    act_efficiency_stats,
)

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

B = 2
T = 8
D = 64
N_LAYERS = 2
MAX_STEPS = 4


def _make_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D)


def _make_config() -> ACTConfig:
    return ACTConfig(max_steps=MAX_STEPS, epsilon=0.01, ponder_cost=0.001, d_model=D)


def _make_act_layer(config: ACTConfig) -> ACTLayer:
    torch.manual_seed(0)
    proxy = torch.nn.Linear(D, D)
    return ACTLayer(proxy, config)


def _make_transformer(config: ACTConfig) -> ACTTransformer:
    torch.manual_seed(0)
    return ACTTransformer(d_model=D, n_layers=N_LAYERS, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_act_config_defaults():
    cfg = ACTConfig()
    assert cfg.max_steps == 8
    assert cfg.epsilon == 0.01
    assert cfg.ponder_cost == 0.001
    assert cfg.d_model == 64


def test_halting_unit_shape():
    torch.manual_seed(0)
    hu = HaltingUnit(D)
    x = _make_input()
    h = hu(x)
    assert h.shape == (B, T), f"Expected ({B}, {T}), got {h.shape}"


def test_halting_unit_range():
    torch.manual_seed(0)
    hu = HaltingUnit(D)
    x = _make_input()
    h = hu(x)
    assert (h > 0).all(), "Halting probs must be > 0"
    assert (h < 1).all(), "Halting probs must be < 1"


def test_act_layer_output_shape():
    config = _make_config()
    act_layer = _make_act_layer(config)
    x = _make_input()
    accumulated, n_updates, remainders = act_layer(x)
    assert accumulated.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {accumulated.shape}"


def test_act_layer_n_updates_shape():
    config = _make_config()
    act_layer = _make_act_layer(config)
    x = _make_input()
    _, n_updates, _ = act_layer(x)
    assert n_updates.shape == (B, T), f"Expected ({B}, {T}), got {n_updates.shape}"


def test_act_layer_n_updates_bounded():
    config = _make_config()
    act_layer = _make_act_layer(config)
    x = _make_input()
    _, n_updates, _ = act_layer(x)
    assert (n_updates >= 1).all(), "Every position must run at least 1 step"
    assert (n_updates <= MAX_STEPS).all(), f"n_updates must not exceed max_steps={MAX_STEPS}"


def test_act_ponder_cost_positive():
    config = _make_config()
    act_layer = _make_act_layer(config)
    x = _make_input()
    _, n_updates, remainders = act_layer(x)
    cost = act_ponder_cost(n_updates, remainders, config.ponder_cost)
    assert cost.item() > 0, "Ponder cost must be positive"


def test_act_ponder_cost_scalar():
    config = _make_config()
    act_layer = _make_act_layer(config)
    x = _make_input()
    _, n_updates, remainders = act_layer(x)
    cost = act_ponder_cost(n_updates, remainders, config.ponder_cost)
    assert cost.shape == torch.Size([]), f"Expected scalar, got shape {cost.shape}"


def test_act_transformer_output_shape():
    config = _make_config()
    model = _make_transformer(config)
    x = _make_input()
    output, _ = model(x)
    assert output.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {output.shape}"


def test_act_transformer_ponder_cost_positive():
    config = _make_config()
    model = _make_transformer(config)
    x = _make_input()
    _, total_ponder_cost = model(x)
    assert total_ponder_cost.item() > 0, "Total ponder cost must be positive"


def test_act_efficiency_stats_keys():
    config = _make_config()
    act_layer = _make_act_layer(config)
    x = _make_input()
    _, n_updates, _ = act_layer(x)
    stats = act_efficiency_stats(n_updates, MAX_STEPS)
    assert "mean_steps" in stats
    assert "max_steps_used" in stats
    assert "early_halt_rate" in stats
    assert isinstance(stats["mean_steps"], float)
    assert isinstance(stats["max_steps_used"], float)
    assert isinstance(stats["early_halt_rate"], float)
