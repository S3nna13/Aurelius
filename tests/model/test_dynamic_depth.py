"""Tests for dynamic_depth.py — DynamicDepthTransformer and related components."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.dynamic_depth import (
    AdaptiveLayerSelector,
    DynamicDepthConfig,
    DynamicDepthTransformer,
    ExitRouter,
    LayerSkipRouter,
    compute_exit_confidence,
)
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """Tiny model config suitable for fast CPU tests."""
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
def dd_cfg():
    return DynamicDepthConfig(
        exit_threshold=0.9,
        skip_threshold=0.1,
        min_layers=1,
        temperature=1.0,
        use_learned_router=True,
    )


@pytest.fixture
def base_model(cfg):
    torch.manual_seed(0)
    return AureliusTransformer(cfg)


@pytest.fixture
def dd_model(base_model, dd_cfg):
    return DynamicDepthTransformer(base_model, dd_cfg)


@pytest.fixture
def input_ids(cfg):
    torch.manual_seed(42)
    return torch.randint(0, cfg.vocab_size, (3, 8))


# ---------------------------------------------------------------------------
# 1. DynamicDepthConfig defaults
# ---------------------------------------------------------------------------


def test_dynamic_depth_config_defaults():
    cfg = DynamicDepthConfig()
    assert cfg.exit_threshold == 0.9
    assert cfg.skip_threshold == 0.1
    assert cfg.min_layers == 1
    assert cfg.temperature == 1.0
    assert cfg.use_learned_router is True


# ---------------------------------------------------------------------------
# 2. ExitRouter output shape (B, 1)
# ---------------------------------------------------------------------------


def test_exit_router_output_shape():
    router = ExitRouter(d_model=64)
    B = 4
    x = torch.randn(B, 64)
    out = router(x)
    assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. ExitRouter output in [0, 1]
# ---------------------------------------------------------------------------


def test_exit_router_output_range():
    torch.manual_seed(1)
    router = ExitRouter(d_model=64)
    x = torch.randn(8, 64) * 10.0  # large values to stress sigmoid
    out = router(x)
    assert out.min().item() >= 0.0, "ExitRouter output below 0"
    assert out.max().item() <= 1.0, "ExitRouter output above 1"


# ---------------------------------------------------------------------------
# 4. LayerSkipRouter output shape and range
# ---------------------------------------------------------------------------


def test_layer_skip_router_output_shape_and_range():
    router = LayerSkipRouter(d_model=32)
    B = 5
    x = torch.randn(B, 32) * 5.0
    out = router(x)
    assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"
    assert out.min().item() >= 0.0, "LayerSkipRouter output below 0"
    assert out.max().item() <= 1.0, "LayerSkipRouter output above 1"


# ---------------------------------------------------------------------------
# 5. compute_exit_confidence shape (B,) and range [0, 1]
# ---------------------------------------------------------------------------


def test_compute_exit_confidence_shape_and_range():
    B, V = 6, 256
    logits = torch.randn(B, V)
    conf = compute_exit_confidence(logits)
    assert conf.shape == (B,), f"Expected ({B},), got {conf.shape}"
    assert conf.min().item() >= 0.0, "Confidence below 0"
    assert conf.max().item() <= 1.0, "Confidence above 1"


# ---------------------------------------------------------------------------
# 6. DynamicDepthTransformer.forward returns 3-tuple
# ---------------------------------------------------------------------------


def test_dynamic_depth_transformer_returns_3_tuple(dd_model, input_ids):
    result = dd_model(input_ids)
    assert isinstance(result, tuple), "forward should return a tuple"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"


# ---------------------------------------------------------------------------
# 7. DynamicDepthTransformer logits shape (B, T, V)
# ---------------------------------------------------------------------------


def test_dynamic_depth_transformer_logits_shape(dd_model, input_ids, cfg):
    B, T = input_ids.shape
    _loss, logits, _exit_layers = dd_model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# 8. DynamicDepthTransformer exit_layers list length == B
# ---------------------------------------------------------------------------


def test_dynamic_depth_transformer_exit_layers_length(dd_model, input_ids):
    B = input_ids.shape[0]
    _loss, _logits, exit_layers = dd_model(input_ids)
    assert len(exit_layers) == B, f"exit_layers length should be {B}, got {len(exit_layers)}"


# ---------------------------------------------------------------------------
# 9. DynamicDepthTransformer exit layer index <= n_layers - 1
# ---------------------------------------------------------------------------


def test_dynamic_depth_transformer_exit_layer_upper_bound(dd_model, input_ids, cfg):
    _loss, _logits, exit_layers = dd_model(input_ids)
    for idx, e in enumerate(exit_layers):
        assert e <= cfg.n_layers - 1, (
            f"Batch element {idx} exited at layer {e}, max allowed {cfg.n_layers - 1}"
        )


# ---------------------------------------------------------------------------
# 10. compute_efficiency_stats returns required keys
# ---------------------------------------------------------------------------


def test_compute_efficiency_stats_keys(dd_model, input_ids):
    _loss, _logits, exit_layers = dd_model(input_ids)
    stats = dd_model.compute_efficiency_stats(exit_layers)
    required = {"mean_exit_layer", "min_exit_layer", "max_exit_layer", "early_exit_rate"}
    assert required.issubset(stats.keys()), f"Missing keys: {required - stats.keys()}"


# ---------------------------------------------------------------------------
# 11. compute_efficiency_stats early_exit_rate in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_efficiency_stats_early_exit_rate_range(dd_model, input_ids):
    _loss, _logits, exit_layers = dd_model(input_ids)
    stats = dd_model.compute_efficiency_stats(exit_layers)
    rate = stats["early_exit_rate"]
    assert 0.0 <= rate <= 1.0, f"early_exit_rate {rate} out of [0, 1]"


# ---------------------------------------------------------------------------
# 12. AdaptiveLayerSelector.select_layers returns list of ints
# ---------------------------------------------------------------------------


def test_adaptive_layer_selector_returns_list_of_ints():
    selector = AdaptiveLayerSelector(n_layers=6, min_layers=2)
    hidden = torch.randn(2, 8, 64)
    layers = selector.select_layers(hidden)
    assert isinstance(layers, list), "select_layers should return a list"
    assert all(isinstance(i, int) for i in layers), "All elements should be int"


# ---------------------------------------------------------------------------
# 13. AdaptiveLayerSelector.select_layers length >= min_layers
# ---------------------------------------------------------------------------


def test_adaptive_layer_selector_min_layers():
    min_layers = 3
    selector = AdaptiveLayerSelector(n_layers=8, min_layers=min_layers)
    # Test with a zero tensor (low norm) and a large tensor (high norm)
    for hidden in [torch.zeros(1, 4, 64), torch.randn(1, 4, 64) * 100]:
        layers = selector.select_layers(hidden)
        assert len(layers) >= min_layers, f"Expected >= {min_layers} layers, got {len(layers)}"


# ---------------------------------------------------------------------------
# 14. AdaptiveLayerSelector.select_layers all indices < n_layers
# ---------------------------------------------------------------------------


def test_adaptive_layer_selector_indices_in_range():
    n_layers = 5
    selector = AdaptiveLayerSelector(n_layers=n_layers, min_layers=2)
    for _ in range(5):
        hidden = torch.randn(2, 10, 64) * torch.rand(1).item() * 20
        layers = selector.select_layers(hidden)
        for idx in layers:
            assert idx < n_layers, f"Layer index {idx} is out of range [0, {n_layers})"
