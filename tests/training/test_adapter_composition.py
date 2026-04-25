from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.lora_adapter_manager import LoRAConfig, LoRALayer, LoRAAdapterManager
from src.training.adapter_composition import AdapterComposer, ComposedAdapter, CompositionMode


def _manager_with_adapters(names: list[str]) -> tuple[LoRAAdapterManager, dict]:
    cfg = LoRAConfig(rank=4, alpha=8.0, dropout=0.0, target_modules=["q_proj", "v_proj"])
    mgr = LoRAAdapterManager(config=cfg)
    mod = nn.ModuleDict({"q_proj": nn.Linear(8, 8), "v_proj": nn.Linear(8, 8)})
    adapters = {}
    for name in names:
        adapters[name] = mgr.create_adapter(name, mod)
    return mgr, adapters


# ---- CompositionMode ----

def test_composition_mode_values():
    assert CompositionMode.ADD == "add"
    assert CompositionMode.WEIGHTED == "weighted"
    assert CompositionMode.SEQUENTIAL == "sequential"


# ---- AdapterComposer.compose ----

def test_compose_raises_on_missing_adapter():
    mgr, _ = _manager_with_adapters(["a1"])
    composer = AdapterComposer(mgr)
    with pytest.raises(ValueError, match="a2"):
        composer.compose(["a1", "a2"])


def test_compose_default_weights_sum_to_one():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"])
    assert abs(sum(composed.weights) - 1.0) < 1e-6


def test_compose_normalizes_weights():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"], weights=[2.0, 2.0], mode=CompositionMode.WEIGHTED)
    assert abs(sum(composed.weights) - 1.0) < 1e-6
    assert abs(composed.weights[0] - 0.5) < 1e-6


def test_compose_returns_correct_names():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"])
    assert composed.names == ["a1", "a2"]


def test_compose_single_adapter():
    mgr, _ = _manager_with_adapters(["a1"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1"])
    assert composed.names == ["a1"]
    assert abs(composed.weights[0] - 1.0) < 1e-6


def test_compose_sequential_mode():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"], mode=CompositionMode.SEQUENTIAL)
    assert composed.mode == CompositionMode.SEQUENTIAL


# ---- AdapterComposer.apply ----

def test_apply_add_output_shape():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"], mode=CompositionMode.ADD)
    x = torch.randn(3, 8)
    out = composer.apply(x, composed, "q_proj")
    assert out.shape == (3, 8)


def test_apply_weighted_output_shape():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"], mode=CompositionMode.WEIGHTED)
    x = torch.randn(3, 8)
    out = composer.apply(x, composed, "q_proj")
    assert out.shape == (3, 8)


def test_apply_sequential_output_shape():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"], mode=CompositionMode.SEQUENTIAL)
    x = torch.randn(3, 8)
    out = composer.apply(x, composed, "q_proj")
    assert out.shape == (3, 8)


def test_apply_skips_missing_layer_name():
    mgr, _ = _manager_with_adapters(["a1"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1"], mode=CompositionMode.ADD)
    x = torch.randn(3, 8)
    out = composer.apply(x, composed, "k_proj")
    assert torch.allclose(out, x)


def test_apply_add_at_init_equals_input():
    """With lora_B zeroed, ADD mode should return input unchanged."""
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"], mode=CompositionMode.ADD)
    x = torch.randn(2, 8)
    out = composer.apply(x, composed, "q_proj")
    assert torch.allclose(out, x, atol=1e-6)


def test_apply_weighted_at_init_equals_input():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"], mode=CompositionMode.WEIGHTED)
    x = torch.randn(2, 8)
    out = composer.apply(x, composed, "q_proj")
    assert torch.allclose(out, x, atol=1e-6)


# ---- AdapterComposer.merge_to_delta ----

def test_merge_to_delta_shape():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"])
    delta = composer.merge_to_delta(composed, "q_proj")
    assert delta is not None
    assert delta.shape == (8, 8)


def test_merge_to_delta_none_for_missing_layer():
    mgr, _ = _manager_with_adapters(["a1"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1"])
    delta = composer.merge_to_delta(composed, "k_proj")
    assert delta is None


def test_merge_to_delta_zero_at_init():
    mgr, _ = _manager_with_adapters(["a1", "a2"])
    composer = AdapterComposer(mgr)
    composed = composer.compose(["a1", "a2"])
    delta = composer.merge_to_delta(composed, "q_proj")
    assert delta is not None
    assert torch.all(delta == 0)
