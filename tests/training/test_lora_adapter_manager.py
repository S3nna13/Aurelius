from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.lora_adapter_manager import LoRAConfig, LoRALayer, LoRAAdapterManager


def _tiny_module() -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "q_proj": nn.Linear(8, 8),
            "v_proj": nn.Linear(8, 8),
        }
    )


def _manager(target: list[str] | None = None) -> LoRAAdapterManager:
    cfg = LoRAConfig(
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=target or ["q_proj", "v_proj"],
    )
    return LoRAAdapterManager(config=cfg)


# ---- LoRAConfig ----

def test_loraconfig_defaults():
    cfg = LoRAConfig()
    assert cfg.rank == 8
    assert cfg.alpha == 16.0
    assert cfg.dropout == 0.05
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.merge_weights is False


def test_loraconfig_custom():
    cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.1, target_modules=["k_proj"])
    assert cfg.rank == 16
    assert cfg.target_modules == ["k_proj"]


# ---- LoRALayer ----

def test_lora_layer_init_shapes():
    layer = LoRALayer(in_features=8, out_features=8, rank=4, alpha=8.0, dropout=0.0)
    assert layer.lora_A.weight.shape == (4, 8)
    assert layer.lora_B.weight.shape == (8, 4)


def test_lora_layer_B_init_zero():
    layer = LoRALayer(in_features=8, out_features=8, rank=4, alpha=8.0, dropout=0.0)
    assert torch.all(layer.lora_B.weight == 0)


def test_lora_layer_scaling():
    layer = LoRALayer(in_features=8, out_features=8, rank=4, alpha=8.0, dropout=0.0)
    assert layer.scaling == pytest.approx(2.0)


def test_lora_layer_forward_shape():
    layer = LoRALayer(in_features=8, out_features=8, rank=4, alpha=8.0, dropout=0.0)
    x = torch.randn(3, 8)
    out = layer(x)
    assert out.shape == (3, 8)


def test_lora_layer_forward_identity_at_init():
    """With lora_B zeroed, output should equal input at init."""
    layer = LoRALayer(in_features=8, out_features=8, rank=4, alpha=8.0, dropout=0.0)
    x = torch.randn(2, 8)
    out = layer(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_lora_layer_merge_shape():
    layer = LoRALayer(in_features=8, out_features=8, rank=4, alpha=8.0, dropout=0.0)
    delta = layer.merge()
    assert delta.shape == (8, 8)


def test_lora_layer_merge_zero_at_init():
    layer = LoRALayer(in_features=8, out_features=8, rank=4, alpha=8.0, dropout=0.0)
    delta = layer.merge()
    assert torch.all(delta == 0)


# ---- LoRAAdapterManager ----

def test_manager_default_config():
    mgr = LoRAAdapterManager()
    assert mgr.config.rank == 8


def test_create_adapter_returns_layers():
    mgr = _manager()
    mod = _tiny_module()
    layers = mgr.create_adapter("a1", mod)
    assert len(layers) == 2
    assert all(isinstance(v, LoRALayer) for v in layers.values())


def test_create_adapter_respects_target_modules():
    mgr = _manager(target=["q_proj"])
    mod = _tiny_module()
    layers = mgr.create_adapter("a1", mod)
    assert len(layers) == 1
    assert any("q_proj" in k for k in layers)


def test_create_adapter_registers_name():
    mgr = _manager()
    mod = _tiny_module()
    mgr.create_adapter("a1", mod)
    assert "a1" in mgr.list_adapters()


def test_get_adapter_returns_none_for_missing():
    mgr = _manager()
    assert mgr.get_adapter("nonexistent") is None


def test_get_adapter_returns_registered():
    mgr = _manager()
    mod = _tiny_module()
    mgr.create_adapter("a1", mod)
    assert mgr.get_adapter("a1") is not None


def test_register_adapter_manual():
    mgr = _manager()
    layer = LoRALayer(8, 8, 4, 8.0, 0.0)
    mgr.register_adapter("manual", {"q_proj": layer})
    assert "manual" in mgr.list_adapters()


def test_list_adapters_empty():
    mgr = _manager()
    assert mgr.list_adapters() == []


def test_list_adapters_multiple():
    mgr = _manager()
    mod = _tiny_module()
    mgr.create_adapter("a1", mod)
    mgr.create_adapter("a2", mod)
    names = mgr.list_adapters()
    assert "a1" in names and "a2" in names


def test_remove_adapter_returns_true():
    mgr = _manager()
    mod = _tiny_module()
    mgr.create_adapter("a1", mod)
    assert mgr.remove_adapter("a1") is True


def test_remove_adapter_missing_returns_false():
    mgr = _manager()
    assert mgr.remove_adapter("gone") is False


def test_remove_adapter_clears_entry():
    mgr = _manager()
    mod = _tiny_module()
    mgr.create_adapter("a1", mod)
    mgr.remove_adapter("a1")
    assert mgr.get_adapter("a1") is None


def test_adapter_params_positive():
    mgr = _manager()
    mod = _tiny_module()
    mgr.create_adapter("a1", mod)
    params = mgr.adapter_params("a1")
    assert params > 0


def test_adapter_params_missing_returns_zero():
    mgr = _manager()
    assert mgr.adapter_params("nope") == 0


def test_adapter_params_count():
    mgr = _manager()
    mod = _tiny_module()
    mgr.create_adapter("a1", mod)
    layers = mgr.get_adapter("a1")
    expected = sum(p.numel() for layer in layers.values() for p in layer.parameters())
    assert mgr.adapter_params("a1") == expected
