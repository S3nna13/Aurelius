"""Tests for src/training/lora_v2.py — LoRALinear / LoRAModel API.

Tiny config: IN=8, OUT=16, R=2, B=2, T=4.
Pure PyTorch only.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.lora_v2 import (
    LoRAConfig,
    LoRALinear,
    LoRAModel,
    apply_lora,
    count_total_parameters,
    count_trainable_parameters,
    freeze_non_lora_params,
)

IN = 8
OUT = 16
R = 2
B = 2
T = 4


# ---------------------------------------------------------------------------
# LoRAConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = LoRAConfig()
    assert cfg.r == 8
    assert cfg.lora_alpha == pytest.approx(16.0)
    assert cfg.lora_dropout == pytest.approx(0.0)
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.merge_weights is False


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------


def test_lora_linear_output_shape():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    x = torch.randn(B, T, IN)
    out = layer(x)
    assert out.shape == (B, T, OUT)


def test_lora_linear_lora_b_zeros_delta_starts_zero():
    """With lora_B initialized to zeros, the LoRA delta should be zero."""
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    x = torch.randn(B, IN)
    base = torch.nn.functional.linear(x, layer.weight, layer.bias)
    out = layer(x)
    assert torch.allclose(out, base, atol=1e-6), "lora_B=0 means delta should be zero"


def test_lora_linear_base_weight_frozen():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    assert not layer.weight.requires_grad


def test_lora_linear_adapter_params_trainable():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_lora_linear_forward_finite():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    x = torch.randn(B, T, IN)
    out = layer(x)
    assert torch.isfinite(out).all()


def test_lora_linear_scaling_formula():
    alpha, r = 8.0, 2
    layer = LoRALinear(IN, OUT, r=r, lora_alpha=alpha)
    assert layer.scaling == pytest.approx(alpha / r)


def test_lora_linear_merge_changes_weight():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    # Give lora_B nonzero values so delta is nonzero
    nn.init.normal_(layer.lora_B)
    weight_before = layer.weight.data.clone()
    layer.merge()
    assert not torch.allclose(layer.weight.data, weight_before)


def test_lora_linear_unmerge_restores_weight():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    nn.init.normal_(layer.lora_B)
    weight_before = layer.weight.data.clone()
    layer.merge()
    layer.unmerge()
    assert torch.allclose(layer.weight.data, weight_before, atol=1e-5)


# ---------------------------------------------------------------------------
# count_trainable_parameters / count_total_parameters
# ---------------------------------------------------------------------------


def test_trainable_less_than_total():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    model = nn.ModuleList([layer])
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    assert trainable < total


def test_count_total_parameters():
    layer = nn.Linear(IN, OUT)
    total = count_total_parameters(layer)
    assert total == IN * OUT + OUT  # weight + bias


# ---------------------------------------------------------------------------
# apply_lora
# ---------------------------------------------------------------------------


def test_apply_lora_copies_weight():
    linear = nn.Linear(IN, OUT, bias=True)
    lora_layer = apply_lora(linear, r=R, lora_alpha=4.0)
    assert torch.allclose(lora_layer.weight.data, linear.weight.data)


def test_apply_lora_copies_bias():
    linear = nn.Linear(IN, OUT, bias=True)
    lora_layer = apply_lora(linear, r=R, lora_alpha=4.0)
    assert lora_layer.bias is not None
    assert torch.allclose(lora_layer.bias.data, linear.bias.data)


# ---------------------------------------------------------------------------
# freeze_non_lora_params
# ---------------------------------------------------------------------------


def test_freeze_non_lora_params():
    layer = LoRALinear(IN, OUT, r=R, lora_alpha=4.0)
    model = nn.ModuleList([layer])
    freeze_non_lora_params(model)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_A" in n or "lora_B" in n for n in trainable_names)


# ---------------------------------------------------------------------------
# LoRAModel
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN, OUT, bias=False)
        self.fc2 = nn.Linear(OUT, IN, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_lora_model_replaces_linears():
    base = _TinyModel()
    model = LoRAModel(base, LoRAConfig(r=R, lora_alpha=4.0))
    from src.training.lora_v2 import LoRALinear

    assert isinstance(model.base_model.fc1, LoRALinear)
    assert isinstance(model.base_model.fc2, LoRALinear)


def test_lora_model_forward_works():
    base = _TinyModel()
    model = LoRAModel(base, LoRAConfig(r=R, lora_alpha=4.0))
    x = torch.randn(B, IN)
    out = model(x)
    assert out.shape == (B, IN)
    assert torch.isfinite(out).all()


def test_lora_model_get_lora_params():
    base = _TinyModel()
    model = LoRAModel(base, LoRAConfig(r=R, lora_alpha=4.0))
    params = model.get_lora_params()
    # 2 linears × (lora_A + lora_B) = 4 tensors
    assert len(params) == 4


def test_lora_model_save_adapter_keys():
    base = _TinyModel()
    model = LoRAModel(base, LoRAConfig(r=R, lora_alpha=4.0))
    adapter = model.save_adapter()
    keys = list(adapter.keys())
    assert any("lora_A" in k for k in keys)
    assert any("lora_B" in k for k in keys)


def test_lora_params_small_fraction():
    """LoRA adapter adds < 10% params for typical sizes."""
    IN_BIG, OUT_BIG = 512, 512
    linear = nn.Linear(IN_BIG, OUT_BIG, bias=False)
    lora = apply_lora(linear, r=4, lora_alpha=8.0)
    # Trainable = lora_A (4×512) + lora_B (512×4) = 4096 + 2048 = 6144 ... wait
    # Actually: lora_A is (r, in) = (4, 512) = 2048, lora_B is (out, r) = (512, 4) = 2048
    # Total base = 512 * 512 = 262144
    trainable = count_trainable_parameters(lora)
    total = count_total_parameters(lora)
    assert trainable / total < 0.1
