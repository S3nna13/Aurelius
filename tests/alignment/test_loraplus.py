"""Tests for LoRA++ and RSLoRA implementation."""

from __future__ import annotations

import math

import torch

from src.alignment.loraplus import (
    LoRAPlusConfig,
    LoRAPlusLinear,
    create_loraplus_optimizer,
)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_loraplus_config_defaults():
    cfg = LoRAPlusConfig()
    assert cfg.rank == 8
    assert cfg.alpha == 16.0
    assert cfg.lr_ratio == 16.0


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_loraplus_forward_shape():
    cfg = LoRAPlusConfig(rank=4, alpha=8.0)
    layer = LoRAPlusLinear(in_features=32, out_features=16, cfg=cfg)
    x = torch.randn(3, 32)
    out = layer(x)
    assert out.shape == (3, 16)


# ---------------------------------------------------------------------------
# Parameter properties
# ---------------------------------------------------------------------------


def test_base_weight_frozen():
    cfg = LoRAPlusConfig()
    layer = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)
    assert layer.weight.requires_grad is False


def test_a_b_trainable():
    cfg = LoRAPlusConfig()
    layer = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)
    assert layer.A.requires_grad is True
    assert layer.B.requires_grad is True


def test_a_b_initialized():
    cfg = LoRAPlusConfig(rank=4)
    layer = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)
    # B must be all zeros (standard LoRA init)
    assert torch.all(layer.B == 0).item()
    # A must NOT be all zeros (kaiming init)
    assert not torch.all(layer.A == 0).item()


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_scaling_standard():
    rank = 8
    alpha = 16.0
    cfg = LoRAPlusConfig(rank=rank, alpha=alpha, use_rslora=False)
    layer = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)
    assert math.isclose(layer.scaling, alpha / rank)


def test_scaling_rslora():
    rank = 8
    cfg = LoRAPlusConfig(rank=rank, use_rslora=True)
    layer = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)
    assert math.isclose(layer.scaling, 1.0 / math.sqrt(rank))


def test_rslora_vs_standard_different():
    torch.manual_seed(42)
    in_f, out_f = 32, 16
    rank = 8

    cfg_std = LoRAPlusConfig(rank=rank, alpha=16.0, use_rslora=False)
    cfg_rs = LoRAPlusConfig(rank=rank, alpha=16.0, use_rslora=True)

    std_layer = LoRAPlusLinear(in_f, out_f, cfg_std)
    rs_layer = LoRAPlusLinear(in_f, out_f, cfg_rs)

    # Copy A from std to rs so only scaling differs
    with torch.no_grad():
        rs_layer.A.copy_(std_layer.A)
        rs_layer.B.copy_(std_layer.B)
        rs_layer.weight.copy_(std_layer.weight)

    # Give A/B non-zero values so the LoRA term actually contributes
    with torch.no_grad():
        std_layer.A.fill_(0.1)
        std_layer.B.fill_(0.1)
        rs_layer.A.fill_(0.1)
        rs_layer.B.fill_(0.1)

    x = torch.randn(2, in_f)
    out_std = std_layer(x)
    out_rs = rs_layer(x)

    assert not torch.allclose(out_std, out_rs), (
        "RSLoRA and standard LoRA should produce different outputs when scalings differ"
    )


# ---------------------------------------------------------------------------
# Param groups / asymmetric LR
# ---------------------------------------------------------------------------


def test_get_param_groups_lr_ratio():
    lr_ratio = 16.0
    cfg = LoRAPlusConfig(rank=4, lr_ratio=lr_ratio)
    layer = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)

    base_lr = 1e-4
    groups = layer.get_param_groups(base_lr)

    assert len(groups) == 2
    lr_a = groups[0]["lr"]
    lr_b = groups[1]["lr"]
    assert math.isclose(lr_a, base_lr)
    assert math.isclose(lr_b, base_lr * lr_ratio)


def test_create_optimizer_asymmetric_lr():
    cfg = LoRAPlusConfig(rank=4, lr_ratio=16.0)
    layer1 = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)
    layer2 = LoRAPlusLinear(in_features=8, out_features=4, cfg=cfg)

    base_lr = 2e-4
    opt = create_loraplus_optimizer([layer1, layer2], base_lr=base_lr)

    # 2 layers * 2 groups each = 4 param groups
    assert len(opt.param_groups) == 4

    [pg["lr"] for pg in opt.param_groups]
    # Every other group should be B (high lr)
    for i, pg in enumerate(opt.param_groups):
        if i % 2 == 0:
            assert math.isclose(pg["lr"], base_lr), f"Group {i} (A) lr mismatch"
        else:
            assert math.isclose(pg["lr"], base_lr * cfg.lr_ratio), f"Group {i} (B) lr mismatch"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_loraplus_gradient_flows():
    cfg = LoRAPlusConfig(rank=4, alpha=8.0)
    layer = LoRAPlusLinear(in_features=16, out_features=8, cfg=cfg)

    x = torch.randn(2, 16)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.A.grad is not None, "A should have gradients"
    assert layer.B.grad is not None, "B should have gradients"
    assert layer.weight.grad is None, "Frozen weight should have no gradients"
