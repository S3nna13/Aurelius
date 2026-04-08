"""Tests for PrefixTuning: layer-wise soft prefix tuning (Li & Liang 2021)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.prefix_tuning import (
    PrefixConfig,
    PrefixTuning,
    apply_prefix_to_attention,
    PrefixTuningTrainer,
)

# Small dimensions for fast CPU tests
CFG_SMALL = PrefixConfig(
    prefix_length=4,
    n_layers=3,
    n_kv_heads=2,
    head_dim=8,
    dropout=0.0,
    use_mlp_reparameterization=True,
)

CFG_SMALL_NO_MLP = PrefixConfig(
    prefix_length=4,
    n_layers=3,
    n_kv_heads=2,
    head_dim=8,
    dropout=0.0,
    use_mlp_reparameterization=False,
)


# ---------------------------------------------------------------------------
# 1. test_prefix_config_defaults
# ---------------------------------------------------------------------------

def test_prefix_config_defaults():
    cfg = PrefixConfig()
    assert cfg.prefix_length == 10
    assert cfg.n_layers == 24
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.dropout == 0.1
    assert cfg.use_mlp_reparameterization is True


# ---------------------------------------------------------------------------
# 2. test_prefix_tuning_get_prefix_kv_shape
# ---------------------------------------------------------------------------

def test_prefix_tuning_get_prefix_kv_shape():
    model = PrefixTuning(CFG_SMALL)
    prefix_k, prefix_v = model.get_prefix_kv(0)
    expected = (CFG_SMALL.prefix_length, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    assert prefix_k.shape == expected, f"prefix_k shape {prefix_k.shape} != {expected}"
    assert prefix_v.shape == expected, f"prefix_v shape {prefix_v.shape} != {expected}"


# ---------------------------------------------------------------------------
# 3. test_prefix_tuning_all_layers_count
# ---------------------------------------------------------------------------

def test_prefix_tuning_all_layers_count():
    model = PrefixTuning(CFG_SMALL)
    all_kvs = model.get_all_prefix_kvs()
    assert len(all_kvs) == CFG_SMALL.n_layers, (
        f"Expected {CFG_SMALL.n_layers} entries, got {len(all_kvs)}"
    )


# ---------------------------------------------------------------------------
# 4. test_prefix_tuning_with_mlp
# ---------------------------------------------------------------------------

def test_prefix_tuning_with_mlp():
    cfg = PrefixConfig(
        prefix_length=4, n_layers=3, n_kv_heads=2, head_dim=8,
        dropout=0.0, use_mlp_reparameterization=True,
    )
    model = PrefixTuning(cfg)
    prefix_k, prefix_v = model.get_prefix_kv(0)
    assert prefix_k.shape == (4, 2, 8)
    assert prefix_v.shape == (4, 2, 8)


# ---------------------------------------------------------------------------
# 5. test_prefix_tuning_without_mlp
# ---------------------------------------------------------------------------

def test_prefix_tuning_without_mlp():
    cfg = PrefixConfig(
        prefix_length=4, n_layers=3, n_kv_heads=2, head_dim=8,
        dropout=0.0, use_mlp_reparameterization=False,
    )
    model = PrefixTuning(cfg)
    prefix_k, prefix_v = model.get_prefix_kv(0)
    assert prefix_k.shape == (4, 2, 8)
    assert prefix_v.shape == (4, 2, 8)


# ---------------------------------------------------------------------------
# 6. test_apply_prefix_to_attention_shape
# ---------------------------------------------------------------------------

def test_apply_prefix_to_attention_shape():
    B, S = 2, 6
    prefix_model = PrefixTuning(CFG_SMALL)
    k = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    v = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)

    new_k, new_v = apply_prefix_to_attention(prefix_model, k, v, layer_idx=0)

    expected_seq = CFG_SMALL.prefix_length + S
    assert new_k.shape == (B, expected_seq, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim), (
        f"Expected k shape (B, {expected_seq}, n_kv_heads, head_dim), got {new_k.shape}"
    )
    assert new_v.shape == (B, expected_seq, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim), (
        f"Expected v shape (B, {expected_seq}, n_kv_heads, head_dim), got {new_v.shape}"
    )


# ---------------------------------------------------------------------------
# 7. test_apply_prefix_expands_batch
# ---------------------------------------------------------------------------

def test_apply_prefix_expands_batch():
    B, S = 4, 10
    prefix_model = PrefixTuning(CFG_SMALL)
    k = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    v = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)

    new_k, new_v = apply_prefix_to_attention(prefix_model, k, v, layer_idx=1)

    # Prefix portion should be the same for all batch items
    prefix_k_slice = new_k[:, : CFG_SMALL.prefix_length]  # (B, prefix_len, kv_heads, head_dim)
    assert prefix_k_slice.shape[0] == B, "Batch dim should be B after expansion"
    # All batch items have the same prefix values
    assert torch.allclose(prefix_k_slice[0], prefix_k_slice[1]), (
        "Prefix should be identical across batch items"
    )


# ---------------------------------------------------------------------------
# 8. test_prefix_tuning_trainer_freeze
# ---------------------------------------------------------------------------

def make_simple_backbone() -> nn.Module:
    """Tiny linear backbone for trainer tests."""
    return nn.Linear(16, 16)


def test_prefix_tuning_trainer_freeze():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    frozen_count = trainer.freeze_backbone()
    assert frozen_count > 0, "freeze_backbone should return > 0 frozen params"
    for p in backbone.parameters():
        assert not p.requires_grad, "Backbone params should be frozen after freeze_backbone()"


# ---------------------------------------------------------------------------
# 9. test_prefix_tuning_trainer_unfreeze
# ---------------------------------------------------------------------------

def test_prefix_tuning_trainer_unfreeze():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    trainer.freeze_backbone()
    trainer.unfreeze_backbone()
    for p in backbone.parameters():
        assert p.requires_grad, "Backbone params should be trainable after unfreeze_backbone()"


# ---------------------------------------------------------------------------
# 10. test_prefix_tuning_trainer_trainable_params
# ---------------------------------------------------------------------------

def test_prefix_tuning_trainer_trainable_params():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    trainer.freeze_backbone()
    trainable = trainer.trainable_params()

    # All returned params must have requires_grad=True
    for p in trainable:
        assert p.requires_grad, "trainable_params() should only return grad-enabled params"

    # Must be exactly the prefix_tuning parameters
    prefix_param_ids = {id(p) for p in prefix_model.parameters()}
    trainable_ids = {id(p) for p in trainable}
    assert trainable_ids == prefix_param_ids, (
        "trainable_params() should return exactly the prefix_tuning parameters"
    )


# ---------------------------------------------------------------------------
# 11. test_prefix_tuning_param_count
# ---------------------------------------------------------------------------

def test_prefix_tuning_param_count():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    trainer.freeze_backbone()
    counts = trainer.param_count()

    assert "total" in counts
    assert "trainable" in counts
    assert "frozen" in counts
    assert counts["total"] == counts["trainable"] + counts["frozen"], (
        "total must equal trainable + frozen"
    )
    assert counts["trainable"] > 0, "trainable count should be > 0"
    assert counts["frozen"] > 0, "frozen count should be > 0 (backbone is frozen)"


# ---------------------------------------------------------------------------
# 12. test_prefix_tuning_gradients_flow
# ---------------------------------------------------------------------------

def test_prefix_tuning_gradients_flow():
    B, S = 2, 5
    prefix_model = PrefixTuning(CFG_SMALL)
    k = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    v = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)

    new_k, new_v = apply_prefix_to_attention(prefix_model, k, v, layer_idx=0)
    loss = new_k.sum() + new_v.sum()
    loss.backward()

    # At least one prefix parameter should have a gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in prefix_model.parameters()
    )
    assert has_grad, "At least one prefix parameter should have a non-zero gradient after backward()"
