"""Tests for src/training/prefix_tuning_v2.py.

Tests the per-layer K/V prefix tuning implementation (Li & Liang, 2021).
Uses tiny dimensions so all tests run quickly on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aurelius.training.prefix_tuning_v2 import (
    PrefixConfig,
    PrefixEmbedding,
    PrefixAttention,
    PrefixModel,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------

B = 2
T = 8
D_MODEL = 32
N_HEADS = 4
HEAD_DIM = 8           # D_MODEL == N_HEADS * HEAD_DIM
N_LAYERS = 2
PREFIX_LEN = 10        # matches PrefixConfig default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> PrefixConfig:
    defaults = dict(
        prefix_length=PREFIX_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        reparameterize=True,
    )
    defaults.update(kwargs)
    return PrefixConfig(**defaults)


def _make_prefix_attn(prefix_length: int = PREFIX_LEN) -> PrefixAttention:
    return PrefixAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        prefix_length=prefix_length,
    )


def _make_prefix_kv(prefix_length: int = PREFIX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (prefix_k, prefix_v) shaped (B, prefix_length, n_heads, head_dim)."""
    shape = (B, prefix_length, N_HEADS, HEAD_DIM)
    return torch.randn(*shape), torch.randn(*shape)


def _tiny_backbone() -> nn.Module:
    """A simple linear backbone that maps (B, T, D_MODEL) → (B, T, D_MODEL)."""
    return nn.Linear(D_MODEL, D_MODEL)


# ---------------------------------------------------------------------------
# 1. PrefixConfig defaults
# ---------------------------------------------------------------------------

def test_prefix_config_defaults():
    cfg = PrefixConfig()
    assert cfg.prefix_length == 10
    assert cfg.n_layers == 4
    assert cfg.n_heads == 4
    assert cfg.head_dim == 16
    assert cfg.reparameterize is True


# ---------------------------------------------------------------------------
# 2. PrefixEmbedding output shape — reparameterize=True
# ---------------------------------------------------------------------------

def test_prefix_embedding_output_shape_reparam():
    cfg = _make_config(reparameterize=True)
    pe = PrefixEmbedding(cfg)
    out = pe()
    expected = (N_LAYERS, 2, PREFIX_LEN, N_HEADS, HEAD_DIM)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. PrefixEmbedding output shape — reparameterize=False
# ---------------------------------------------------------------------------

def test_prefix_embedding_output_shape_no_reparam():
    cfg = _make_config(reparameterize=False)
    pe = PrefixEmbedding(cfg)
    out = pe()
    expected = (N_LAYERS, 2, PREFIX_LEN, N_HEADS, HEAD_DIM)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"


# ---------------------------------------------------------------------------
# 4. reparameterize=False has fewer parameters than reparameterize=True
# ---------------------------------------------------------------------------

def test_prefix_embedding_reparam_false_has_fewer_params():
    cfg_reparam = _make_config(reparameterize=True)
    cfg_direct = _make_config(reparameterize=False)
    pe_reparam = PrefixEmbedding(cfg_reparam)
    pe_direct = PrefixEmbedding(cfg_direct)
    reparam_count = sum(p.numel() for p in pe_reparam.parameters())
    direct_count = sum(p.numel() for p in pe_direct.parameters())
    assert direct_count < reparam_count, (
        f"reparameterize=False should have fewer params ({direct_count}) "
        f"than reparameterize=True ({reparam_count})"
    )


# ---------------------------------------------------------------------------
# 5. PrefixEmbedding output is finite
# ---------------------------------------------------------------------------

def test_prefix_embedding_output_finite():
    cfg = _make_config()
    pe = PrefixEmbedding(cfg)
    out = pe()
    assert torch.isfinite(out).all(), "PrefixEmbedding output contains non-finite values"


# ---------------------------------------------------------------------------
# 6. PrefixEmbedding gradient flows
# ---------------------------------------------------------------------------

def test_prefix_embedding_gradient_flows():
    cfg = _make_config()
    pe = PrefixEmbedding(cfg)
    out = pe()
    loss = out.sum()
    loss.backward()
    # Check at least one parameter received a gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in pe.parameters()
    )
    assert has_grad, "No gradient flowed through PrefixEmbedding"


# ---------------------------------------------------------------------------
# 7. PrefixAttention output shape
# ---------------------------------------------------------------------------

def test_prefix_attention_output_shape():
    attn = _make_prefix_attn()
    x = torch.randn(B, T, D_MODEL)
    prefix_k, prefix_v = _make_prefix_kv()
    out = attn(x, prefix_k, prefix_v)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B},{T},{D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 8. PrefixAttention output is finite
# ---------------------------------------------------------------------------

def test_prefix_attention_output_finite():
    attn = _make_prefix_attn()
    x = torch.randn(B, T, D_MODEL)
    prefix_k, prefix_v = _make_prefix_kv()
    out = attn(x, prefix_k, prefix_v)
    assert torch.isfinite(out).all(), "PrefixAttention output contains non-finite values"


# ---------------------------------------------------------------------------
# 9. PrefixAttention gradient flows
# ---------------------------------------------------------------------------

def test_prefix_attention_gradient_flows():
    attn = _make_prefix_attn()
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    prefix_k, prefix_v = _make_prefix_kv()
    prefix_k.requires_grad_(True)
    prefix_v.requires_grad_(True)
    out = attn(x, prefix_k, prefix_v)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "No gradient w.r.t. x"
    assert prefix_k.grad is not None and prefix_k.grad.abs().sum() > 0, "No gradient w.r.t. prefix_k"
    assert prefix_v.grad is not None and prefix_v.grad.abs().sum() > 0, "No gradient w.r.t. prefix_v"


# ---------------------------------------------------------------------------
# 10. PrefixAttention with prefix_length=0 produces standard attention shapes
# ---------------------------------------------------------------------------

def test_prefix_attention_zero_prefix_length():
    """prefix_length=0 should behave like standard self-attention."""
    attn_zero = _make_prefix_attn(prefix_length=0)
    attn_std = _make_prefix_attn(prefix_length=PREFIX_LEN)

    x = torch.randn(B, T, D_MODEL)
    # With prefix_length=0, pass empty prefix tensors
    pk_empty = torch.empty(B, 0, N_HEADS, HEAD_DIM)
    pv_empty = torch.empty(B, 0, N_HEADS, HEAD_DIM)
    out_zero = attn_zero(x, pk_empty, pv_empty)
    out_std = attn_std(x, *_make_prefix_kv())

    # Both produce the same sequence shape
    assert out_zero.shape == (B, T, D_MODEL), f"Expected ({B},{T},{D_MODEL}), got {out_zero.shape}"
    assert out_std.shape == (B, T, D_MODEL), f"Expected ({B},{T},{D_MODEL}), got {out_std.shape}"


# ---------------------------------------------------------------------------
# 11. PrefixModel.freeze_backbone freezes all backbone params
# ---------------------------------------------------------------------------

def test_prefix_model_freeze_backbone():
    backbone = _tiny_backbone()
    cfg = _make_config()
    model = PrefixModel(backbone, cfg)
    model.freeze_backbone()
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, (
            f"Backbone param '{name}' should be frozen after freeze_backbone()"
        )


# ---------------------------------------------------------------------------
# 12. PrefixModel.trainable_parameters returns only prefix params
# ---------------------------------------------------------------------------

def test_prefix_model_trainable_parameters_only_prefix():
    backbone = _tiny_backbone()
    cfg = _make_config()
    model = PrefixModel(backbone, cfg)
    model.freeze_backbone()

    trainable = model.trainable_parameters()
    assert len(trainable) > 0, "trainable_parameters() returned empty list"

    # All returned params must require grad
    for p in trainable:
        assert p.requires_grad, "trainable_parameters() returned a frozen param"

    # None of them should be backbone params
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    for p in trainable:
        assert id(p) not in backbone_ids, (
            "trainable_parameters() included a backbone parameter"
        )


# ---------------------------------------------------------------------------
# 13. Works with prefix_length=1 (minimal prefix)
# ---------------------------------------------------------------------------

def test_prefix_length_1():
    cfg = _make_config(prefix_length=1)
    pe = PrefixEmbedding(cfg)
    out = pe()
    assert out.shape == (N_LAYERS, 2, 1, N_HEADS, HEAD_DIM)

    attn = _make_prefix_attn(prefix_length=1)
    x = torch.randn(B, T, D_MODEL)
    pk, pv = _make_prefix_kv(prefix_length=1)
    attn_out = attn(x, pk, pv)
    assert attn_out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 14. Works with prefix_length=20 (larger prefix)
# ---------------------------------------------------------------------------

def test_prefix_length_20():
    cfg = _make_config(prefix_length=20)
    pe = PrefixEmbedding(cfg)
    out = pe()
    assert out.shape == (N_LAYERS, 2, 20, N_HEADS, HEAD_DIM)

    attn = _make_prefix_attn(prefix_length=20)
    x = torch.randn(B, T, D_MODEL)
    pk, pv = _make_prefix_kv(prefix_length=20)
    attn_out = attn(x, pk, pv)
    assert attn_out.shape == (B, T, D_MODEL)
    assert torch.isfinite(attn_out).all()


# ---------------------------------------------------------------------------
# 15. PrefixModel forward stores last_prefix and returns backbone output
# ---------------------------------------------------------------------------

def test_prefix_model_forward_stores_last_prefix():
    backbone = nn.Identity()
    cfg = _make_config()
    model = PrefixModel(backbone, cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    # Output should be same as Identity(x)
    assert out.shape == (B, T, D_MODEL)
    # last_prefix should be stored
    assert model.last_prefix is not None
    expected_prefix_shape = (N_LAYERS, 2, PREFIX_LEN, N_HEADS, HEAD_DIM)
    assert model.last_prefix.shape == expected_prefix_shape, (
        f"Expected last_prefix shape {expected_prefix_shape}, "
        f"got {model.last_prefix.shape}"
    )
