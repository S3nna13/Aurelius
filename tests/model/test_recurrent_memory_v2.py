"""Tests for src/model/recurrent_memory_v2.py

Tiny config: D=16, N_MEM=4, D_MEM=8, B=2, T=8
"""

from __future__ import annotations

import pytest
import torch

from src.model.recurrent_memory_v2 import (
    GatedMemoryUpdate,
    MemoryAttention,
    RecurrentConfig,
    RecurrentTransformerLayer,
    apply_state_decay,
    init_memory,
)

D = 16
D_MEM = 8
N_MEM = 4
B = 2
T = 8


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = RecurrentConfig()
    assert cfg.d_model == 512
    assert cfg.n_memory_tokens == 16
    assert cfg.d_memory == 64
    assert cfg.chunk_size == 512
    assert cfg.state_decay == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# GatedMemoryUpdate
# ---------------------------------------------------------------------------


def test_gated_memory_update_shape():
    update = GatedMemoryUpdate(D, D_MEM)
    h = torch.randn(B, D)
    m = torch.randn(B, D_MEM)
    out = update(h, m)
    assert out.shape == (B, D_MEM)


def test_gated_memory_update_changes_values():
    update = GatedMemoryUpdate(D, D_MEM)
    h = torch.randn(B, D)
    m = torch.randn(B, D_MEM)
    m_new = update(h, m)
    assert not torch.allclose(m_new, m)


def test_gated_memory_update_output_finite():
    update = GatedMemoryUpdate(D, D_MEM)
    h = torch.randn(B, D)
    m = torch.randn(B, D_MEM)
    out = update(h, m)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MemoryAttention
# ---------------------------------------------------------------------------


def test_memory_attention_shape():
    attn = MemoryAttention(D, N_MEM)
    hidden = torch.randn(B, T, D)
    mem = torch.randn(B, N_MEM, D)
    out = attn(hidden, mem)
    assert out.shape == (B, T, D)


def test_memory_attention_output_finite():
    attn = MemoryAttention(D, N_MEM)
    hidden = torch.randn(B, T, D)
    mem = torch.randn(B, N_MEM, D)
    out = attn(hidden, mem)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# RecurrentTransformerLayer
# ---------------------------------------------------------------------------


def test_recurrent_layer_output_x_shape():
    cfg = RecurrentConfig(d_model=D, n_memory_tokens=N_MEM)
    layer = RecurrentTransformerLayer(cfg)
    x = torch.randn(B, T, D)
    mem = torch.randn(B, N_MEM, D)
    out_x, out_mem = layer(x, mem)
    assert out_x.shape == (B, T, D)


def test_recurrent_layer_output_memory_shape():
    cfg = RecurrentConfig(d_model=D, n_memory_tokens=N_MEM)
    layer = RecurrentTransformerLayer(cfg)
    x = torch.randn(B, T, D)
    mem = torch.randn(B, N_MEM, D)
    out_x, out_mem = layer(x, mem)
    assert out_mem.shape == (B, N_MEM, D)


def test_recurrent_layer_output_finite():
    cfg = RecurrentConfig(d_model=D, n_memory_tokens=N_MEM)
    layer = RecurrentTransformerLayer(cfg)
    x = torch.randn(B, T, D)
    mem = torch.randn(B, N_MEM, D)
    out_x, out_mem = layer(x, mem)
    assert torch.isfinite(out_x).all()
    assert torch.isfinite(out_mem).all()


def test_recurrent_layer_memory_changes():
    cfg = RecurrentConfig(d_model=D, n_memory_tokens=N_MEM)
    layer = RecurrentTransformerLayer(cfg)
    x = torch.randn(B, T, D)
    mem = torch.zeros(B, N_MEM, D)
    _, out_mem = layer(x, mem)
    assert not torch.allclose(out_mem, mem)


# ---------------------------------------------------------------------------
# init_memory / apply_state_decay
# ---------------------------------------------------------------------------


def test_init_memory_shape_and_zeros():
    mem = init_memory(B, N_MEM, D)
    assert mem.shape == (B, N_MEM, D)
    assert torch.all(mem == 0)


def test_apply_state_decay_reduces_magnitude():
    mem = torch.randn(B, N_MEM, D) * 10
    decayed = apply_state_decay(mem, 0.5)
    assert decayed.abs().mean() < mem.abs().mean()


def test_sequential_memory_updates_preserve_shape():
    cfg = RecurrentConfig(d_model=D, n_memory_tokens=N_MEM)
    layer = RecurrentTransformerLayer(cfg)
    mem = init_memory(B, N_MEM, D)
    for _ in range(3):
        x = torch.randn(B, T, D)
        _, mem = layer(x, mem)
    assert mem.shape == (B, N_MEM, D)
