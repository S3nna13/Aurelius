"""Tests for titans.py — Titans Neural Memory module."""

from __future__ import annotations

import torch

from src.model.titans import (
    NeuralMemory,
    TitansConfig,
    TitansLayer,
    build_titans_layer,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

torch.manual_seed(42)

D_MODEL = 64
BATCH = 2
SEQ = 8

TINY_CFG = TitansConfig(
    d_model=D_MODEL,
    n_heads=4,
    memory_dim=32,
    n_persistent=8,
)


def make_memory() -> NeuralMemory:
    return NeuralMemory(d_model=D_MODEL, memory_dim=32)


def make_layer() -> TitansLayer:
    return build_titans_layer(TINY_CFG)


def make_xy(seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(BATCH, seq, D_MODEL)
    y = torch.randn(BATCH, seq, D_MODEL)
    return x, y


# ---------------------------------------------------------------------------
# Test 1 — NeuralMemory instantiates correctly
# ---------------------------------------------------------------------------


def test_neural_memory_instantiation():
    mem = make_memory()
    assert isinstance(mem, NeuralMemory)
    assert mem.d_model == D_MODEL
    assert mem.memory_dim == 32
    # MLP should have two linear layers
    linears = [m for m in mem.memory_mlp.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2


# ---------------------------------------------------------------------------
# Test 2 — write() returns surprise tensor with correct shape (batch, seq)
# ---------------------------------------------------------------------------


def test_write_surprise_shape():
    mem = make_memory()
    x, y = make_xy()
    surprise = mem.write(x, y)
    assert surprise.shape == (BATCH, SEQ), f"Expected ({BATCH},{SEQ}), got {surprise.shape}"
    assert surprise.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 3 — read() returns tensor with correct shape (batch, seq, d_model)
# ---------------------------------------------------------------------------


def test_read_shape():
    mem = make_memory()
    x, _ = make_xy()
    out = mem.read(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH},{SEQ},{D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Memory writes update internal MLP weights
# ---------------------------------------------------------------------------


def test_write_updates_weights():
    mem = make_memory()
    # Capture parameter snapshot before write
    before = {n: p.clone() for n, p in mem.memory_mlp.named_parameters()}
    x, y = make_xy()
    mem.write(x, y)
    after = {n: p.clone() for n, p in mem.memory_mlp.named_parameters()}
    changed = any(not torch.equal(before[n], after[n]) for n in before)
    assert changed, "No MLP parameter changed after write()"


# ---------------------------------------------------------------------------
# Test 5 — After write then read, output is not all zeros
# ---------------------------------------------------------------------------


def test_write_then_read_nonzero():
    mem = make_memory()
    x, y = make_xy()
    mem.write(x, y)
    out = mem.read(x)
    assert not torch.all(out == 0), "read() returned all zeros after write"
    assert torch.isfinite(out).all(), "read() contains NaN/Inf after write"


# ---------------------------------------------------------------------------
# Test 6 — reset() restores original weights
# ---------------------------------------------------------------------------


def test_reset_restores_weights():
    mem = make_memory()
    x, y = make_xy()

    # Capture read output before any write
    with torch.no_grad():
        out_before = mem.read(x).clone()

    # Write to alter weights
    for _ in range(5):
        mem.write(x, y)

    with torch.no_grad():
        out_after_write = mem.read(x).clone()

    # Weights should have changed
    assert not torch.allclose(out_before, out_after_write, atol=1e-6)

    # Reset and verify read output matches original
    mem.reset()
    with torch.no_grad():
        out_after_reset = mem.read(x).clone()

    assert torch.allclose(out_before, out_after_reset, atol=1e-6), (
        "After reset(), read output does not match pre-write output"
    )


# ---------------------------------------------------------------------------
# Test 7 — TitansLayer forward pass produces correct output shape
# ---------------------------------------------------------------------------


def test_titans_layer_output_shape():
    layer = make_layer()
    x, _ = make_xy()
    out = layer(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH},{SEQ},{D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 8 — TitansLayer forward is differentiable (backward() works)
# ---------------------------------------------------------------------------


def test_titans_layer_backward():
    layer = make_layer()
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()  # should not raise
    assert x.grad is not None, "No gradient flowed to input"
    assert torch.isfinite(x.grad).all(), "Gradient contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 9 — Persistent memory contributes to output (compare with/without)
# ---------------------------------------------------------------------------


def test_persistent_memory_contributes():
    torch.manual_seed(7)
    cfg_with = TitansConfig(
        d_model=D_MODEL, n_heads=4, memory_dim=32, n_persistent=8, use_persistent_memory=True
    )
    cfg_without = TitansConfig(
        d_model=D_MODEL, n_heads=4, memory_dim=32, n_persistent=8, use_persistent_memory=False
    )

    layer_with = build_titans_layer(cfg_with)
    layer_without = build_titans_layer(cfg_without)

    # Copy shared weights so the only difference is persistent memory
    layer_without.query_proj.load_state_dict(layer_with.query_proj.state_dict())
    layer_without.out_gate.load_state_dict(layer_with.out_gate.state_dict())
    layer_without.layer_norm.load_state_dict(layer_with.layer_norm.state_dict())
    layer_without.neural_memory.memory_mlp.load_state_dict(
        layer_with.neural_memory.memory_mlp.state_dict()
    )
    # Copy attention weights
    layer_without.attn.load_state_dict(layer_with.attn.state_dict())

    x, _ = make_xy()
    with torch.no_grad():
        out_with = layer_with(x)
        out_without = layer_without(x)

    assert not torch.allclose(out_with, out_without, atol=1e-5), (
        "Persistent memory made no difference to output"
    )


# ---------------------------------------------------------------------------
# Test 10 — build_titans_layer creates TitansLayer from config
# ---------------------------------------------------------------------------


def test_build_titans_layer_from_config():
    cfg = TitansConfig(d_model=32, n_heads=2, memory_dim=16, n_persistent=4)
    layer = build_titans_layer(cfg)
    assert isinstance(layer, TitansLayer)
    assert layer.d_model == 32
    x = torch.randn(1, 6, 32)
    out = layer(x)
    assert out.shape == (1, 6, 32)


# ---------------------------------------------------------------------------
# Test 11 — Multiple write steps reduce surprise (memory improves over time)
# ---------------------------------------------------------------------------


def test_multiple_writes_reduce_surprise():
    mem = NeuralMemory(d_model=D_MODEL, memory_dim=32, lr=0.05)
    torch.manual_seed(1)
    x = torch.randn(1, 4, D_MODEL)
    y = torch.randn(1, 4, D_MODEL)

    surprises = []
    for _ in range(20):
        s = mem.write(x, y)
        surprises.append(s.mean().item())

    # Average surprise over last 5 steps should be less than first 5
    first_avg = sum(surprises[:5]) / 5
    last_avg = sum(surprises[-5:]) / 5
    assert last_avg < first_avg, (
        f"Surprise did not decrease: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 12 — TitansLayer handles different sequence lengths without error
# ---------------------------------------------------------------------------


def test_titans_layer_variable_seq_len():
    layer = make_layer()
    for seq_len in [1, 4, 16, 32]:
        x = torch.randn(BATCH, seq_len, D_MODEL)
        out = layer(x)
        assert out.shape == (BATCH, seq_len, D_MODEL), (
            f"seq_len={seq_len}: expected ({BATCH},{seq_len},{D_MODEL}), got {out.shape}"
        )
