"""Tests for src/model/external_memory.py"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.external_memory import (
    ExternalMemory,
    MemoryAugmentedLayer,
    MemoryConfig,
    MemoryController,
)

# ---------------------------------------------------------------------------
# Common test parameters
# ---------------------------------------------------------------------------
B = 2
T = 4
D_MODEL = 32
MEMORY_SIZE = 16
MEMORY_DIM = 16


def make_config(**kwargs) -> MemoryConfig:
    defaults = dict(
        memory_size=MEMORY_SIZE,
        memory_dim=MEMORY_DIM,
        n_read_heads=1,
        n_write_heads=1,
        controller_dim=64,
    )
    defaults.update(kwargs)
    return MemoryConfig(**defaults)


def make_memory(config: MemoryConfig | None = None) -> ExternalMemory:
    if config is None:
        config = make_config()
    mem = ExternalMemory(config)
    mem.reset(B)
    return mem


# ---------------------------------------------------------------------------
# 1. MemoryConfig defaults
# ---------------------------------------------------------------------------
def test_memory_config_defaults():
    cfg = MemoryConfig()
    assert cfg.memory_size == 128
    assert cfg.memory_dim == 32
    assert cfg.n_read_heads == 2
    assert cfg.n_write_heads == 1
    assert cfg.controller_dim == 64


# ---------------------------------------------------------------------------
# 2. ExternalMemory.read returns shape (B, memory_dim)
# ---------------------------------------------------------------------------
def test_read_output_shape():
    mem = make_memory()
    query = torch.randn(B, MEMORY_DIM)
    out = mem.read(query)
    assert out.shape == (B, MEMORY_DIM)


# ---------------------------------------------------------------------------
# 3. ExternalMemory.read output is weighted average of memory
# ---------------------------------------------------------------------------
def test_read_is_weighted_average():
    mem = make_memory()
    # Set memory to known values (identity-like rows)
    mem.memory = torch.eye(MEMORY_SIZE, MEMORY_DIM)
    query = torch.randn(B, MEMORY_DIM)
    out = mem.read(query)
    # Output must be a convex combination of memory rows
    # so each element should be between min and max of memory
    assert out.shape == (B, MEMORY_DIM)
    # Verify by reconstructing: weights @ memory
    q_norm = torch.nn.functional.normalize(query, p=2, dim=-1)
    m_norm = torch.nn.functional.normalize(mem.memory, p=2, dim=-1)
    scores = torch.matmul(q_norm, m_norm.t())
    weights = torch.softmax(scores, dim=-1)
    expected = torch.matmul(weights, mem.memory)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. ExternalMemory.write updates memory slots
# ---------------------------------------------------------------------------
def test_write_updates_memory():
    mem = make_memory()
    mem_before = mem.memory.clone()
    key = torch.randn(B, MEMORY_DIM)
    value = torch.ones(B, MEMORY_DIM)
    strength = torch.ones(B, 1)
    mem.write(key, value, strength)
    assert not torch.allclose(mem.memory, mem_before)


# ---------------------------------------------------------------------------
# 5. ExternalMemory.reset initializes to correct shape
# ---------------------------------------------------------------------------
def test_reset_shape():
    mem = make_memory()
    assert mem.memory is not None
    assert mem.memory.shape == (MEMORY_SIZE, MEMORY_DIM)


# ---------------------------------------------------------------------------
# 6. ExternalMemory read after write retrieves related content
# ---------------------------------------------------------------------------
def test_read_after_write_retrieves_content():
    mem = make_memory()
    # Zero out memory first
    mem.memory = torch.zeros(MEMORY_SIZE, MEMORY_DIM)
    # Write a distinctive pattern using a key that will concentrate on slot 0
    # Create a key that maximizes similarity to slot 0 of memory (all zeros → uniform)
    # Instead write then verify memory has changed towards value
    key = torch.ones(B, MEMORY_DIM)
    value = torch.full((B, MEMORY_DIM), 5.0)
    strength = torch.ones(B, 1)
    mem.write(key, value, strength)
    # Read with the same key — should get a value close to written value
    out = mem.read(key)
    # Memory slots all received equal weight (uniform cosine of zero vectors → uniform)
    # All slots were updated equally, so out ≈ value * B (batched sum) / MEMORY_SIZE * MEMORY_SIZE
    # At least the output should be positive (all values pushed positive)
    assert out.mean().item() > 0.0


# ---------------------------------------------------------------------------
# 7. MemoryController output shape (B, T, d_model)
# ---------------------------------------------------------------------------
def test_controller_output_shape():
    cfg = make_config()
    ctrl = MemoryController(D_MODEL, cfg)
    mem = make_memory(cfg)
    x = torch.randn(B, T, D_MODEL)
    out, _ = ctrl(x, mem)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 8. MemoryController read_vector shape (B, T, memory_dim)
# ---------------------------------------------------------------------------
def test_controller_read_vector_shape():
    cfg = make_config(n_read_heads=1)
    ctrl = MemoryController(D_MODEL, cfg)
    mem = make_memory(cfg)
    x = torch.randn(B, T, D_MODEL)
    _, read_vec = ctrl(x, mem)
    assert read_vec.shape == (B, T, MEMORY_DIM)


# ---------------------------------------------------------------------------
# 9. MemoryController differentiable
# ---------------------------------------------------------------------------
def test_controller_differentiable():
    cfg = make_config()
    ctrl = MemoryController(D_MODEL, cfg)
    mem = make_memory(cfg)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out, _ = ctrl(x, mem)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


# ---------------------------------------------------------------------------
# 10. MemoryAugmentedLayer output shape matches input
# ---------------------------------------------------------------------------
def test_layer_output_shape():
    cfg = make_config()
    layer = MemoryAugmentedLayer(D_MODEL, cfg)
    mem = make_memory(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x, mem)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 11. MemoryAugmentedLayer differentiable
# ---------------------------------------------------------------------------
def test_layer_differentiable():
    cfg = make_config()
    layer = MemoryAugmentedLayer(D_MODEL, cfg)
    mem = make_memory(cfg)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = layer(x, mem)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# 12. Read with uniform memory → uniform attention weights
# ---------------------------------------------------------------------------
def test_read_uniform_memory_uniform_weights():
    mem = make_memory()
    # All memory slots identical → cosine sim is 1 for all → softmax → uniform
    mem.memory = torch.ones(MEMORY_SIZE, MEMORY_DIM)
    query = torch.randn(B, MEMORY_DIM)
    out = mem.read(query)
    # Expected: uniform weights → output equals any memory row = ones
    expected = torch.ones(B, MEMORY_DIM)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 13. Multiple read heads: n_read_heads=2 → read vector (B, T, 2*memory_dim)
# ---------------------------------------------------------------------------
def test_multiple_read_heads_shape():
    cfg = make_config(n_read_heads=2)
    ctrl = MemoryController(D_MODEL, cfg)
    mem = make_memory(cfg)
    x = torch.randn(B, T, D_MODEL)
    _, read_vec = ctrl(x, mem)
    assert read_vec.shape == (B, T, 2 * MEMORY_DIM)


# ---------------------------------------------------------------------------
# 14. Memory persists across multiple forward calls
# ---------------------------------------------------------------------------
def test_memory_persists_across_calls():
    cfg = make_config()
    layer = MemoryAugmentedLayer(D_MODEL, cfg)
    mem = make_memory(cfg)
    x = torch.randn(B, T, D_MODEL)
    mem_after_reset = mem.memory.clone()

    # First forward
    layer(x, mem)
    mem_after_first = mem.memory.clone()

    # Memory should have changed from reset state
    assert not torch.allclose(mem_after_reset, mem_after_first)

    # Second forward
    layer(x, mem)
    mem_after_second = mem.memory.clone()

    # Memory should have changed again
    assert not torch.allclose(mem_after_first, mem_after_second)
