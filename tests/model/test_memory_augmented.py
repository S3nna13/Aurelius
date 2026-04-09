"""Tests for memory_augmented.py — Memorizing Transformers module."""

from __future__ import annotations

import torch
import pytest

from src.model.memory_augmented import (
    MemoryConfig,
    ExternalMemoryBank,
    MemoryAttentionLayer,
    MemorizingTransformerBlock,
    MemoryUtilizationTracker,
    build_memory_augmented_model,
)

# ---------------------------------------------------------------------------
# Common fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 2
D_FF = 128
MEMORY_SIZE = 32
KEY_DIM = 16
VALUE_DIM = 32
TOP_K = 8
BATCH = 2
SEQ_LEN = 8

torch.manual_seed(0)


def make_config(**kwargs) -> MemoryConfig:
    defaults = dict(
        memory_size=MEMORY_SIZE,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        n_memory_heads=2,
        top_k=TOP_K,
    )
    defaults.update(kwargs)
    return MemoryConfig(**defaults)


def make_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_memory_config_defaults():
    """MemoryConfig dataclass should expose the correct defaults."""
    cfg = MemoryConfig()
    assert cfg.memory_size == 512
    assert cfg.key_dim == 32
    assert cfg.value_dim == 64
    assert cfg.n_memory_heads == 2
    assert cfg.top_k == 32
    assert cfg.update_strategy == "fifo"


def test_memory_bank_read_shape():
    """ExternalMemoryBank.read should return retrieved values of shape (B, n_q, value_dim)."""
    torch.manual_seed(0)
    cfg = make_config()
    bank = ExternalMemoryBank(cfg)
    queries = torch.randn(BATCH, SEQ_LEN, KEY_DIM)
    retrieved, weights = bank.read(queries)
    assert retrieved.shape == (BATCH, SEQ_LEN, VALUE_DIM), f"Got {retrieved.shape}"
    assert weights.shape == (BATCH, SEQ_LEN, MEMORY_SIZE), f"Got {weights.shape}"


def test_memory_bank_read_weights_sum():
    """Attention weights from top-k softmax should sum to ~1 per query position."""
    torch.manual_seed(0)
    cfg = make_config()
    bank = ExternalMemoryBank(cfg)
    queries = torch.randn(BATCH, SEQ_LEN, KEY_DIM)
    _, weights = bank.read(queries)
    weight_sums = weights.sum(dim=-1)  # (B, n_q)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
        f"Weights do not sum to 1; got min={weight_sums.min():.4f} max={weight_sums.max():.4f}"
    )


def test_memory_bank_write_fifo():
    """FIFO writes should advance _write_ptr by the number of entries written."""
    cfg = make_config(update_strategy="fifo")
    bank = ExternalMemoryBank(cfg)
    assert bank._write_ptr == 0
    n_write = 5
    bank.write(torch.randn(n_write, KEY_DIM), torch.randn(n_write, VALUE_DIM))
    assert bank._write_ptr == n_write, f"Expected {n_write}, got {bank._write_ptr}"
    # Second write
    bank.write(torch.randn(3, KEY_DIM), torch.randn(3, VALUE_DIM))
    assert bank._write_ptr == n_write + 3


def test_memory_bank_reset():
    """reset() should zero all keys, values, and access counts."""
    cfg = make_config()
    bank = ExternalMemoryBank(cfg)
    # Write something non-zero
    bank.write(torch.ones(4, KEY_DIM), torch.ones(4, VALUE_DIM))
    bank.reset()
    assert bank.keys.abs().sum().item() == 0.0, "keys not zeroed"
    assert bank.values.abs().sum().item() == 0.0, "values not zeroed"
    assert bank._access_count.abs().sum().item() == 0.0, "access_count not zeroed"
    assert bank._write_ptr == 0


def test_memory_attention_layer_output_shape():
    """MemoryAttentionLayer.forward should return output of shape (B, T, D)."""
    torch.manual_seed(0)
    cfg = make_config()
    layer = MemoryAttentionLayer(D_MODEL, cfg)
    x = make_input()
    out, weights = layer(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Got {out.shape}"
    assert weights.shape == (BATCH, SEQ_LEN, MEMORY_SIZE), f"Got {weights.shape}"


def test_memory_attention_gate_range():
    """The learnable gate, after sigmoid, should be in (0, 1)."""
    cfg = make_config()
    layer = MemoryAttentionLayer(D_MODEL, cfg)
    g = torch.sigmoid(layer.gate)
    assert (g > 0).all() and (g < 1).all(), f"Gate out of range: {g.item():.4f}"


def test_memorizing_block_output_shape():
    """MemorizingTransformerBlock.forward should return (B, T, D)."""
    torch.manual_seed(0)
    cfg = make_config()
    block = MemorizingTransformerBlock(D_MODEL, N_HEADS, D_FF, cfg)
    x = make_input()
    out = block(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Got {out.shape}"


def test_memorizing_block_gradient_flow():
    """Backward pass through MemorizingTransformerBlock should not raise and produce gradients."""
    torch.manual_seed(0)
    cfg = make_config()
    block = MemorizingTransformerBlock(D_MODEL, N_HEADS, D_FF, cfg)
    x = make_input().requires_grad_(True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    # Check at least some model parameters received gradients
    has_param_grad = any(p.grad is not None for p in block.parameters() if p.requires_grad)
    assert has_param_grad, "No parameter gradients"


def test_build_memory_model_count():
    """build_memory_augmented_model should return exactly n_layers blocks."""
    cfg = make_config()
    n_layers = 4
    model = build_memory_augmented_model(D_MODEL, n_layers, N_HEADS, D_FF, cfg)
    assert len(model) == n_layers, f"Expected {n_layers} layers, got {len(model)}"
    for block in model:
        assert isinstance(block, MemorizingTransformerBlock)


def test_memory_tracker_stats_keys():
    """MemoryUtilizationTracker.utilization_stats should return the required keys."""
    cfg = make_config()
    bank = ExternalMemoryBank(cfg)
    tracker = MemoryUtilizationTracker(bank)
    # Record a few steps with dummy weights
    for _ in range(3):
        dummy_weights = torch.zeros(BATCH, SEQ_LEN, MEMORY_SIZE)
        dummy_weights[:, :, :TOP_K] = 1.0 / TOP_K  # uniform over first TOP_K slots
        tracker.record_step(dummy_weights)
    stats = tracker.utilization_stats()
    assert "mean_utilized_slots" in stats
    assert "coverage" in stats
    assert "entropy" in stats
    # Values should be non-negative
    for k, v in stats.items():
        assert v >= 0.0, f"{k}={v} is negative"


def test_memory_write_read_cycle():
    """Writing known values into memory and reading them back should return similar values."""
    torch.manual_seed(0)
    cfg = make_config(memory_size=MEMORY_SIZE, top_k=TOP_K)
    bank = ExternalMemoryBank(cfg)

    # Zero out all memory first
    bank.reset()

    # Write a distinctive pattern into the first TOP_K slots
    known_keys = torch.zeros(TOP_K, KEY_DIM)
    known_values = torch.ones(TOP_K, VALUE_DIM)  # all ones
    bank.write(known_keys, known_values)

    # Query with the same known key pattern — should retrieve ~1.0 values
    queries = torch.zeros(1, 1, KEY_DIM)  # (B=1, n_q=1, key_dim)
    retrieved, weights = bank.read(queries)

    # Retrieved values should be close to 1 (the known pattern)
    # At least the mean should be significantly > 0
    mean_val = retrieved.mean().item()
    assert mean_val > 0.5, (
        f"Expected retrieved values close to 1.0, got mean={mean_val:.4f}. "
        "Memory write-read cycle may be broken."
    )
