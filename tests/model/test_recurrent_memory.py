"""Tests for src/model/recurrent_memory.py."""

import torch
import torch.nn as nn
import pytest

from src.model.recurrent_memory import (
    RMTConfig,
    MemoryTokens,
    segment_sequence,
    RMTWrapper,
    compute_memory_utilization,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_small_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


def _make_base_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(_make_small_config())


def _make_rmt_config(**kwargs) -> RMTConfig:
    defaults = dict(n_memory_tokens=4, segment_size=16, memory_dim=64)
    defaults.update(kwargs)
    return RMTConfig(**defaults)


def _make_wrapper(rmt_config: RMTConfig | None = None) -> RMTWrapper:
    base = _make_base_model()
    cfg = rmt_config or _make_rmt_config()
    return RMTWrapper(base, cfg)


# ---------------------------------------------------------------------------
# 1. RMTConfig defaults
# ---------------------------------------------------------------------------

def test_rmt_config_defaults():
    cfg = RMTConfig()
    assert cfg.n_memory_tokens == 16
    assert cfg.segment_size == 512
    assert cfg.memory_layers == [0, -1]
    assert cfg.memory_dim == 64
    assert cfg.detach_memory is False


# ---------------------------------------------------------------------------
# 2. MemoryTokens output shape (B, n_tokens, d_model)
# ---------------------------------------------------------------------------

def test_memory_tokens_output_shape():
    n_tokens, d_model = 8, 64
    mem = MemoryTokens(n_tokens, d_model)
    B = 3
    out = mem(B)
    assert out.shape == (B, n_tokens, d_model)


# ---------------------------------------------------------------------------
# 3. MemoryTokens is nn.Parameter (trainable)
# ---------------------------------------------------------------------------

def test_memory_tokens_is_parameter():
    mem = MemoryTokens(8, 64)
    assert isinstance(mem.memory, nn.Parameter)
    assert mem.memory.requires_grad is True


# ---------------------------------------------------------------------------
# 4. MemoryTokens.update changes stored memory
# ---------------------------------------------------------------------------

def test_memory_tokens_update_changes_memory():
    n_tokens, d_model = 4, 64
    mem = MemoryTokens(n_tokens, d_model)
    original = mem.memory.data.clone()

    new_state = torch.ones(2, n_tokens, d_model)  # B=2 batch
    mem.update(new_state)

    # Memory should have changed
    assert not torch.allclose(mem.memory.data, original)
    # Should be mean over batch dimension, keepdim=True
    assert mem.memory.data.shape == (1, n_tokens, d_model)
    assert torch.allclose(mem.memory.data, new_state.mean(0, keepdim=True))


# ---------------------------------------------------------------------------
# 5. segment_sequence correct number of segments
# ---------------------------------------------------------------------------

def test_segment_sequence_num_segments():
    B, T, seg_size = 2, 32, 16
    ids = torch.randint(0, 100, (B, T))
    segs = segment_sequence(ids, seg_size)
    expected = (T + seg_size - 1) // seg_size  # ceil division
    assert len(segs) == expected


# ---------------------------------------------------------------------------
# 6. segment_sequence each segment correct size
# ---------------------------------------------------------------------------

def test_segment_sequence_segment_size():
    B, T, seg_size = 2, 48, 16
    ids = torch.randint(0, 100, (B, T))
    segs = segment_sequence(ids, seg_size)
    # All segments except possibly last should have exactly seg_size tokens
    for seg in segs[:-1]:
        assert seg.shape == (B, seg_size)


# ---------------------------------------------------------------------------
# 7. segment_sequence last segment handles remainder
# ---------------------------------------------------------------------------

def test_segment_sequence_last_segment_remainder():
    B, T, seg_size = 2, 50, 16
    ids = torch.randint(0, 100, (B, T))
    segs = segment_sequence(ids, seg_size)
    remainder = T % seg_size  # 50 % 16 = 2
    assert segs[-1].shape == (B, remainder)


# ---------------------------------------------------------------------------
# 8. RMTWrapper forward returns 3-tuple
# ---------------------------------------------------------------------------

def test_rmt_wrapper_forward_returns_3_tuple():
    torch.manual_seed(0)
    wrapper = _make_wrapper()
    ids = torch.randint(0, 256, (2, 16))
    result = wrapper(ids)
    assert isinstance(result, tuple)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 9. RMTWrapper logits shape (B, T, vocab_size)
# ---------------------------------------------------------------------------

def test_rmt_wrapper_logits_shape():
    torch.manual_seed(0)
    wrapper = _make_wrapper()
    B, T = 2, 16
    ids = torch.randint(0, 256, (B, T))
    loss, logits, pkv = wrapper(ids)
    assert logits.shape == (B, T, 256)


# ---------------------------------------------------------------------------
# 10. RMTWrapper works with single segment
# ---------------------------------------------------------------------------

def test_rmt_wrapper_single_segment():
    torch.manual_seed(0)
    cfg = _make_rmt_config(segment_size=32)
    wrapper = _make_wrapper(cfg)
    B, T = 2, 8  # T < segment_size → single segment
    ids = torch.randint(0, 256, (B, T))
    loss, logits, pkv = wrapper(ids)
    assert logits.shape == (B, T, 256)
    assert loss is None
    assert pkv == []


# ---------------------------------------------------------------------------
# 11. RMTWrapper works with multiple segments
# ---------------------------------------------------------------------------

def test_rmt_wrapper_multiple_segments():
    torch.manual_seed(0)
    cfg = _make_rmt_config(segment_size=8)
    wrapper = _make_wrapper(cfg)
    B, T = 2, 32  # 4 segments of size 8
    ids = torch.randint(0, 256, (B, T))
    loss, logits, pkv = wrapper(ids)
    assert logits.shape == (B, T, 256)


# ---------------------------------------------------------------------------
# 12. compute_memory_utilization returns float in [-1, 1]
# ---------------------------------------------------------------------------

def test_compute_memory_utilization_range():
    torch.manual_seed(0)
    a = torch.randn(4, 16, 64)
    b = torch.randn(4, 16, 64)
    result = compute_memory_utilization(a, b)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 13. compute_memory_utilization identical tensors → ~1.0
# ---------------------------------------------------------------------------

def test_compute_memory_utilization_identical():
    x = torch.randn(4, 16, 64)
    result = compute_memory_utilization(x, x)
    assert abs(result - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 14. RMTWrapper.process_segment returns correct shapes
# ---------------------------------------------------------------------------

def test_rmt_wrapper_process_segment_shapes():
    torch.manual_seed(0)
    n_mem = 4
    cfg = _make_rmt_config(n_memory_tokens=n_mem)
    wrapper = _make_wrapper(cfg)

    B, S = 2, 12
    seg_ids = torch.randint(0, 256, (B, S))
    memory = wrapper.memory_tokens(B)  # (B, n_mem, d_model)

    logits_seg, new_memory = wrapper.process_segment(seg_ids, memory)

    assert logits_seg.shape == (B, S, 256), f"Expected ({B}, {S}, 256), got {logits_seg.shape}"
    assert new_memory.shape == (B, n_mem, 64), f"Expected ({B}, {n_mem}, 64), got {new_memory.shape}"
