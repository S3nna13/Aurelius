"""Integration tests for KIVI INT4 KV cache quantization registry wiring."""

from __future__ import annotations

import torch

from src.longcontext import (
    LONGCONTEXT_STRATEGY_REGISTRY,
    CompressedKIVI,
    KIVIQuantizer,
    KVInt8Compressor,
)


def test_registry_contains_kivi_int4():
    assert "kv_kivi_int4" in LONGCONTEXT_STRATEGY_REGISTRY
    assert LONGCONTEXT_STRATEGY_REGISTRY["kv_kivi_int4"] is KIVIQuantizer


def test_registry_preserves_existing_entries():
    for key in ("kv_int8", "attention_sinks", "ring_attention", "context_compaction"):
        assert key in LONGCONTEXT_STRATEGY_REGISTRY, f"missing strategy: {key}"


def test_kivi_exposed_classes_importable():
    # Classes themselves must be usable (not just names).
    q = KIVIQuantizer(n_heads=2, head_dim=16, group_size=16)
    k = torch.randn(1, 2, 32, 16)
    v = torch.randn(1, 2, 32, 16)
    c = q.compress(k, v)
    assert isinstance(c, CompressedKIVI)


def test_int4_buffer_smaller_than_int8_buffer_same_tokens():
    """INT4 packed KV should be strictly smaller than INT8 packed KV."""
    B, H, S, D = 1, 2, 64, 16
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    kivi = KIVIQuantizer(n_heads=H, head_dim=D, group_size=16)
    c_int4 = kivi.compress(k, v)

    int8c = KVInt8Compressor(n_heads=H, head_dim=D)
    c_int8 = int8c.compress(k, v)

    int4_bytes = (
        c_int4.k_q.numel() * c_int4.k_q.element_size()
        + c_int4.v_q.numel() * c_int4.v_q.element_size()
    )
    int8_bytes = (
        c_int8.k_q.numel() * c_int8.k_q.element_size()
        + c_int8.v_q.numel() * c_int8.v_q.element_size()
    )
    assert int4_bytes < int8_bytes, (
        f"INT4 payload ({int4_bytes}B) should be smaller than INT8 ({int8_bytes}B)"
    )
    # Expect roughly half for the quantized payload.
    assert int4_bytes <= int8_bytes // 2 + 1
