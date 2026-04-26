"""Integration tests for the longcontext strategy registry."""

from __future__ import annotations

import sys

import torch


def test_registry_entry_importable_and_usable():
    from src.longcontext import LONGCONTEXT_STRATEGY_REGISTRY

    cls = LONGCONTEXT_STRATEGY_REGISTRY["kv_int8"]
    comp = cls(head_dim=16, n_heads=4)

    torch.manual_seed(0)
    k = torch.randn(1, 4, 4, 16, dtype=torch.float32)
    v = torch.randn(1, 4, 4, 16, dtype=torch.float32)

    packed = comp.compress(k, v)
    kd, vd = comp.decompress(packed)

    assert kd.shape == k.shape
    assert vd.shape == v.shape
    assert torch.allclose(kd, k, atol=0.05)
    assert packed.k_q.dtype == torch.int8


def test_int8_buffer_smaller_than_fp32_original():
    from src.longcontext import KVInt8Compressor

    comp = KVInt8Compressor(head_dim=16, n_heads=4)
    k = torch.randn(2, 4, 64, 16, dtype=torch.float32)
    v = torch.randn(2, 4, 64, 16, dtype=torch.float32)
    packed = comp.compress(k, v)

    fp32_bytes = k.numel() * 4 + v.numel() * 4
    assert packed.nbytes() < fp32_bytes
    # Expect roughly 4x compression (int8 vs fp32) minus tiny scale overhead.
    ratio = fp32_bytes / packed.nbytes()
    assert ratio > 3.5


def test_import_has_no_side_effects_on_existing_modules():
    # Guard against the longcontext package accidentally importing (and
    # therefore mutating) sibling modules like model / training / etc.
    pre_mods = set(sys.modules.keys())
    import src.longcontext  # noqa: F401

    post_mods = set(sys.modules.keys())

    newly_loaded = post_mods - pre_mods
    forbidden_prefixes = (
        "src.model",
        "src.training",
        "src.inference",
        "src.serving",
        "src.alignment",
        "src.optimizers",
        "src.security",
        "src.eval",
        "src.data",
        "src.interpretability",
    )
    leaked = [m for m in newly_loaded if m.startswith(forbidden_prefixes)]
    assert not leaked, f"longcontext import leaked sibling modules: {leaked}"


def test_streaming_append_roundtrip():
    from src.longcontext import KVInt8Compressor

    comp = KVInt8Compressor(head_dim=8, n_heads=2)
    torch.manual_seed(42)
    k_full = torch.randn(1, 2, 10, 8, dtype=torch.float32)
    v_full = torch.randn(1, 2, 10, 8, dtype=torch.float32)

    packed = comp.compress(k_full[:, :, :3, :], v_full[:, :, :3, :])
    packed = comp.append(packed, k_full[:, :, 3:7, :], v_full[:, :, 3:7, :])
    packed = comp.append(packed, k_full[:, :, 7:, :], v_full[:, :, 7:, :])

    assert packed.seq_len == 10
    kd, vd = comp.decompress(packed)
    assert torch.allclose(kd, k_full, atol=0.08)
    assert torch.allclose(vd, v_full, atol=0.08)
