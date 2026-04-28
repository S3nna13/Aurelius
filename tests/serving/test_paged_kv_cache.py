"""Smoke tests for the serving-side paged KV cache helper."""

from __future__ import annotations

import pytest
import torch

from src.serving.paged_kv_cache import PagedKVCache

N_LAYERS = 2
N_KV_HEADS = 2
HEAD_DIM = 4
BLOCK_SIZE = 4
MAX_BLOCKS = 8


def _make(**overrides) -> PagedKVCache:
    kwargs = dict(
        n_layers=N_LAYERS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=MAX_BLOCKS,
    )
    kwargs.update(overrides)
    return PagedKVCache(**kwargs)


def _rand_kv(seed: int):
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn(N_KV_HEADS, HEAD_DIM, generator=generator)
    v = torch.randn(N_KV_HEADS, HEAD_DIM, generator=generator)
    return k, v


def test_append_and_gather_round_trip():
    cache = _make()
    cache.init_sequence(1, 3)

    expected = []
    for pos in range(3):
        k, v = _rand_kv(seed=pos)
        cache.append_tokens(1, 0, k, v)
        expected.append((k, v))

    assert cache._get_seq_len(1) == 3
    out_k, out_v = cache.gather(1, 3, 0)

    assert out_k.shape == (1, N_KV_HEADS, 3, HEAD_DIM)
    assert out_v.shape == (1, N_KV_HEADS, 3, HEAD_DIM)
    for pos, (k, v) in enumerate(expected):
        assert torch.allclose(out_k[0, :, pos, :], k.to(dtype=out_k.dtype))
        assert torch.allclose(out_v[0, :, pos, :], v.to(dtype=out_v.dtype))


def test_prefix_share_triggers_copy_on_write():
    cache = _make()
    cache.init_sequence(1, 2)
    source_tokens = []
    for pos in range(2):
        k, v = _rand_kv(seed=pos + 10)
        cache.append_tokens(1, 0, k, v)
        source_tokens.append((k, v))

    cache.block_table.share_prefix(1, 2, 2)
    cache._seq_lengths[2] = 2

    shared_block = cache.block_table.get_physical_id(1, 0)
    assert shared_block == cache.block_table.get_physical_id(2, 0)

    new_k, new_v = _rand_kv(seed=999)
    cache.append_tokens(2, 0, new_k, new_v)

    assert cache.block_table.get_physical_id(2, 0) != shared_block
    src_k, src_v = cache.gather(1, 2, 0)
    tgt_k, tgt_v = cache.gather(2, 3, 0)

    assert torch.allclose(src_k[0, :, 0, :], source_tokens[0][0].to(dtype=src_k.dtype))
    assert torch.allclose(src_v[0, :, 0, :], source_tokens[0][1].to(dtype=src_v.dtype))
    assert torch.allclose(src_k[0, :, 1, :], source_tokens[1][0].to(dtype=src_k.dtype))
    assert torch.allclose(src_v[0, :, 1, :], source_tokens[1][1].to(dtype=src_v.dtype))
    assert torch.allclose(tgt_k[0, :, 0, :], source_tokens[0][0].to(dtype=tgt_k.dtype))
    assert torch.allclose(tgt_v[0, :, 0, :], source_tokens[0][1].to(dtype=tgt_v.dtype))
    assert torch.allclose(tgt_k[0, :, 1, :], source_tokens[1][0].to(dtype=tgt_k.dtype))
    assert torch.allclose(tgt_v[0, :, 1, :], source_tokens[1][1].to(dtype=tgt_v.dtype))
    assert torch.allclose(tgt_k[0, :, 2, :], new_k.to(dtype=tgt_k.dtype))
    assert torch.allclose(tgt_v[0, :, 2, :], new_v.to(dtype=tgt_v.dtype))


def test_free_sequence_returns_capacity():
    cache = _make()
    cache.init_sequence(1, 4)
    assert cache.utilization == pytest.approx(1 / MAX_BLOCKS)

    cache.free_sequence(1)

    assert cache.utilization == pytest.approx(0.0)
    assert cache.block_table.used_blocks == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_layers": 0},
        {"n_layers": 2, "n_kv_heads": 0},
        {"n_layers": 2, "n_kv_heads": 2, "head_dim": 0},
        {"n_layers": 2, "n_kv_heads": 2, "head_dim": 4, "block_size": 0},
        {"n_layers": 2, "n_kv_heads": 2, "head_dim": 4, "block_size": 4, "max_blocks": 0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        _make(**kwargs)
