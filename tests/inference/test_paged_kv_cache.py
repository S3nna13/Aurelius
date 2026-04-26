"""Tests for the PagedAttention-style KV cache manager."""

import pytest
import torch

from src.inference.paged_kv_cache import (
    PAGE_SIZE,
    PagedKVCacheManager,
    PagePool,
    SequenceKVCache,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

N_KV_HEADS = 4
HEAD_DIM = 8
PAGE_SZ = PAGE_SIZE  # 16 tokens per page


def make_pool(n_pages: int = 32) -> PagePool:
    return PagePool(
        n_pages=n_pages,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SZ,
        device="cpu",
    )


def rand_kv() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random (key, value) pair of shape (N_KV_HEADS, HEAD_DIM)."""
    k = torch.randn(N_KV_HEADS, HEAD_DIM)
    v = torch.randn(N_KV_HEADS, HEAD_DIM)
    return k, v


# ---------------------------------------------------------------------------
# 1. test_page_pool_allocate
# ---------------------------------------------------------------------------


def test_page_pool_allocate():
    pool = make_pool(n_pages=8)
    initial_free = pool.available_pages()
    assert initial_free == 8

    pid = pool.allocate_page()
    assert 0 <= pid < 8
    assert pool.available_pages() == 7


# ---------------------------------------------------------------------------
# 2. test_page_pool_exhaustion
# ---------------------------------------------------------------------------


def test_page_pool_exhaustion():
    pool = make_pool(n_pages=3)
    for _ in range(3):
        pool.allocate_page()

    with pytest.raises(RuntimeError, match="exhausted"):
        pool.allocate_page()


# ---------------------------------------------------------------------------
# 3. test_page_pool_free_recycles
# ---------------------------------------------------------------------------


def test_page_pool_free_recycles():
    pool = make_pool(n_pages=2)
    pid0 = pool.allocate_page()
    pool.allocate_page()
    assert pool.available_pages() == 0

    pool.free_page(pid0)
    assert pool.available_pages() == 1

    recycled = pool.allocate_page()
    assert recycled == pid0  # FIFO deque returns it first
    assert pool.available_pages() == 0


# ---------------------------------------------------------------------------
# 4. test_write_and_read
# ---------------------------------------------------------------------------


def test_write_and_read():
    pool = make_pool()
    pid = pool.allocate_page()
    k, v = rand_kv()

    pool.write_token(pid, slot=0, key=k, value=v)
    k_page, v_page = pool.read_page(pid)

    assert torch.allclose(k_page[0], k)
    assert torch.allclose(v_page[0], v)


# ---------------------------------------------------------------------------
# 5. test_sequence_append_token
# ---------------------------------------------------------------------------


def test_sequence_append_token():
    pool = make_pool(n_pages=64)
    seq = SequenceKVCache(pool=pool, n_layers=2)

    n = 20
    for _ in range(n):
        k, v = rand_kv()
        seq.append_token(layer=0, key=k, value=v)

    assert seq.n_tokens == n


# ---------------------------------------------------------------------------
# 6. test_sequence_gather_kv_shape
# ---------------------------------------------------------------------------


def test_sequence_gather_kv_shape():
    pool = make_pool(n_pages=64)
    seq = SequenceKVCache(pool=pool, n_layers=2)

    n = 20
    for _ in range(n):
        k, v = rand_kv()
        seq.append_token(layer=0, key=k, value=v)

    keys, values = seq.gather_kv(layer=0)
    assert keys.shape == (n, N_KV_HEADS, HEAD_DIM)
    assert values.shape == (n, N_KV_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# 7. test_sequence_gather_kv_values
# ---------------------------------------------------------------------------


def test_sequence_gather_kv_values():
    pool = make_pool(n_pages=64)
    seq = SequenceKVCache(pool=pool, n_layers=2)

    n = 20
    stored_k = []
    stored_v = []
    for _ in range(n):
        k, v = rand_kv()
        stored_k.append(k)
        stored_v.append(v)
        seq.append_token(layer=0, key=k, value=v)

    keys, values = seq.gather_kv(layer=0)

    expected_k = torch.stack(stored_k)  # (n, N_KV_HEADS, HEAD_DIM)
    expected_v = torch.stack(stored_v)

    assert torch.allclose(keys, expected_k, atol=1e-6)
    assert torch.allclose(values, expected_v, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. test_sequence_free_recycles_pages
# ---------------------------------------------------------------------------


def test_sequence_free_recycles_pages():
    pool = make_pool(n_pages=64)
    initial_free = pool.available_pages()

    seq = SequenceKVCache(pool=pool, n_layers=2)

    # Append to both layers so pages are consumed from both
    for _ in range(20):
        k, v = rand_kv()
        seq.append_token(layer=0, key=k, value=v)
        seq.append_token(layer=1, key=k, value=v)

    assert pool.available_pages() < initial_free

    seq.free()
    assert pool.available_pages() == initial_free


# ---------------------------------------------------------------------------
# 9. test_manager_create_and_free
# ---------------------------------------------------------------------------


def test_manager_create_and_free():
    manager = PagedKVCacheManager(
        n_pages=128,
        n_layers=2,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SZ,
    )
    initial_free = manager.available_pages()

    seqs = [manager.create_sequence(i) for i in range(3)]

    # Append tokens so pages are actually allocated
    for seq in seqs:
        for _ in range(20):
            k, v = rand_kv()
            seq.append_token(layer=0, key=k, value=v)
            seq.append_token(layer=1, key=k, value=v)

    assert manager.available_pages() < initial_free

    for i in range(3):
        manager.free_sequence(i)

    assert manager.available_pages() == initial_free


# ---------------------------------------------------------------------------
# 10. test_multi_layer_independent
# ---------------------------------------------------------------------------


def test_multi_layer_independent():
    pool = make_pool(n_pages=128)
    seq = SequenceKVCache(pool=pool, n_layers=4)

    stored_k0, stored_v0 = [], []
    stored_k1, stored_v1 = [], []

    n = 25
    for _ in range(n):
        k0, v0 = rand_kv()
        k1, v1 = rand_kv()
        seq.append_token(layer=0, key=k0, value=v0)
        seq.append_token(layer=1, key=k1, value=v1)
        stored_k0.append(k0)
        stored_v0.append(v0)
        stored_k1.append(k1)
        stored_v1.append(v1)

    keys0, values0 = seq.gather_kv(layer=0)
    keys1, values1 = seq.gather_kv(layer=1)

    assert keys0.shape == (n, N_KV_HEADS, HEAD_DIM)
    assert keys1.shape == (n, N_KV_HEADS, HEAD_DIM)

    assert torch.allclose(keys0, torch.stack(stored_k0), atol=1e-6)
    assert torch.allclose(values0, torch.stack(stored_v0), atol=1e-6)
    assert torch.allclose(keys1, torch.stack(stored_k1), atol=1e-6)
    assert torch.allclose(values1, torch.stack(stored_v1), atol=1e-6)

    # Layers must store independent data — layer 0 keys != layer 1 keys
    assert not torch.allclose(keys0, keys1)
