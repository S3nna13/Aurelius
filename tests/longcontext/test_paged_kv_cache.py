"""Unit tests for PagedKVCache."""

from __future__ import annotations

import pytest
import torch

from src.longcontext.paged_kv_cache import (
    PagedKVCache,
    PagedKVOutOfMemory,
    PageTable,
)

N_HEADS = 2
HEAD_DIM = 8
PAGE_SIZE = 4
NUM_PAGES = 8


def _make(**overrides) -> PagedKVCache:
    kwargs = dict(
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        num_pages=NUM_PAGES,
    )
    kwargs.update(overrides)
    return PagedKVCache(**kwargs)


def _rand_kv(n_heads: int = N_HEADS, head_dim: int = HEAD_DIM, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n_heads, head_dim, generator=g)
    v = torch.randn(n_heads, head_dim, generator=g)
    return k, v


def test_allocate_returns_correct_num_pages():
    cache = _make()
    table = cache.allocate("r0", 10)  # page_size=4 -> ceil(10/4)=3 pages
    assert isinstance(table, PageTable)
    assert table.request_id == "r0"
    assert len(table.logical_pages) == 3
    # Pages should be distinct.
    assert len(set(table.logical_pages)) == 3


def test_write_read_round_trip():
    cache = _make()
    cache.allocate("r0", 4)
    ks, vs = [], []
    for pos in range(4):
        k, v = _rand_kv(seed=pos)
        cache.write("r0", pos, k, v)
        ks.append(k)
        vs.append(v)
    out_k, out_v = cache.read("r0", 0, 4)
    assert out_k.shape == (4, N_HEADS, HEAD_DIM)
    for i in range(4):
        assert torch.allclose(out_k[i], ks[i])
        assert torch.allclose(out_v[i], vs[i])


def test_deallocate_returns_pages():
    cache = _make()
    initial = cache.num_free_pages()
    cache.allocate("r0", 10)  # 3 pages
    assert cache.num_free_pages() == initial - 3
    cache.deallocate("r0")
    assert cache.num_free_pages() == initial


def test_out_of_memory_raises():
    cache = _make()  # 8 pages, page_size=4 -> 32 tokens capacity
    with pytest.raises(PagedKVOutOfMemory):
        cache.allocate("r0", 4 * 8 + 1)  # 9 pages needed


def test_read_across_page_boundary():
    cache = _make()  # page_size=4
    cache.allocate("r0", 8)  # spans 2 pages
    tokens = []
    for pos in range(8):
        k, v = _rand_kv(seed=pos + 100)
        cache.write("r0", pos, k, v)
        tokens.append((k, v))
    out_k, out_v = cache.read("r0", 2, 6)  # crosses page boundary at 4
    assert out_k.shape == (4, N_HEADS, HEAD_DIM)
    for i, pos in enumerate(range(2, 6)):
        assert torch.allclose(out_k[i], tokens[pos][0])
        assert torch.allclose(out_v[i], tokens[pos][1])


def test_read_beyond_allocated_length_raises():
    cache = _make()
    cache.allocate("r0", 4)
    k, v = _rand_kv()
    cache.write("r0", 0, k, v)
    with pytest.raises(IndexError):
        cache.read("r0", 0, 4)  # only wrote 1 token


def test_two_requests_independent():
    cache = _make()
    cache.allocate("a", 4)
    cache.allocate("b", 4)
    a_kvs, b_kvs = [], []
    # Interleaved writes.
    for pos in range(4):
        ka, va = _rand_kv(seed=pos)
        kb, vb = _rand_kv(seed=pos + 500)
        cache.write("a", pos, ka, va)
        cache.write("b", pos, kb, vb)
        a_kvs.append((ka, va))
        b_kvs.append((kb, vb))
    a_k, a_v = cache.read("a", 0, 4)
    b_k, b_v = cache.read("b", 0, 4)
    for i in range(4):
        assert torch.allclose(a_k[i], a_kvs[i][0])
        assert torch.allclose(a_v[i], a_kvs[i][1])
        assert torch.allclose(b_k[i], b_kvs[i][0])
        assert torch.allclose(b_v[i], b_kvs[i][1])


def test_prefix_share_target_sees_source_tokens():
    cache = _make()
    cache.allocate("src", 8)
    tokens = []
    for pos in range(8):
        k, v = _rand_kv(seed=pos + 7)
        cache.write("src", pos, k, v)
        tokens.append((k, v))
    tgt_table = cache.prefix_share("src", "tgt", n_shared_tokens=8)
    assert tgt_table.request_id == "tgt"
    # Shared pages should match.
    src_table = cache._tables["src"]
    assert tgt_table.logical_pages == src_table.logical_pages
    out_k, out_v = cache.read("tgt", 0, 8)
    for i in range(8):
        assert torch.allclose(out_k[i], tokens[i][0])
        assert torch.allclose(out_v[i], tokens[i][1])


def test_prefix_share_then_target_extends_allocates_new_pages():
    cache = _make()
    cache.allocate("src", 8)  # 2 pages
    for pos in range(8):
        k, v = _rand_kv(seed=pos)
        cache.write("src", pos, k, v)
    free_before = cache.num_free_pages()
    cache.prefix_share("src", "tgt", n_shared_tokens=8)
    # Sharing must not consume free pages.
    assert cache.num_free_pages() == free_before
    # Target extends by 4 more tokens -> 1 new page.
    cache.extend("tgt", 4)
    assert cache.num_free_pages() == free_before - 1
    # Writes past shared region land in new page (not source).
    src_table = cache._tables["src"]
    for pos in range(8, 12):
        k, v = _rand_kv(seed=pos + 1000)
        cache.write("tgt", pos, k, v)
    tgt_table = cache._tables["tgt"]
    # First 2 logical pages still shared.
    assert tgt_table.logical_pages[:2] == src_table.logical_pages[:2]
    # New page id differs from anything in source.
    assert tgt_table.logical_pages[2] not in src_table.logical_pages
    # Source's length/contents unchanged where we only wrote to target.
    src_k, _ = cache.read("src", 0, 8)
    assert src_k.shape == (8, N_HEADS, HEAD_DIM)


def test_num_free_pages_decreases_and_increases():
    cache = _make()
    start = cache.num_free_pages()
    assert start == NUM_PAGES
    cache.allocate("r0", 8)  # 2 pages
    assert cache.num_free_pages() == start - 2
    cache.allocate("r1", 4)  # 1 page
    assert cache.num_free_pages() == start - 3
    cache.deallocate("r0")
    assert cache.num_free_pages() == start - 1
    cache.deallocate("r1")
    assert cache.num_free_pages() == start


def test_page_size_one():
    cache = _make(page_size=1, num_pages=16)
    cache.allocate("r0", 5)
    tokens = []
    for pos in range(5):
        k, v = _rand_kv(seed=pos + 3)
        cache.write("r0", pos, k, v)
        tokens.append((k, v))
    out_k, out_v = cache.read("r0", 0, 5)
    for i in range(5):
        assert torch.allclose(out_k[i], tokens[i][0])
        assert torch.allclose(out_v[i], tokens[i][1])


def test_determinism():
    def run():
        cache = _make()
        cache.allocate("r0", 8)
        cache.allocate("r1", 4)
        for pos in range(8):
            k, v = _rand_kv(seed=pos)
            cache.write("r0", pos, k, v)
        for pos in range(4):
            k, v = _rand_kv(seed=pos + 50)
            cache.write("r1", pos, k, v)
        return (
            list(cache._tables["r0"].logical_pages),
            list(cache._tables["r1"].logical_pages),
            cache.read("r0", 0, 8),
            cache.read("r1", 0, 4),
        )

    a0, a1, (ak0, av0), (ak1, av1) = run()
    b0, b1, (bk0, bv0), (bk1, bv1) = run()
    assert a0 == b0
    assert a1 == b1
    assert torch.equal(ak0, bk0)
    assert torch.equal(av0, bv0)
    assert torch.equal(ak1, bk1)
    assert torch.equal(av1, bv1)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        PagedKVCache(n_heads=2, head_dim=8, page_size=0, num_pages=4)
    with pytest.raises(ValueError):
        PagedKVCache(n_heads=2, head_dim=8, page_size=-1, num_pages=4)
    with pytest.raises(ValueError):
        PagedKVCache(n_heads=2, head_dim=8, page_size=4, num_pages=0)
    with pytest.raises(ValueError):
        PagedKVCache(n_heads=2, head_dim=8, page_size=4, num_pages=-3)
    with pytest.raises(ValueError):
        PagedKVCache(n_heads=0, head_dim=8, page_size=4, num_pages=4)
    with pytest.raises(ValueError):
        PagedKVCache(n_heads=2, head_dim=0, page_size=4, num_pages=4)


def test_prefix_share_cow_on_write_into_shared_page():
    cache = _make(page_size=4, num_pages=8)
    cache.allocate("src", 8)
    for pos in range(8):
        k, v = _rand_kv(seed=pos + 11)
        cache.write("src", pos, k, v)
    # Share all 8 tokens (2 pages).
    cache.prefix_share("src", "tgt", n_shared_tokens=8)
    src_pages_before = list(cache._tables["src"].logical_pages)
    # Target overwrites token 0 (inside shared page 0) -> COW.
    new_k, new_v = _rand_kv(seed=999)
    cache.write("tgt", 0, new_k, new_v)
    tgt_pages_after = cache._tables["tgt"].logical_pages
    # Target's page 0 must now differ from source's page 0.
    assert tgt_pages_after[0] != src_pages_before[0]
    # Source still sees its original token 0.
    src_k, _ = cache.read("src", 0, 1)
    assert not torch.allclose(src_k[0], new_k)
    # Target now sees the overwritten value.
    tgt_k, _ = cache.read("tgt", 0, 1)
    assert torch.allclose(tgt_k[0], new_k)
