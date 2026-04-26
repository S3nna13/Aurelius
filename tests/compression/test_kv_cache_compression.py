"""Tests for KV cache compression."""

from __future__ import annotations

from src.compression.kv_cache_compression import (
    KV_CACHE_COMPRESSOR,
    EvictionPolicy,
    KVCacheCompressor,
    KVEntry,
)

# --- EvictionPolicy enum ---


def test_eviction_policy_recency():
    assert EvictionPolicy.RECENCY == "recency"


def test_eviction_policy_attention_score():
    assert EvictionPolicy.ATTENTION_SCORE == "attention_score"


def test_eviction_policy_random():
    assert EvictionPolicy.RANDOM == "random"


def test_eviction_policy_h2o():
    assert EvictionPolicy.H2O == "h2o"


# --- KVEntry fields ---


def test_kventry_required_fields():
    e = KVEntry(token_id=1, key=[0.1], value=[0.2])
    assert e.token_id == 1
    assert e.key == [0.1]
    assert e.value == [0.2]


def test_kventry_default_attention_score():
    e = KVEntry(token_id=1, key=[], value=[])
    assert e.attention_score == 0.0


def test_kventry_default_position():
    e = KVEntry(token_id=1, key=[], value=[])
    assert e.position == 0


def test_kventry_custom_attention_score():
    e = KVEntry(token_id=5, key=[1.0], value=[2.0], attention_score=0.9)
    assert e.attention_score == 0.9


def test_kventry_custom_position():
    e = KVEntry(token_id=5, key=[1.0], value=[2.0], position=7)
    assert e.position == 7


# --- KVCacheCompressor.add ---


def test_add_increases_len():
    c = KVCacheCompressor(max_tokens=10)
    e = KVEntry(token_id=1, key=[0.1], value=[0.2])
    c.add(e)
    assert len(c) == 1


def test_add_multiple():
    c = KVCacheCompressor(max_tokens=10)
    for i in range(5):
        c.add(KVEntry(token_id=i, key=[float(i)], value=[float(i)]))
    assert len(c) == 5


def test_add_entries_accessible():
    c = KVCacheCompressor(max_tokens=10)
    e = KVEntry(token_id=42, key=[1.0], value=[2.0])
    c.add(e)
    assert c.entries()[0].token_id == 42


# --- Eviction on overflow ---


def test_add_over_max_triggers_eviction():
    c = KVCacheCompressor(max_tokens=3, policy=EvictionPolicy.RECENCY)
    for i in range(5):
        c.add(KVEntry(token_id=i, key=[float(i)], value=[float(i)]))
    assert len(c) == 3


def test_len_never_exceeds_max_tokens():
    c = KVCacheCompressor(max_tokens=5, policy=EvictionPolicy.ATTENTION_SCORE)
    for i in range(20):
        c.add(KVEntry(token_id=i, key=[float(i)], value=[float(i)], attention_score=float(i)))
    assert len(c) <= 5


# --- RECENCY eviction ---


def test_recency_removes_oldest():
    c = KVCacheCompressor(max_tokens=3, policy=EvictionPolicy.RECENCY)
    for i in range(4):
        c.add(KVEntry(token_id=i, key=[float(i)], value=[float(i)]))
    ids = [e.token_id for e in c.entries()]
    assert 0 not in ids


def test_recency_keeps_newest():
    c = KVCacheCompressor(max_tokens=3, policy=EvictionPolicy.RECENCY)
    for i in range(5):
        c.add(KVEntry(token_id=i, key=[float(i)], value=[float(i)]))
    ids = [e.token_id for e in c.entries()]
    assert 4 in ids
    assert 3 in ids
    assert 2 in ids


# --- ATTENTION_SCORE eviction ---


def test_attention_score_removes_lowest():
    c = KVCacheCompressor(max_tokens=3, policy=EvictionPolicy.ATTENTION_SCORE)
    entries = [
        KVEntry(token_id=0, key=[0.0], value=[0.0], attention_score=0.1),
        KVEntry(token_id=1, key=[1.0], value=[1.0], attention_score=0.9),
        KVEntry(token_id=2, key=[2.0], value=[2.0], attention_score=0.5),
        KVEntry(token_id=3, key=[3.0], value=[3.0], attention_score=0.8),
    ]
    for e in entries:
        c.add(e)
    ids = [e.token_id for e in c.entries()]
    # token_id=0 has lowest score, should be evicted
    assert 0 not in ids


def test_attention_score_keeps_highest():
    c = KVCacheCompressor(max_tokens=2, policy=EvictionPolicy.ATTENTION_SCORE)
    entries = [
        KVEntry(token_id=0, key=[0.0], value=[0.0], attention_score=0.1),
        KVEntry(token_id=1, key=[1.0], value=[1.0], attention_score=0.9),
        KVEntry(token_id=2, key=[2.0], value=[2.0], attention_score=0.5),
    ]
    for e in entries:
        c.add(e)
    ids = [e.token_id for e in c.entries()]
    assert 1 in ids  # highest attention score kept


# --- compress_int8 ---


def test_compress_int8_returns_tuple_of_three():
    c = KVCacheCompressor()
    result = c.compress_int8([1.0, 2.0, 3.0])
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_compress_int8_quants_are_list_of_int():
    c = KVCacheCompressor()
    quants, scale, zero = c.compress_int8([1.0, 2.0, 3.0])
    assert isinstance(quants, list)
    assert all(isinstance(q, int) for q in quants)


def test_compress_int8_values_in_0_255():
    c = KVCacheCompressor()
    values = [-10.0, 0.0, 5.0, 100.0, 255.0]
    quants, scale, zero = c.compress_int8(values)
    assert all(0 <= q <= 255 for q in quants)


def test_compress_int8_scale_positive():
    c = KVCacheCompressor()
    _, scale, _ = c.compress_int8([0.0, 1.0, 2.0])
    assert scale > 0


def test_compress_int8_uniform_values():
    c = KVCacheCompressor()
    quants, scale, zero = c.compress_int8([5.0, 5.0, 5.0])
    # All same -> scale is 1e-8, all quants are 0
    assert all(q == 0 for q in quants)


# --- decompress_int8 ---


def test_decompress_int8_roundtrip():
    c = KVCacheCompressor()
    values = [1.0, 2.5, 3.7, 0.0, 5.0]
    quants, scale, zero = c.compress_int8(values)
    recovered = c.decompress_int8(quants, scale, zero)
    assert len(recovered) == len(values)
    for orig, rec in zip(values, recovered):
        assert abs(orig - rec) <= scale + 1e-9


def test_decompress_int8_zero_quant():
    c = KVCacheCompressor()
    result = c.decompress_int8([0], 0.5, 1.0)
    assert abs(result[0] - 1.0) < 1e-9


def test_decompress_int8_nonzero_quant():
    c = KVCacheCompressor()
    result = c.decompress_int8([255], 1.0, 0.0)
    assert abs(result[0] - 255.0) < 1e-9


# --- KV_CACHE_COMPRESSOR singleton ---


def test_kv_cache_compressor_exists():
    assert KV_CACHE_COMPRESSOR is not None


def test_kv_cache_compressor_is_instance():
    assert isinstance(KV_CACHE_COMPRESSOR, KVCacheCompressor)


def test_kv_cache_compressor_default_max():
    assert KV_CACHE_COMPRESSOR.max_tokens == 512


# --- __len__ and entries ---


def test_len_empty():
    c = KVCacheCompressor()
    assert len(c) == 0


def test_entries_returns_list():
    c = KVCacheCompressor()
    assert isinstance(c.entries(), list)


def test_entries_is_copy():
    c = KVCacheCompressor()
    e = KVEntry(token_id=1, key=[1.0], value=[1.0])
    c.add(e)
    entries = c.entries()
    entries.clear()
    assert len(c) == 1
