"""Tests for src/inference/semantic_cache.py

Tests the new API: SemanticCacheConfig, CacheEntry, TextEmbedder,
compute_cosine_similarity, SemanticCache, CachedInferenceEngine.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F

from src.inference.semantic_cache import (
    CachedInferenceEngine,
    CacheEntry,
    SemanticCache,
    SemanticCacheConfig,
    TextEmbedder,
    compute_cosine_similarity,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Tiny model fixture (fast, no GPU required)
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


@pytest.fixture(scope="module")
def tiny_model() -> AureliusTransformer:
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


def _tokenizer_encode(text: str) -> list[int]:
    """Simple byte-level tokenizer, clamped to vocab_size=256."""
    return [b % 256 for b in text.encode("utf-8")][:64] or [0]


def _tokenizer_decode(ids) -> str:
    """Decode byte token IDs back to a string, ignoring errors."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Test 1: SemanticCacheConfig defaults
# ---------------------------------------------------------------------------


def test_semantic_cache_config_defaults():
    cfg = SemanticCacheConfig()
    assert cfg.max_size == 1000
    assert cfg.similarity_threshold == 0.85
    assert cfg.embedding_dim == 64
    assert cfg.eviction_policy == "lru"
    assert cfg.ttl_seconds == 3600.0


# ---------------------------------------------------------------------------
# Test 2: CacheEntry fields
# ---------------------------------------------------------------------------


def test_cache_entry_fields():
    emb = torch.ones(64)
    before = time.time()
    entry = CacheEntry(key_text="hello", key_embedding=emb, value="world")
    after = time.time()

    assert entry.key_text == "hello"
    assert torch.equal(entry.key_embedding, emb)
    assert entry.value == "world"
    assert entry.hits == 0
    assert before <= entry.created_at <= after + 0.1
    assert before <= entry.last_accessed <= after + 0.1


# ---------------------------------------------------------------------------
# Test 3: compute_cosine_similarity -- identical vectors -> 1.0
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sim = compute_cosine_similarity(v, v)
    assert sim == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 4: compute_cosine_similarity -- orthogonal vectors -> 0.0
# ---------------------------------------------------------------------------


def test_cosine_similarity_orthogonal():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    sim = compute_cosine_similarity(a, b)
    assert sim == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 5: SemanticCache.lookup returns None when empty
# ---------------------------------------------------------------------------


def test_lookup_returns_none_when_empty():
    cache = SemanticCache(SemanticCacheConfig())
    query = F.normalize(torch.ones(64), dim=0)
    result = cache.lookup(query)
    assert result is None


# ---------------------------------------------------------------------------
# Test 6: SemanticCache.insert increases size
# ---------------------------------------------------------------------------


def test_insert_increases_size():
    cache = SemanticCache(SemanticCacheConfig())
    assert len(cache) == 0
    emb = F.normalize(torch.ones(64), dim=0)
    cache.insert("prompt", emb, "response")
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# Test 7: SemanticCache.lookup finds inserted entry (same embedding)
# ---------------------------------------------------------------------------


def test_lookup_finds_inserted_entry():
    cfg = SemanticCacheConfig(similarity_threshold=0.85)
    cache = SemanticCache(cfg)
    emb = F.normalize(torch.ones(64), dim=0)
    cache.insert("hello world", emb, "cached response")

    result = cache.lookup(emb)
    assert result is not None
    assert result.key_text == "hello world"
    assert result.value == "cached response"
    assert result.hits == 1


# ---------------------------------------------------------------------------
# Test 8: SemanticCache.lookup returns None for dissimilar query
# ---------------------------------------------------------------------------


def test_lookup_returns_none_for_dissimilar():
    cfg = SemanticCacheConfig(similarity_threshold=0.99)
    cache = SemanticCache(cfg)

    # Insert a vector in the positive direction
    pos = F.normalize(torch.ones(64), dim=0)
    cache.insert("positive", pos, "pos_response")

    # Use all-negative to guarantee very low similarity (cosine ~ -1)
    anti = F.normalize(-torch.ones(64), dim=0)
    result = cache.lookup(anti)
    assert result is None


# ---------------------------------------------------------------------------
# Test 9: SemanticCache.insert respects max_size (LRU eviction)
# ---------------------------------------------------------------------------


def test_insert_respects_max_size_lru():
    cfg = SemanticCacheConfig(max_size=3, eviction_policy="lru")
    cache = SemanticCache(cfg)

    # Insert 3 distinct embeddings
    for i in range(3):
        direction = torch.zeros(64)
        direction[i] = 1.0
        cache.insert(f"prompt_{i}", direction, f"response_{i}")

    assert len(cache) == 3

    # Insert one more -- should evict oldest (prompt_0)
    direction = torch.zeros(64)
    direction[3] = 1.0
    cache.insert("prompt_3", direction, "response_3")

    assert len(cache) == 3
    # prompt_0 should be gone
    assert not any(e.key_text == "prompt_0" for e in cache._entries)


# ---------------------------------------------------------------------------
# Test 10: SemanticCache.invalidate removes entry
# ---------------------------------------------------------------------------


def test_invalidate_removes_entry():
    cache = SemanticCache(SemanticCacheConfig())
    emb = F.normalize(torch.ones(64), dim=0)
    cache.insert("to_remove", emb, "value")
    assert len(cache) == 1

    removed = cache.invalidate("to_remove")
    assert removed is True
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Test 11: SemanticCache.invalidate returns False for missing key
# ---------------------------------------------------------------------------


def test_invalidate_returns_false_for_missing():
    cache = SemanticCache(SemanticCacheConfig())
    result = cache.invalidate("nonexistent_key")
    assert result is False


# ---------------------------------------------------------------------------
# Test 12: SemanticCache.stats returns required keys
# ---------------------------------------------------------------------------


def test_stats_returns_required_keys():
    cache = SemanticCache(SemanticCacheConfig())
    s = cache.stats()
    assert "size" in s
    assert "hits" in s
    assert "misses" in s
    assert "hit_rate" in s
    assert isinstance(s["size"], int)
    assert isinstance(s["hits"], int)
    assert isinstance(s["misses"], int)
    assert isinstance(s["hit_rate"], float)


# ---------------------------------------------------------------------------
# Test 13: CachedInferenceEngine.generate returns (str, bool)
# ---------------------------------------------------------------------------


def test_cached_inference_engine_generate_returns_str_bool(tiny_model):
    embedder = TextEmbedder(tiny_model, d_model=TINY_CFG.d_model)
    cfg = SemanticCacheConfig(similarity_threshold=0.85, embedding_dim=TINY_CFG.d_model)
    cache = SemanticCache(cfg)
    engine = CachedInferenceEngine(
        model=tiny_model,
        cache=cache,
        tokenizer_encode=_tokenizer_encode,
        tokenizer_decode=_tokenizer_decode,
        embedder=embedder,
    )
    result, is_hit = engine.generate("hello", max_new_tokens=4)
    assert isinstance(result, str)
    assert isinstance(is_hit, bool)
    assert is_hit is False  # first call is always a miss


# ---------------------------------------------------------------------------
# Test 14: CachedInferenceEngine.generate cache hit on repeated query
# ---------------------------------------------------------------------------


def test_cached_inference_engine_cache_hit_on_repeat(tiny_model):
    embedder = TextEmbedder(tiny_model, d_model=TINY_CFG.d_model)
    cfg = SemanticCacheConfig(similarity_threshold=0.85, embedding_dim=TINY_CFG.d_model)
    cache = SemanticCache(cfg)
    engine = CachedInferenceEngine(
        model=tiny_model,
        cache=cache,
        tokenizer_encode=_tokenizer_encode,
        tokenizer_decode=_tokenizer_decode,
        embedder=embedder,
    )
    prompt = "repeated prompt for cache"
    result1, hit1 = engine.generate(prompt, max_new_tokens=4)
    result2, hit2 = engine.generate(prompt, max_new_tokens=4)

    assert hit1 is False  # first call: miss
    assert hit2 is True  # second call: hit
    assert result1 == result2  # same result


# ---------------------------------------------------------------------------
# Test 15: SemanticCache.stats hit_rate in [0, 1]
# ---------------------------------------------------------------------------


def test_stats_hit_rate_in_range():
    cfg = SemanticCacheConfig(similarity_threshold=0.85)
    cache = SemanticCache(cfg)

    # No lookups yet -- hit_rate should be 0.0
    s = cache.stats()
    assert 0.0 <= s["hit_rate"] <= 1.0
    assert s["hit_rate"] == 0.0

    emb = F.normalize(torch.ones(64), dim=0)
    cache.insert("test", emb, "response")

    # Hit: same embedding
    cache.lookup(emb)
    # Miss: anti-parallel embedding (cosine sim ~ -1)
    anti = F.normalize(-torch.ones(64), dim=0)
    cache.lookup(anti)

    s2 = cache.stats()
    assert 0.0 <= s2["hit_rate"] <= 1.0
    assert s2["hits"] == 1
    assert s2["misses"] == 1
    assert s2["hit_rate"] == pytest.approx(0.5, abs=1e-5)
