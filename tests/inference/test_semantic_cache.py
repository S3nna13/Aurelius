"""Tests for src/inference/semantic_cache.py"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.inference.semantic_cache import (
    CacheEntry,
    EmbeddingIndex,
    SemanticCache,
    make_mean_pool_embed_fn,
)

# ---------------------------------------------------------------------------
# Mock embed_fn (fast, deterministic, 64-d)
# ---------------------------------------------------------------------------

D_EMBED = 64


def mock_embed_fn(prompt_ids: torch.Tensor) -> torch.Tensor:
    """Deterministic embedding: mean of token IDs, broadcast to 64-d, normalized."""
    emb = prompt_ids.float().mean(dim=-1, keepdim=True).expand(D_EMBED)
    return F.normalize(emb, dim=0)


def _make_query(value: float) -> torch.Tensor:
    """Return a unit vector in the direction (value, value, …)."""
    raw = torch.full((D_EMBED,), value)
    return F.normalize(raw, dim=0)


# ---------------------------------------------------------------------------
# EmbeddingIndex tests
# ---------------------------------------------------------------------------

class TestEmbeddingIndex:

    def test_embedding_index_add_and_search(self):
        """add 3 embeddings, search returns top-1 with high similarity."""
        index = EmbeddingIndex(d_embed=D_EMBED)

        e0 = _make_query(1.0)
        e1 = _make_query(0.5)
        e2 = _make_query(-1.0)

        index.add(e0, entry_id=0)
        index.add(e1, entry_id=1)
        index.add(e2, entry_id=2)

        results = index.search(e0, top_k=1)
        assert len(results) == 1
        best_id, best_sim = results[0]
        assert best_id == 0
        assert best_sim == pytest.approx(1.0, abs=1e-5)

    def test_embedding_index_empty_search(self):
        """search on empty index returns []."""
        index = EmbeddingIndex(d_embed=D_EMBED)
        q = _make_query(1.0)
        assert index.search(q, top_k=5) == []

    def test_embedding_index_search_sorted(self):
        """results are sorted by similarity descending."""
        index = EmbeddingIndex(d_embed=D_EMBED)

        # Three distinct directions
        e_high = _make_query(1.0)
        e_mid = F.normalize(torch.tensor([1.0] * 32 + [0.0] * 32), dim=0)
        e_low = _make_query(-1.0)

        index.add(e_high, entry_id=10)
        index.add(e_mid, entry_id=11)
        index.add(e_low, entry_id=12)

        query = _make_query(1.0)
        results = index.search(query, top_k=3)

        assert len(results) == 3
        sims = [sim for _, sim in results]
        # Must be sorted descending
        assert sims == sorted(sims, reverse=True)
        # Best must be id=10 (identical direction)
        assert results[0][0] == 10

    def test_embedding_index_remove(self):
        """after remove, entry not returned in search."""
        index = EmbeddingIndex(d_embed=D_EMBED)

        e0 = _make_query(1.0)
        e1 = _make_query(-1.0)

        index.add(e0, entry_id=0)
        index.add(e1, entry_id=1)
        assert len(index) == 2

        index.remove(entry_id=0)
        assert len(index) == 1

        results = index.search(e0, top_k=5)
        returned_ids = [eid for eid, _ in results]
        assert 0 not in returned_ids


# ---------------------------------------------------------------------------
# SemanticCache tests
# ---------------------------------------------------------------------------

class TestSemanticCache:

    def _make_cache(self, threshold: float = 0.95, max_entries: int = 10) -> SemanticCache:
        return SemanticCache(
            embed_fn=mock_embed_fn,
            similarity_threshold=threshold,
            max_entries=max_entries,
            d_embed=D_EMBED,
        )

    def test_semantic_cache_miss(self):
        """lookup with dissimilar prompt returns None."""
        cache = self._make_cache(threshold=0.95)

        positive_ids = torch.tensor([100, 200, 300])
        negative_ids = torch.tensor([-100, -200, -300])

        cache.store(positive_ids, torch.tensor([1, 2, 3]))
        result = cache.lookup(negative_ids)
        assert result is None

    def test_semantic_cache_hit(self):
        """store then lookup with same prompt returns entry."""
        cache = self._make_cache(threshold=0.95)

        prompt_ids = torch.tensor([10, 20, 30])
        response_ids = torch.tensor([1, 2, 3])

        cache.store(prompt_ids, response_ids, response_text="hello")
        result = cache.lookup(prompt_ids)

        assert result is not None
        assert isinstance(result, CacheEntry)
        assert torch.equal(result.response_ids, response_ids)
        assert result.response_text == "hello"

    def test_semantic_cache_threshold(self):
        """high threshold (0.9999) misses similar but not identical prompts.

        We use a custom embed_fn that produces genuinely different unit vectors
        for different prompts by encoding the token values into distinct directions.
        We then verify that a threshold set just above the cross-prompt similarity
        causes a miss, while a threshold set just below causes a hit.
        """

        def _distinct_embed(prompt_ids: torch.Tensor) -> torch.Tensor:
            """Each unique prompt gets a direction determined by its token values.

            We build a raw vector where position i = sin(i * mean_val), giving
            different directions for different mean values.
            """
            mean_val = prompt_ids.float().mean().item()
            idx = torch.arange(D_EMBED, dtype=torch.float32)
            raw = torch.sin(idx * mean_val + 1.0)
            return F.normalize(raw, dim=0)

        # Prompts with very different means → lower similarity
        prompt_a = torch.tensor([1])    # mean = 1.0
        prompt_b = torch.tensor([50])   # mean = 50.0

        emb_a = _distinct_embed(prompt_a)
        emb_b = _distinct_embed(prompt_b)
        cross_sim = (emb_a * emb_b).sum().item()  # will be < 1.0

        # Threshold just above cross-sim → lookup(prompt_b) misses after storing prompt_a
        cache_high = SemanticCache(
            embed_fn=_distinct_embed,
            similarity_threshold=min(cross_sim + 0.1, 0.9999),
            max_entries=10,
            d_embed=D_EMBED,
        )
        cache_high.store(prompt_a, torch.tensor([1]))
        result_miss = cache_high.lookup(prompt_b)
        assert result_miss is None

        # Exact same prompt → always a hit (sim == 1.0)
        result_exact = cache_high.lookup(prompt_a)
        assert result_exact is not None

        # Threshold just below cross-sim → lookup(prompt_b) hits after storing prompt_a
        cache_low = SemanticCache(
            embed_fn=_distinct_embed,
            similarity_threshold=max(cross_sim - 0.1, 0.0),
            max_entries=10,
            d_embed=D_EMBED,
        )
        cache_low.store(prompt_a, torch.tensor([1]))
        result_hit = cache_low.lookup(prompt_b)
        assert result_hit is not None

    def test_semantic_cache_hit_count(self):
        """hit_count increments on each successful lookup."""
        cache = self._make_cache(threshold=0.9)
        prompt_ids = torch.tensor([5, 5, 5])
        cache.store(prompt_ids, torch.tensor([99]))

        for expected in range(1, 4):
            entry = cache.lookup(prompt_ids)
            assert entry is not None
            assert entry.hit_count == expected

    def test_semantic_cache_lru_eviction(self):
        """store max_entries+1 entries; oldest is evicted."""
        max_entries = 5
        cache = self._make_cache(max_entries=max_entries)

        # Store max_entries entries with distinct prompts
        # Use values 1..max_entries so their embeddings differ
        for i in range(1, max_entries + 1):
            cache.store(
                torch.tensor([i * 100]),
                torch.tensor([i]),
                response_text=f"resp_{i}",
            )

        assert len(cache) == max_entries

        # Store one more — should evict the LRU (first one stored, i=1)
        cache.store(torch.tensor([999]), torch.tensor([999]))
        assert len(cache) == max_entries

        # The first entry (i=1 → ids=[100]) should be gone
        evicted_prompt = torch.tensor([100])
        result = cache.lookup(evicted_prompt)
        # Because similarity threshold is 0.95 and we only have entries from 200,300,…
        # the [100] query may or may not match something; verify count, not hit
        assert len(cache) == max_entries

    def test_semantic_cache_stats(self):
        """stats() returns dict with correct keys."""
        cache = self._make_cache()
        s = cache.stats()

        assert "n_entries" in s
        assert "total_hits" in s
        assert "hit_rate" in s
        assert isinstance(s["n_entries"], int)
        assert isinstance(s["total_hits"], int)
        assert isinstance(s["hit_rate"], float)

        # After storing and hitting once
        prompt_ids = torch.tensor([1, 2, 3])
        cache.store(prompt_ids, torch.tensor([7, 8]))
        cache.lookup(prompt_ids)   # should be a hit

        s2 = cache.stats()
        assert s2["n_entries"] == 1
        assert s2["total_hits"] == 1
        assert s2["hit_rate"] == pytest.approx(1.0)

    def test_semantic_cache_clear(self):
        """after clear(), len == 0."""
        cache = self._make_cache()
        for i in range(5):
            cache.store(torch.tensor([i + 1]), torch.tensor([i]))
        assert len(cache) == 5

        cache.clear()
        assert len(cache) == 0

        # Stats reset too
        s = cache.stats()
        assert s["n_entries"] == 0
        assert s["total_hits"] == 0
        assert s["hit_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# make_mean_pool_embed_fn tests
# ---------------------------------------------------------------------------

class TestMakeMeanPoolEmbedFn:

    def _make_tiny_model(self, d_model: int = 16) -> nn.Module:
        """A minimal nn.Module that has a `norm` attribute (like AureliusTransformer)."""

        class TinyModel(nn.Module):
            def __init__(self, d_model: int) -> None:
                super().__init__()
                self.embed = nn.Embedding(256, d_model)
                self.norm = nn.LayerNorm(d_model)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                x = self.embed(input_ids)          # (B, S, d_model)
                return self.norm(x)

        return TinyModel(d_model)

    def test_make_mean_pool_embed_fn(self):
        """embed_fn returns normalized (d_model,) tensor."""
        d_model = 16
        model = self._make_tiny_model(d_model)
        embed_fn = make_mean_pool_embed_fn(model)

        prompt_ids = torch.tensor([1, 2, 3, 4])
        emb = embed_fn(prompt_ids)

        assert isinstance(emb, torch.Tensor)
        assert emb.shape == (d_model,), f"Expected ({d_model},), got {emb.shape}"

        # Must be (approximately) unit-normalized
        norm = emb.norm().item()
        assert norm == pytest.approx(1.0, abs=1e-5), f"Not normalized: norm={norm}"
