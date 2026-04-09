"""Prefix caching: reuse KV cache across requests sharing common prefixes (system prompt caching)."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix KV cache."""

    max_cached_prefixes: int = 64
    max_prefix_tokens: int = 512
    eviction_policy: str = "lru"  # "lru" | "fifo" | "random"
    compression_enabled: bool = False
    ttl_seconds: float = 3600.0


@dataclass
class CacheEntry:
    """A single cached prefix entry."""

    prefix_hash: str
    prefix_ids: list[int]
    kv_cache: list[tuple[Tensor, Tensor]]  # one (K, V) pair per layer
    n_layers: int
    created_at: float
    last_accessed: float
    access_count: int
    prefix_len: int


def hash_prefix(token_ids: list[int]) -> str:
    """Return SHA256 hex digest of the token ids bytes."""
    data = b"".join(i.to_bytes(4, "little") for i in token_ids)
    return hashlib.sha256(data).hexdigest()


def find_longest_prefix_match(
    token_ids: list[int], cache: dict[str, CacheEntry]
) -> tuple[str | None, int]:
    """Find the longest cached prefix that matches the start of token_ids.

    Returns (cache_key, matched_length), or (None, 0) if no match.
    """
    best_key: str | None = None
    best_len: int = 0

    for key, entry in cache.items():
        plen = entry.prefix_len
        if plen <= len(token_ids) and token_ids[:plen] == entry.prefix_ids:
            if plen > best_len:
                best_len = plen
                best_key = key

    return best_key, best_len


class PrefixCache:
    """LRU/FIFO/random cache for KV caches keyed on prefix token ids."""

    def __init__(self, config: PrefixCacheConfig) -> None:
        self.config = config
        self._cache: dict[str, CacheEntry] = {}
        self._insertion_order: list[str] = []  # for FIFO eviction

    def get(self, prefix_ids: list[int]) -> CacheEntry | None:
        """Look up exact prefix match by hash; update access time and count."""
        key = hash_prefix(prefix_ids)
        entry = self._cache.get(key)
        if entry is not None:
            entry.last_accessed = time.time()
            entry.access_count += 1
        return entry

    def put(self, prefix_ids: list[int], kv_cache: list[tuple[Tensor, Tensor]]) -> None:
        """Add entry; evict according to policy if at capacity."""
        key = hash_prefix(prefix_ids)
        if key in self._cache:
            # Update existing entry
            entry = self._cache[key]
            entry.kv_cache = kv_cache
            entry.last_accessed = time.time()
            entry.access_count += 1
            return

        # Evict if at capacity
        if len(self._cache) >= self.config.max_cached_prefixes:
            self._evict_one()

        now = time.time()
        entry = CacheEntry(
            prefix_hash=key,
            prefix_ids=list(prefix_ids),
            kv_cache=kv_cache,
            n_layers=len(kv_cache),
            created_at=now,
            last_accessed=now,
            access_count=0,
            prefix_len=len(prefix_ids),
        )
        self._cache[key] = entry
        self._insertion_order.append(key)

    def _evict_one(self) -> None:
        """Evict a single entry according to the configured policy."""
        if not self._cache:
            return

        policy = self.config.eviction_policy
        if policy == "lru":
            # Evict least recently accessed
            victim = min(self._cache.values(), key=lambda e: e.last_accessed)
            victim_key = victim.prefix_hash
        elif policy == "fifo":
            # Evict first inserted that is still in cache
            victim_key = None
            for k in self._insertion_order:
                if k in self._cache:
                    victim_key = k
                    break
            if victim_key is None:
                victim_key = next(iter(self._cache))
        else:  # random
            victim_key = random.choice(list(self._cache.keys()))

        del self._cache[victim_key]
        if victim_key in self._insertion_order:
            self._insertion_order.remove(victim_key)

    def evict_expired(self) -> int:
        """Remove entries older than ttl_seconds. Returns count removed."""
        cutoff = time.time() - self.config.ttl_seconds
        expired = [k for k, e in self._cache.items() if e.created_at < cutoff]
        for k in expired:
            del self._cache[k]
            if k in self._insertion_order:
                self._insertion_order.remove(k)
        return len(expired)

    def stats(self) -> dict[str, int]:
        """Returns size, total_accesses, and capacity."""
        total_accesses = sum(e.access_count for e in self._cache.values())
        return {
            "size": len(self._cache),
            "total_accesses": total_accesses,
            "capacity": self.config.max_cached_prefixes,
        }

    def clear(self) -> None:
        """Empty the cache."""
        self._cache.clear()
        self._insertion_order.clear()


class PrefixCachedInference:
    """Inference engine that leverages prefix caching."""

    def __init__(
        self,
        model: nn.Module,
        cache: PrefixCache,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.cache = cache
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    @torch.no_grad()
    def _compute_kv_cache(self, prefix_ids: list[int]) -> list[tuple[Tensor, Tensor]]:
        """Run model forward on prefix_ids to generate KV cache entries."""
        input_tensor = torch.tensor([prefix_ids], dtype=torch.long)
        output = self.model(input_tensor)
        # output is (loss, logits, past_key_values)
        past_key_values = output[2] if len(output) > 2 else None
        if past_key_values is None:
            return []
        return list(past_key_values)

    def cache_prefix(self, prefix_text: str) -> int:
        """Encode prefix_text, compute KV cache, store in cache. Returns prefix token length."""
        prefix_ids = self.tokenizer_encode(prefix_text)
        kv = self._compute_kv_cache(prefix_ids)
        self.cache.put(prefix_ids, kv)
        return len(prefix_ids)

    @torch.no_grad()
    def generate_with_cache(self, prompt: str, max_new_tokens: int = 8) -> str:
        """Encode prompt, find matching prefix in cache, greedy decode from cached position.

        Returns generated text (excluding prompt).
        """
        prompt_ids = self.tokenizer_encode(prompt)

        # Find a matching prefix in cache
        cache_key, match_len = find_longest_prefix_match(prompt_ids, self.cache._cache)

        past_key_values: list[tuple[Tensor, Tensor]] | None = None
        start_pos = 0

        if cache_key is not None and match_len > 0:
            entry = self.cache._cache[cache_key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            past_key_values = entry.kv_cache if entry.kv_cache else None
            start_pos = match_len

        # Build remaining input ids (tokens after the cached prefix)
        remaining_ids = prompt_ids[start_pos:]
        if not remaining_ids and past_key_values is None:
            remaining_ids = prompt_ids

        generated: list[int] = []
        cur_ids = torch.tensor([remaining_ids], dtype=torch.long) if remaining_ids else None

        for _ in range(max_new_tokens):
            if cur_ids is not None:
                _, logits, past_key_values = self.model(
                    cur_ids, past_key_values=past_key_values if past_key_values else None
                )
            else:
                break

            # Greedy: pick argmax of last token logits
            next_token_id = int(logits[0, -1].argmax().item())
            generated.append(next_token_id)

            # Next step: only the new token
            cur_ids = torch.tensor([[next_token_id]], dtype=torch.long)

        return self.tokenizer_decode(generated)


def compute_cache_hit_rate(cache: PrefixCache, queries: list[list[int]]) -> float:
    """Simulate queries against cache; return hit rate (fraction with match)."""
    if not queries:
        return 0.0
    hits = 0
    for q in queries:
        key, matched = find_longest_prefix_match(q, cache._cache)
        if key is not None and matched > 0:
            hits += 1
    return hits / len(queries)
