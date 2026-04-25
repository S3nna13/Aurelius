from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import torch


@dataclass
class PrefixEntry:
    prefix_hash: str
    token_ids: list[int]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    n_layers: int
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class PrefixKVCache:
    """Exact-match prefix KV cache with LRU eviction."""

    def __init__(self, max_entries: int = 512, max_prefix_tokens: int = 2048) -> None:
        self._max_entries = max_entries
        self._max_prefix_tokens = max_prefix_tokens
        self._store: dict[str, PrefixEntry] = {}
        self._total_hits = 0
        self._total_lookups = 0

    def _hash_tokens(self, token_ids: list[int]) -> str:
        raw = bytes(bytearray(b for tid in token_ids for b in tid.to_bytes(4, "little")))
        return hashlib.sha256(raw, usedforsecurity=False).hexdigest()[:16]

    def lookup(self, token_ids: list[int]) -> PrefixEntry | None:
        self._total_lookups += 1
        h = self._hash_tokens(token_ids)
        entry = self._store.get(h)
        if entry is not None:
            entry.hit_count += 1
            entry.last_used = time.time()
            self._total_hits += 1
            return entry
        return None

    def store(
        self,
        token_ids: list[int],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> PrefixEntry:
        h = self._hash_tokens(token_ids)
        if h in self._store:
            entry = self._store[h]
            entry.k_cache = k_cache
            entry.v_cache = v_cache
            entry.last_used = time.time()
            return entry
        if len(self._store) >= self._max_entries:
            self.evict_lru()
        n_layers = k_cache.shape[0] if k_cache.dim() >= 1 else 1
        entry = PrefixEntry(
            prefix_hash=h,
            token_ids=list(token_ids),
            k_cache=k_cache,
            v_cache=v_cache,
            n_layers=n_layers,
        )
        self._store[h] = entry
        return entry

    def evict_lru(self) -> str | None:
        if not self._store:
            return None
        victim_hash = min(self._store, key=lambda k: self._store[k].last_used)
        del self._store[victim_hash]
        return victim_hash

    def hit_rate(self) -> float:
        if self._total_lookups == 0:
            return 0.0
        return self._total_hits / self._total_lookups

    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
        self._total_hits = 0
        self._total_lookups = 0

    def stats(self) -> dict:
        return {
            "size": self.size(),
            "max_entries": self._max_entries,
            "hit_rate": self.hit_rate(),
            "total_hits": self._total_hits,
            "total_lookups": self._total_lookups,
        }


@dataclass
class CacheConfig:
    min_prefix_tokens: int = 64
    max_prefix_tokens: int = 2048
    cache_system_prompts: bool = True
    ttl_s: float = 3600.0


class PromptCacheManager:
    """Manage prompt prefix caching for common system prompts and few-shot examples."""

    def __init__(
        self,
        prefix_cache: PrefixKVCache | None = None,
        config: CacheConfig | None = None,
    ) -> None:
        self._cache = prefix_cache if prefix_cache is not None else PrefixKVCache()
        self._config = config if config is not None else CacheConfig()

    def should_cache(self, token_ids: list[int]) -> bool:
        n = len(token_ids)
        return self._config.min_prefix_tokens <= n <= self._config.max_prefix_tokens

    def get_cached_prefix(
        self, full_prompt_ids: list[int]
    ) -> tuple[PrefixEntry | None, int]:
        best_entry: PrefixEntry | None = None
        best_len = 0
        now = time.time()
        for entry in list(self._cache._store.values()):
            if now - entry.created_at > self._config.ttl_s:
                continue
            plen = len(entry.token_ids)
            if plen <= len(full_prompt_ids) and full_prompt_ids[:plen] == entry.token_ids:
                if plen > best_len:
                    best_len = plen
                    best_entry = entry
        if best_entry is not None:
            best_entry.hit_count += 1
            best_entry.last_used = now
        return best_entry, best_len

    def cache_prefix(
        self,
        token_ids: list[int],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> bool:
        if not self.should_cache(token_ids):
            return False
        self._cache.store(token_ids, k_cache, v_cache)
        return True

    def prune_expired(self) -> int:
        now = time.time()
        expired = [
            h
            for h, entry in self._cache._store.items()
            if now - entry.created_at > self._config.ttl_s
        ]
        for h in expired:
            del self._cache._store[h]
        return len(expired)

    def register_system_prompt(
        self,
        prompt_tokens: list[int],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> str:
        entry = self._cache.store(prompt_tokens, k_cache, v_cache)
        return entry.prefix_hash

    def stats(self) -> dict:
        return self._cache.stats()
