"""
tokenizer_cache.py
Caches tokenization results to avoid re-tokenizing the same text.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedTokenization:
    """Immutable record of a tokenization result stored in the cache."""

    text: str
    tokens: list
    token_strs: list
    hash_key: str


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------


class TokenizerCache:
    """LRU cache for tokenization results keyed by text hash."""

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size: int = max_size
        self._store: OrderedDict = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        """Return a 16-character hex digest (SHA-256) of *text*."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[CachedTokenization]:
        """Return the cached result for *text*, or None on a cache miss.

        A hit moves the entry to the end (most-recently-used position).
        """
        key = self._hash(text)
        if key in self._store:
            # Move to MRU end
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(
        self,
        text: str,
        tokens: list,
        token_strs: list,
    ) -> CachedTokenization:
        """Store a tokenization result and return the new CachedTokenization.

        If the cache is at capacity the least-recently-used entry is evicted
        before the new entry is inserted.
        """
        key = self._hash(text)
        entry = CachedTokenization(
            text=text,
            tokens=list(tokens),
            token_strs=list(token_strs),
            hash_key=key,
        )
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = entry
            return entry

        if len(self._store) >= self._max_size:
            # Evict LRU (first item)
            self._store.popitem(last=False)

        self._store[key] = entry
        return entry

    def invalidate(self, text: str) -> bool:
        """Remove the entry for *text* from the cache.

        Returns True if the entry existed and was removed, False otherwise.
        """
        key = self._hash(text)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries from the cache (counters are preserved)."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def hit_rate(self) -> float:
        """Return hits / (hits + misses), or 0.0 when no requests made."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict:
        """Return a summary dict with size, max_size, hits, misses, hit_rate."""
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate(),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOKENIZER_CACHE_REGISTRY: dict = {"default": TokenizerCache}
