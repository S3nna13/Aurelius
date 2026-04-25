"""Response cache for Aurelius serving infrastructure.

Stores (prompt, model_name) → completion pairs with TTL expiry and LRU
eviction when the capacity cap (128 entries) is reached.

Key derivation uses SHA-256 (usedforsecurity=False) over the concatenation of
prompt_text and model_name so bandit does not flag it as a security usage.

Pure stdlib only.  Thread-safe via threading.Lock.
"""

from __future__ import annotations

import hashlib
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class CachePolicy(str, Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL_ONLY = "ttl_only"
    HYBRID = "hybrid"


@dataclass
class CachedResponse:
    key: str
    response: str
    created_at: float
    ttl: float
    hits: int = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        t = now if now is not None else time.time()
        return (t - self.created_at) >= self.ttl


class ResponseCache:
    """LRU + TTL response cache with a default capacity of 128 entries."""

    _MAX_CAPACITY = 128

    def __init__(self, capacity: int = _MAX_CAPACITY) -> None:
        self._capacity = capacity
        # OrderedDict used as an LRU store (most-recently-used at the end)
        self._store: OrderedDict[str, CachedResponse] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(prompt_text: str, model_name: str) -> str:
        """Derive a cache key from *prompt_text* and *model_name*."""
        raw = (prompt_text + model_name).encode()
        return hashlib.sha256(raw, usedforsecurity=False).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Return cached response string, or ``None`` if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[key]
                return None
            # Move to end (most-recently used)
            self._store.move_to_end(key)
            entry.hits += 1
            return entry.response

    def put(self, key: str, response: str, ttl: float = 300.0) -> None:
        """Store *response* under *key* with the given *ttl* (seconds)."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                entry = self._store[key]
                entry.response = response
                entry.created_at = time.time()
                entry.ttl = ttl
                return

            # Evict LRU entries until we have room
            while len(self._store) >= self._capacity:
                self._store.popitem(last=False)

            self._store[key] = CachedResponse(
                key=key,
                response=response,
                created_at=time.time(),
                ttl=ttl,
            )

    def invalidate(self, key: str) -> None:
        """Remove *key* from the cache (no-op if absent)."""
        with self._lock:
            self._store.pop(key, None)

    def evict_expired(self) -> int:
        """Remove all expired entries; return number of entries removed."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._store.items() if v.is_expired(now)]
            for k in expired:
                del self._store[k]
            return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

SERVING_REGISTRY: dict = {
    "response_cache": ResponseCache(),
}
