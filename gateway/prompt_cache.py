"""Application-level prompt cache for agent loops.

This caches full (prompt -> completion) pairs keyed by a hash over the
prompt text plus sampling parameters. It is distinct from the KV-cache-
level prefix cache in ``src/longcontext/prefix_cache.py`` which operates
on token-id prefixes within a single decoding session.

The cache is intended for the API server to short-circuit repeated
identical requests from agent loops (e.g. planner retries hitting the
same tool-observation -> prompt mapping). Eviction is LRU by
access-order; entries may also carry a TTL.

Concurrency: single-threaded assumption. The implementation is not
thread-safe by design - callers that need multi-threaded access should
wrap operations in an external lock. This keeps the hot path cheap for
the common single-threaded serving loop.

Pure stdlib: ``hashlib``, ``time``, ``json``, ``collections``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CachedResponse:
    """One entry stored in the prompt cache."""

    prompt_hash: str
    completion: str
    created_at: float
    hit_count: int
    ttl: float | None

    def is_expired(self, now: float | None = None) -> bool:
        if self.ttl is None:
            return False
        if now is None:
            now = time.time()
        return (now - self.created_at) > self.ttl


def _default_hasher(key_material: str) -> str:
    """SHA-256 hex digest of ``key_material``."""
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def _canonical_params(params: dict | None) -> str:
    """JSON-serialize params with sorted keys for order independence.

    ``None`` and ``{}`` both collapse to ``"{}"`` so callers that elide
    the params argument still collide with callers that pass an empty
    dict (the intended behaviour).
    """
    if not params:
        return "{}"
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


class PromptCache:
    """LRU + TTL cache mapping hashed prompts to completions."""

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: float | None = None,
        hasher: Callable[[str], str] | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._hasher: Callable[[str], str] = hasher or _default_hasher
        self._store: OrderedDict[str, CachedResponse] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ---- key construction -------------------------------------------------

    def _key(self, prompt: str, params: dict | None) -> str:
        material = prompt + "\x1f" + _canonical_params(params)
        return self._hasher(material)

    # ---- core operations --------------------------------------------------

    def get(self, prompt: str, params: dict | None = None) -> CachedResponse | None:
        key = self._key(prompt, params)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired():
            # lazy expiry on access
            del self._store[key]
            self._misses += 1
            return None
        entry.hit_count += 1
        self._store.move_to_end(key)
        self._hits += 1
        return entry

    def put(
        self,
        prompt: str,
        completion: str,
        params: dict | None = None,
        ttl: float | None = None,
    ) -> None:
        key = self._key(prompt, params)
        effective_ttl = ttl if ttl is not None else self.default_ttl
        entry = CachedResponse(
            prompt_hash=key,
            completion=completion,
            created_at=time.time(),
            hit_count=0,
            ttl=effective_ttl,
        )
        if key in self._store:
            self._store[key] = entry
            self._store.move_to_end(key)
            return
        self._store[key] = entry
        # evict until under cap
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self._evictions += 1

    def invalidate(self, prompt: str, params: dict | None = None) -> bool:
        key = self._key(prompt, params)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        n = len(self._store)
        self._store.clear()
        return n

    def prune_expired(self) -> int:
        now = time.time()
        expired_keys = [k for k, v in self._store.items() if v.is_expired(now)]
        for k in expired_keys:
            del self._store[k]
        return len(expired_keys)

    # ---- introspection ----------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._store),
        }

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, prompt_and_params) -> bool:  # pragma: no cover
        # Convenience: accept a tuple ``(prompt, params)`` or bare prompt.
        if isinstance(prompt_and_params, tuple):
            prompt, params = prompt_and_params
        else:
            prompt, params = prompt_and_params, None
        return self._key(prompt, params) in self._store


__all__ = ["CachedResponse", "PromptCache"]
