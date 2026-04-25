import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum


class DedupStrategy(str, Enum):
    EXACT_HASH = "exact_hash"
    PREFIX_HASH = "prefix_hash"
    SEMANTIC_HASH = "semantic_hash"


@dataclass
class DedupEntry:
    key: str
    response: str
    hit_count: int = 0
    created_at: float = field(default_factory=time.monotonic)


class ResponseDedup:
    def __init__(
        self,
        strategy: DedupStrategy = DedupStrategy.EXACT_HASH,
        max_entries: int = 1024,
        ttl_s: float = 300.0,
    ):
        self.strategy = strategy
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._store: dict[str, DedupEntry] = {}
        self._total_hits: int = 0

    def _compute_key(self, prompt: str) -> str:
        if self.strategy == DedupStrategy.EXACT_HASH:
            return hashlib.sha256(prompt.encode()).hexdigest()[:16]
        elif self.strategy == DedupStrategy.PREFIX_HASH:
            return hashlib.sha256(prompt[:256].encode()).hexdigest()[:16]
        elif self.strategy == DedupStrategy.SEMANTIC_HASH:
            normalized = re.sub(r"\s+", " ", prompt.lower()).strip()
            return hashlib.sha256(normalized.encode()).hexdigest()[:16]
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def lookup(self, prompt: str) -> str | None:
        key = self._compute_key(prompt)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.created_at > self.ttl_s:
            del self._store[key]
            return None
        entry.hit_count += 1
        self._total_hits += 1
        return entry.response

    def store(self, prompt: str, response: str) -> None:
        key = self._compute_key(prompt)
        if key not in self._store and len(self._store) >= self.max_entries:
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]
        self._store[key] = DedupEntry(key=key, response=response)

    def invalidate(self, prompt: str) -> None:
        key = self._compute_key(prompt)
        self._store.pop(key, None)

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "hits": self._total_hits,
            "strategy": self.strategy.value,
        }

    def clear(self) -> None:
        self._store.clear()
        self._total_hits = 0


RESPONSE_DEDUP_REGISTRY: dict[str, type[ResponseDedup]] = {"default": ResponseDedup}
