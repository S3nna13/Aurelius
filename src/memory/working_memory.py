"""Working memory: key-value short-term store with TTL eviction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class WorkingMemorySlot:
    """A single slot in working memory with TTL support."""

    key: str
    value: object
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 60.0


class WorkingMemory:
    """Fixed-capacity key-value store with per-slot TTL eviction."""

    def __init__(self, capacity: int = 16) -> None:
        self._capacity = capacity
        self._slots: dict[str, WorkingMemorySlot] = {}

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    def is_expired(self, slot: WorkingMemorySlot) -> bool:
        """Return True if the slot has exceeded its TTL."""
        return time.monotonic() - slot.created_at > slot.ttl_seconds

    def evict_expired(self) -> int:
        """Remove all expired slots. Returns count removed."""
        expired_keys = [k for k, s in self._slots.items() if self.is_expired(s)]
        for k in expired_keys:
            del self._slots[k]
        return len(expired_keys)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def set(self, key: str, value: object, ttl_seconds: float = 60.0) -> None:
        """Store a value under *key*, evicting expired slots first.

        If the store is at capacity after eviction, the oldest slot
        (by created_at) is evicted to make room.
        """
        self.evict_expired()
        # If already exists, just replace
        if key not in self._slots and len(self._slots) >= self._capacity:
            # Evict oldest slot
            oldest_key = min(self._slots, key=lambda k: self._slots[k].created_at)
            del self._slots[oldest_key]
        self._slots[key] = WorkingMemorySlot(
            key=key, value=value, ttl_seconds=ttl_seconds
        )

    def get(self, key: str) -> object | None:
        """Return value for *key* if not expired, else evict and return None."""
        slot = self._slots.get(key)
        if slot is None:
            return None
        if self.is_expired(slot):
            del self._slots[key]
            return None
        return slot.value

    def keys(self) -> list[str]:
        """Return list of non-expired keys."""
        return [k for k, s in self._slots.items() if not self.is_expired(s)]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Count of non-expired slots."""
        return sum(1 for s in self._slots.values() if not self.is_expired(s))


WORKING_MEMORY = WorkingMemory()
