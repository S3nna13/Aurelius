"""Model garbage collection manager for Aurelius runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto


class GCPolicy(Enum):
    EAGER = auto()
    DEFERRED = auto()
    LRU = auto()
    NONE = auto()


@dataclass
class ModelHandle:
    model_id: str
    size_bytes: int
    last_used: float
    ref_count: int = 0


class ModelGCManager:
    """Manages model lifecycle and memory via configurable GC policy."""

    def __init__(
        self,
        max_memory_bytes: int = 4 * 1024**3,
        policy: GCPolicy = GCPolicy.LRU,
    ) -> None:
        self.max_memory_bytes = max_memory_bytes
        self.policy = policy
        self._registry: dict[str, ModelHandle] = {}

    def register(self, handle: ModelHandle) -> None:
        """Add a ModelHandle to the registry."""
        self._registry[handle.model_id] = handle

    def acquire(self, model_id: str) -> bool:
        """Increment ref_count and refresh last_used. Returns False if not registered."""
        if model_id not in self._registry:
            return False
        handle = self._registry[model_id]
        handle.ref_count += 1
        handle.last_used = time.monotonic()
        return True

    def release(self, model_id: str) -> bool:
        """Decrement ref_count (floor 0). Returns False if not registered."""
        if model_id not in self._registry:
            return False
        handle = self._registry[model_id]
        handle.ref_count = max(0, handle.ref_count - 1)
        return True

    def evict_candidates(self) -> list[str]:
        """Return model_ids eligible for eviction based on current policy."""
        if self.policy is GCPolicy.NONE or self.policy is GCPolicy.DEFERRED:
            return []

        free_handles = [h for h in self._registry.values() if h.ref_count == 0]

        if self.policy is GCPolicy.EAGER:
            return [h.model_id for h in free_handles]

        # LRU: sort by last_used ascending (oldest first)
        sorted_handles = sorted(free_handles, key=lambda h: h.last_used)
        return [h.model_id for h in sorted_handles]

    def total_memory(self) -> int:
        """Return total size_bytes of all registered handles."""
        return sum(h.size_bytes for h in self._registry.values())

    def evict(self, model_id: str) -> bool:
        """Remove model from registry. Returns True if it existed."""
        if model_id in self._registry:
            del self._registry[model_id]
            return True
        return False

    def status(self) -> list[dict]:
        """Return a list of dicts describing all registered handles."""
        return [
            {
                "id": h.model_id,
                "size_bytes": h.size_bytes,
                "last_used": h.last_used,
                "ref_count": h.ref_count,
            }
            for h in self._registry.values()
        ]


MODEL_GC_REGISTRY: dict[str, type[ModelGCManager]] = {"default": ModelGCManager}
