"""
associative_memory.py — Associative memory with Hopfield-inspired pattern matching.

Pure-Python implementation; no ML frameworks or numerical libraries required.
Part of the Aurelius memory subsystem. Stdlib-only, no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    """An associative memory pattern identified by a string key."""
    key: str
    features: list[float]


class AssociativeMemory:
    """Content-addressable memory store backed by cosine-similarity recall.

    Parameters
    ----------
    capacity:
        Maximum number of patterns that can be stored simultaneously.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._store: dict[str, Pattern] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def store(self, pattern: Pattern) -> None:
        """Store a pattern.

        If the key already exists it is overwritten without consuming extra
        capacity. If the store is full and the key is new a ValueError is raised.

        Parameters
        ----------
        pattern:
            The pattern to store.

        Raises
        ------
        ValueError
            If capacity is exceeded when adding a new key.
        """
        if pattern.key not in self._store and len(self._store) >= self.capacity:
            raise ValueError(
                f"AssociativeMemory capacity ({self.capacity}) exceeded. "
                "Forget a pattern before storing a new one."
            )
        self._store[pattern.key] = pattern

    def recall(
        self, query_features: list[float], top_k: int = 1
    ) -> list[Pattern]:
        """Retrieve the *top_k* most similar patterns by cosine similarity.

        Parameters
        ----------
        query_features:
            Feature vector to match against stored patterns.
        top_k:
            Number of best-matching patterns to return.

        Returns
        -------
        list[Pattern]
            Ordered from most to least similar. Empty list if the store is empty.
        """
        if not self._store:
            return []

        scored: list[tuple[float, Pattern]] = [
            (self.similarity(query_features, p.features), p)
            for p in self._store.values()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def forget(self, key: str) -> bool:
        """Remove a pattern by key.

        Parameters
        ----------
        key:
            The key of the pattern to remove.

        Returns
        -------
        bool
            True if the pattern was found and removed, False otherwise.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def __len__(self) -> int:
        """Return the number of stored patterns."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Similarity metric
    # ------------------------------------------------------------------

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two feature vectors.

        sim = dot(a, b) / (|a| * |b| + 1e-10)

        Returns 0.0 for zero vectors (no crash).

        Parameters
        ----------
        a, b:
            Feature vectors (must be the same length for meaningful results;
            shorter vector is zero-padded implicitly via zip).

        Returns
        -------
        float
            Value in [-1.0, 1.0].
        """
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        return dot / (norm_a * norm_b + 1e-10)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ASSOCIATIVE_MEMORY_REGISTRY: dict[str, type] = {
    "default": AssociativeMemory,
}
