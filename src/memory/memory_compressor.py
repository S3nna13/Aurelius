"""
memory_compressor.py — Compresses memory stores via deduplication and summarization stubs.

Part of the Aurelius memory subsystem. Stdlib-only, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CompressionStrategy(Enum):
    """Strategies for compressing a memory store."""
    DEDUP = "dedup"
    TRUNCATE = "truncate"
    MERGE_SIMILAR = "merge_similar"


@dataclass(frozen=True)
class CompressionResult:
    """Immutable summary of a compression operation."""
    original_count: int
    compressed_count: int
    strategy: CompressionStrategy
    removed_ids: list[str]


class MemoryCompressor:
    """Compresses a list of memory dicts using various strategies.

    Each memory is expected to be a ``dict`` with at least:
    - ``"id"`` (str)  — unique identifier
    - ``"content"`` (str) — textual content used for comparison
    """

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.similarity_threshold = similarity_threshold
        self.last_result_memories: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Character-level Jaccard similarity between two strings."""
        set_a = set(a)
        set_b = set(b)
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def deduplicate(self, memories: list[dict]) -> CompressionResult:
        """Remove exact duplicate memories by content (case-insensitive, stripped).

        The first occurrence of each unique content is kept. Results are stored
        in ``self.last_result_memories``.

        Parameters
        ----------
        memories:
            List of dicts, each with ``"id"`` and ``"content"`` keys.

        Returns
        -------
        CompressionResult
        """
        original_count = len(memories)
        seen_contents: set[str] = set()
        kept: list[dict] = []
        removed_ids: list[str] = []

        for mem in memories:
            normalised = mem["content"].strip().lower()
            if normalised in seen_contents:
                removed_ids.append(mem["id"])
            else:
                seen_contents.add(normalised)
                kept.append(mem)

        self.last_result_memories = kept
        return CompressionResult(
            original_count=original_count,
            compressed_count=len(kept),
            strategy=CompressionStrategy.DEDUP,
            removed_ids=removed_ids,
        )

    def truncate(self, memories: list[dict], max_count: int) -> CompressionResult:
        """Keep the first *max_count* memories; discard the rest.

        Parameters
        ----------
        memories:
            List of dicts, each with ``"id"`` and ``"content"`` keys.
        max_count:
            Maximum number of memories to retain.

        Returns
        -------
        CompressionResult
        """
        original_count = len(memories)
        kept = memories[:max_count]
        removed = memories[max_count:]
        removed_ids = [m["id"] for m in removed]

        self.last_result_memories = kept
        return CompressionResult(
            original_count=original_count,
            compressed_count=len(kept),
            strategy=CompressionStrategy.TRUNCATE,
            removed_ids=removed_ids,
        )

    def merge_similar(self, memories: list[dict]) -> CompressionResult:
        """Merge memories whose content similarity exceeds ``self.similarity_threshold``.

        Uses character-level Jaccard similarity. For merged groups the first
        item is kept. Results are stored in ``self.last_result_memories``.

        Parameters
        ----------
        memories:
            List of dicts, each with ``"id"`` and ``"content"`` keys.

        Returns
        -------
        CompressionResult
        """
        original_count = len(memories)
        merged_into: dict[int, int] = {}  # index -> representative index
        removed_ids: list[str] = []

        for i in range(len(memories)):
            if i in merged_into:
                continue
            for j in range(i + 1, len(memories)):
                if j in merged_into:
                    continue
                sim = self._jaccard(
                    memories[i]["content"].strip().lower(),
                    memories[j]["content"].strip().lower(),
                )
                if sim > self.similarity_threshold:
                    merged_into[j] = i
                    removed_ids.append(memories[j]["id"])

        kept = [m for idx, m in enumerate(memories) if idx not in merged_into]
        self.last_result_memories = kept
        return CompressionResult(
            original_count=original_count,
            compressed_count=len(kept),
            strategy=CompressionStrategy.MERGE_SIMILAR,
            removed_ids=removed_ids,
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def compress(
        self,
        memories: list[dict],
        strategy: CompressionStrategy,
        **kwargs: Any,
    ) -> CompressionResult:
        """Dispatch compression to the appropriate strategy method.

        Parameters
        ----------
        memories:
            List of memory dicts.
        strategy:
            Which ``CompressionStrategy`` to apply.
        **kwargs:
            Extra keyword arguments forwarded to the strategy (e.g.
            ``max_count`` for TRUNCATE).

        Returns
        -------
        CompressionResult
        """
        if strategy is CompressionStrategy.DEDUP:
            return self.deduplicate(memories)
        elif strategy is CompressionStrategy.TRUNCATE:
            max_count = kwargs.get("max_count", len(memories))
            return self.truncate(memories, max_count=max_count)
        elif strategy is CompressionStrategy.MERGE_SIMILAR:
            return self.merge_similar(memories)
        else:
            raise ValueError(f"Unknown compression strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MEMORY_COMPRESSOR_REGISTRY: dict[str, type] = {
    "default": MemoryCompressor,
}
