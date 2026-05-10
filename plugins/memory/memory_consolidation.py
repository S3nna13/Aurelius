"""Memory consolidation: compress episodic → semantic, importance decay."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum


class ConsolidationPolicy(StrEnum):
    RECENCY = "recency"
    IMPORTANCE = "importance"
    FREQUENCY = "frequency"


@dataclass
class ConsolidationResult:
    """Result of a consolidation pass."""

    consolidated_count: int
    summary: str
    dropped_ids: list[str]


class MemoryConsolidator:
    """Consolidates episodic memory entries into semantic memory."""

    def __init__(
        self,
        policy: ConsolidationPolicy = ConsolidationPolicy.IMPORTANCE,
        decay_factor: float = 0.95,
    ) -> None:
        self.policy = policy
        self.decay_factor = decay_factor

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay_importance(self, entries, steps: int = 1) -> list:
        """Return a new list where each entry.importance is multiplied by decay_factor^steps.

        Non-destructive: uses dataclasses.replace to copy entries.
        """
        factor = self.decay_factor**steps
        return [dataclasses.replace(e, importance=e.importance * factor) for e in entries]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_for_consolidation(
        self, entries, threshold: float = 0.3, max_count: int = 10
    ) -> list:
        """Return entries with importance < threshold, sorted by importance asc, up to max_count."""
        below = [e for e in entries if e.importance < threshold]
        below.sort(key=lambda e: e.importance)
        return below[:max_count]

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, entries, semantic_memory=None) -> ConsolidationResult:
        """Consolidate low-importance entries, optionally into semantic memory.

        Selects entries with importance < 0.3 (up to 10), adds their tags/roles
        as concepts in semantic_memory if provided, and returns a ConsolidationResult.
        """
        selected = self.select_for_consolidation(entries, threshold=0.3, max_count=10)
        n = len(selected)

        if semantic_memory is not None:
            seen: set[str] = set()
            for entry in selected:
                # Use tags attribute if present, else fall back to role
                tags = getattr(entry, "tags", None)
                if tags:
                    for tag in tags:
                        if tag not in seen:
                            seen.add(tag)
                            if semantic_memory.get_concept(tag) is None:
                                semantic_memory.add_concept(tag)
                role = getattr(entry, "role", None)
                if role and role not in seen:
                    seen.add(role)
                    if semantic_memory.get_concept(role) is None:
                        semantic_memory.add_concept(role)

        summary = (
            f"Consolidated {n} memories: " + "; ".join(entry.content[:30] for entry in selected)
            if selected
            else f"Consolidated {n} memories: "
        )

        dropped_ids = [e.id for e in selected]
        return ConsolidationResult(
            consolidated_count=n,
            summary=summary,
            dropped_ids=dropped_ids,
        )
