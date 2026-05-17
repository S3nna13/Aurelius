"""Minimal Tier-2 episodic hook for AMC-first experiments.

This is intentionally small. It is not the full Aurelian Memory Core; it is the
first stable seam that lets runtimes decide whether a turn is surprising enough
to write into episodic memory and retrieve a compact prompt context later.
"""

from __future__ import annotations

from dataclasses import dataclass

from .episodic_memory import EpisodicMemory, MemoryEntry


@dataclass(frozen=True)
class AMCTier2Config:
    """Configuration for the Tier-2 episodic memory hook."""

    surprise_threshold: float = 0.5
    max_retrieved: int = 5
    default_importance: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.surprise_threshold <= 1.0:
            raise ValueError("surprise_threshold must be between 0.0 and 1.0")
        if self.max_retrieved < 1:
            raise ValueError("max_retrieved must be >= 1")
        if self.default_importance < 0.0:
            raise ValueError("default_importance must be >= 0.0")


class AMCTier2Hook:
    """Small AMC Tier-2 adapter around the existing episodic memory store."""

    def __init__(
        self,
        config: AMCTier2Config | None = None,
        episodic_memory: EpisodicMemory | None = None,
    ) -> None:
        self.config = config or AMCTier2Config()
        self.episodic = episodic_memory or EpisodicMemory()

    def observe(
        self,
        role: str,
        content: str,
        *,
        surprise: float,
        importance: float | None = None,
    ) -> MemoryEntry | None:
        """Store one event if it crosses the Tier-2 surprise gate."""
        if not content.strip():
            return None
        if not 0.0 <= surprise <= 1.0:
            raise ValueError("surprise must be between 0.0 and 1.0")
        if surprise < self.config.surprise_threshold:
            return None
        return self.episodic.store(
            role=role,
            content=content,
            importance=self.config.default_importance if importance is None else importance,
        )

    def retrieve(self, query: str, *, limit: int | None = None) -> list[MemoryEntry]:
        """Retrieve matching Tier-2 memories, falling back to recent entries."""
        n = limit or self.config.max_retrieved
        if n < 1:
            raise ValueError("limit must be >= 1")
        matches = self.episodic.search(query.strip()) if query.strip() else []
        if matches:
            return matches[:n]
        return self.episodic.retrieve_recent(n)

    def build_context(self, query: str, *, limit: int | None = None) -> str:
        """Build a compact prompt-context block from retrieved Tier-2 memories."""
        entries = self.retrieve(query, limit=limit)
        if not entries:
            return ""
        lines = ["Tier-2 episodic memory:"]
        lines.extend(f"- {entry.role}: {entry.content}" for entry in entries)
        return "\n".join(lines)

    def stats(self) -> dict[str, int | float]:
        """Return small operational stats for health checks and benchmark logs."""
        return {
            "episodic_entries": len(self.episodic),
            "surprise_threshold": self.config.surprise_threshold,
            "max_retrieved": self.config.max_retrieved,
        }


__all__ = ["AMCTier2Config", "AMCTier2Hook"]
