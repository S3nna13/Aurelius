"""Progressive 3-layer memory search inspired by claude-mem."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class ProgressiveSearchError(Exception):
    """Raised for errors during progressive memory search operations."""


@dataclass
class IndexEntry:
    """Lightweight index entry for Layer-1 compact search."""

    entry_id: str
    summary_tags: list[str]
    timestamp: float
    layer: int
    access_count: int = 0


@dataclass
class SearchResult:
    """Result returned after Layer-2 / Layer-3 expansion."""

    entry_id: str
    score: float
    timeline_context: list[str]
    full_content: str | None


class ProgressiveSearcher:
    """3-layer progressive searcher: compact index → timeline context → full details."""

    def __init__(self, index: list[IndexEntry] | None = None) -> None:
        self._index: list[IndexEntry] = list(index) if index is not None else []
        self._entry_map: dict[str, IndexEntry] = {e.entry_id: e for e in self._index}
        self._full_content: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Index mutation
    # ------------------------------------------------------------------

    def index_entry(self, entry: IndexEntry) -> None:
        """Add or overwrite an entry in the compact index."""
        # Remove existing entry with same ID to avoid duplicates in _index
        if entry.entry_id in self._entry_map:
            self._index = [e for e in self._index if e.entry_id != entry.entry_id]
        self._index.append(entry)
        self._entry_map[entry.entry_id] = entry

    def set_full_content(self, entry_id: str, content: str) -> None:
        """Store full content for Layer-3 retrieval."""
        self._full_content[entry_id] = content

    def remove(self, entry_id: str) -> bool:
        """Remove *entry_id* from index and full-content store. Returns True if present."""
        if entry_id not in self._entry_map:
            return False
        self._index = [e for e in self._index if e.entry_id != entry_id]
        del self._entry_map[entry_id]
        self._full_content.pop(entry_id, None)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        timeline_radius: int = 2,
        fetch_full: bool = True,
    ) -> list[SearchResult]:
        """Run 3-layer progressive search and return ranked *SearchResult* objects."""
        if top_k < 0:
            raise ProgressiveSearchError("top_k must be non-negative")
        if timeline_radius < 0:
            raise ProgressiveSearchError("timeline_radius must be non-negative")
        if not self._index:
            return []

        query_terms = [t.lower() for t in query.split() if t]
        if not query_terms:
            return []

        # Layer 1: compact index scoring
        candidates = self._layer1_candidates(query_terms, top_k * 2)

        # Build timestamp-sorted positional list once
        ts_sorted = sorted(self._index, key=lambda e: e.timestamp)

        results: list[SearchResult] = []
        for entry, score in candidates:
            timeline = self._layer2_timeline(entry, ts_sorted, timeline_radius)
            full = self._layer3_full(entry.entry_id) if fetch_full else None
            results.append(
                SearchResult(
                    entry_id=entry.entry_id,
                    score=score,
                    timeline_context=timeline,
                    full_content=full,
                )
            )

        return results

    def _layer1_candidates(
        self, query_terms: list[str], candidate_limit: int
    ) -> list[tuple[IndexEntry, float]]:
        """Score entries by keyword overlap + recency bonus, return top *candidate_limit*."""
        max_ts = max(e.timestamp for e in self._index) if self._index else 1.0
        scored: list[tuple[IndexEntry, float]] = []

        for entry in self._index:
            tags_lower = {t.lower() for t in entry.summary_tags}
            overlap = sum(1 for qt in query_terms if qt in tags_lower)
            if overlap == 0:
                continue
            # Recency bonus: up to 0.1 for newest entry
            recency = (entry.timestamp / max_ts) * 0.1 if max_ts else 0.0
            score = overlap + recency
            scored.append((entry, score))

        # Sort by score desc, then access_count desc as tie-break
        scored.sort(key=lambda x: (x[1], x[0].access_count), reverse=True)
        return scored[:candidate_limit]

    def _layer2_timeline(
        self, entry: IndexEntry, ts_sorted: list[IndexEntry], radius: int
    ) -> list[str]:
        """Fetch surrounding temporal context within ±*radius* in timestamp-sorted index."""
        # Find position in timestamp-sorted list
        try:
            pos = next(
                i for i, e in enumerate(ts_sorted) if e.entry_id == entry.entry_id
            )
        except StopIteration:
            return []

        start = max(0, pos - radius)
        end = min(len(ts_sorted), pos + radius + 1)
        neighbors: list[IndexEntry] = []
        for i in range(start, end):
            if i == pos:
                continue
            neighbors.append(ts_sorted[i])

        # Sort by timestamp proximity to the candidate
        neighbors.sort(key=lambda n: abs(n.timestamp - entry.timestamp))
        return [self._format_summary(n) for n in neighbors]

    @staticmethod
    def _format_summary(entry: IndexEntry) -> str:
        """Format an entry as a brief summary string."""
        tags = " ".join(entry.summary_tags)
        return f"{entry.entry_id} [{entry.layer}]: {tags}"

    def _layer3_full(self, entry_id: str) -> str | None:
        """Retrieve full content for *entry_id* if available."""
        return self._full_content.get(entry_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int | float]:
        """Return statistics about the current index state."""
        size = len(self._index)
        if size == 0:
            return {
                "index_size": 0,
                "avg_tags_per_entry": 0.0,
                "full_content_store_size": 0,
                "avg_access_count": 0.0,
            }

        total_tags = sum(len(e.summary_tags) for e in self._index)
        total_access = sum(e.access_count for e in self._index)
        return {
            "index_size": size,
            "avg_tags_per_entry": total_tags / size,
            "full_content_store_size": len(self._full_content),
            "avg_access_count": total_access / size,
        }


# ------------------------------------------------------------------
# Singleton & registry
# ------------------------------------------------------------------

DEFAULT_PROGRESSIVE_SEARCHER = ProgressiveSearcher()
PROGRESSIVE_SEARCH_REGISTRY: dict[str, ProgressiveSearcher] = {
    "default": DEFAULT_PROGRESSIVE_SEARCHER,
}
