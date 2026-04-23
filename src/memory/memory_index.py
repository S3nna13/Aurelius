"""Memory index: BM25-style inverted index over memory entries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


@dataclass
class IndexedEntry:
    """Represents a single indexed document."""

    entry_id: str
    tokens: list[str]
    content: str


class MemoryIndex:
    """Inverted index with TF-IDF-like scoring for memory entry retrieval."""

    def __init__(self) -> None:
        self._entries: dict[str, IndexedEntry] = {}
        self._inverted: dict[str, set[str]] = {}  # token -> set of entry_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric characters."""
        return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def index(self, entry_id: str, content: str) -> None:
        """Index *content* under *entry_id*."""
        tokens = self._tokenize(content)
        indexed = IndexedEntry(entry_id=entry_id, tokens=tokens, content=content)
        self._entries[entry_id] = indexed
        for token in tokens:
            self._inverted.setdefault(token, set()).add(entry_id)

    def remove(self, entry_id: str) -> bool:
        """Remove *entry_id* from index. Returns True if it existed."""
        if entry_id not in self._entries:
            return False
        entry = self._entries.pop(entry_id)
        for token in set(entry.tokens):
            if token in self._inverted:
                self._inverted[token].discard(entry_id)
                if not self._inverted[token]:
                    del self._inverted[token]
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top_k (entry_id, score) pairs sorted by TF-IDF-like score."""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self._entries:
            return []

        n_docs = len(self._entries)
        scores: dict[str, float] = {}

        for token in query_tokens:
            if token not in self._inverted:
                continue
            df = len(self._inverted[token])
            idf = math.log(n_docs / df + 1)
            for entry_id in self._inverted[token]:
                entry = self._entries[entry_id]
                tf = entry.tokens.count(token)
                scores[entry_id] = scores.get(entry_id, 0.0) + tf * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)


MEMORY_INDEX = MemoryIndex()
