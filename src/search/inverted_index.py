"""Aurelius search – classic inverted index for full-text search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Posting:
    """A single posting: which document and how often the term appears there."""

    doc_id: int
    term_freq: int


class InvertedIndex:
    """Classic inverted index: term → list[Posting]."""

    def __init__(self) -> None:
        # term -> {doc_id -> count}
        self._index: dict[str, dict[int, int]] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Index *tokens* from *doc_id*, accumulating term frequencies."""
        for token in tokens:
            if token not in self._index:
                self._index[token] = {}
            doc_map = self._index[token]
            doc_map[doc_id] = doc_map.get(doc_id, 0) + 1

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def lookup(self, term: str) -> list[Posting]:
        """Return postings for *term*, sorted ascending by doc_id."""
        doc_map = self._index.get(term, {})
        return [Posting(doc_id=doc_id, term_freq=freq) for doc_id, freq in sorted(doc_map.items())]

    def search(self, query_tokens: list[str]) -> list[int]:
        """AND search: return doc_ids present in *all* query token postings."""
        if not query_tokens:
            return []

        # Build sets of doc_ids per token; unknown token → empty set stops here
        sets: list[set] = []
        for token in query_tokens:
            doc_map = self._index.get(token)
            if doc_map is None:
                return []
            sets.append(set(doc_map.keys()))

        intersection = sets[0]
        for s in sets[1:]:
            intersection = intersection & s

        return sorted(intersection)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def doc_count(self) -> int:
        """Number of distinct documents ever indexed."""
        all_docs: set = set()
        for doc_map in self._index.values():
            all_docs.update(doc_map.keys())
        return len(all_docs)

    def vocab_size(self) -> int:
        """Number of distinct terms in the index."""
        return len(self._index)


INVERTED_INDEX_REGISTRY: dict[str, type] = {"default": InvertedIndex}
