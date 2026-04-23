"""Semantic search: TF-IDF vector index over text corpus."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SemanticDocument:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class SemanticSearch:
    """TF-IDF index for semantic search over a text corpus."""

    def __init__(self) -> None:
        # doc_id -> {token: count}
        self._tf: dict[str, dict[str, int]] = {}
        # doc_id -> total token count
        self._lengths: dict[str, int] = {}
        # token -> number of docs containing it
        self._df: dict[str, int] = defaultdict(int)

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Add a document to the index."""
        if doc_id in self._tf:
            self.remove(doc_id)

        tokens = re.findall(r"\w+", text.lower())
        counts: dict[str, int] = defaultdict(int)
        for tok in tokens:
            counts[tok] += 1

        self._tf[doc_id] = dict(counts)
        self._lengths[doc_id] = len(tokens)

        for tok in counts:
            self._df[tok] += 1

    def compute_tfidf(self, doc_id: str, token: str) -> float:
        """Compute TF-IDF for a token in a document."""
        if doc_id not in self._tf:
            return 0.0
        count = self._tf[doc_id].get(token, 0)
        total = self._lengths.get(doc_id, 0)
        if total == 0:
            return 0.0
        tf = count / total
        N = len(self._tf)
        df = self._df.get(token, 0)
        idf = math.log((N + 1) / (df + 1) + 1)
        return tf * idf

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Score all documents against the query, return top_k (doc_id, score) pairs descending."""
        if not self._tf:
            return []

        tokens = re.findall(r"\w+", text.lower())
        scores: dict[str, float] = {}
        for doc_id in self._tf:
            score = sum(self.compute_tfidf(doc_id, tok) for tok in tokens)
            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the index. Returns True if found."""
        if doc_id not in self._tf:
            return False

        for tok in self._tf[doc_id]:
            self._df[tok] -= 1
            if self._df[tok] <= 0:
                del self._df[tok]

        del self._tf[doc_id]
        del self._lengths[doc_id]
        return True

    def __len__(self) -> int:
        return len(self._tf)


SEMANTIC_SEARCH = SemanticSearch()
