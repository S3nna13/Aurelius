"""Pseudo-relevance feedback (PRF) query expansion for sparse retrieval.

Classic Rocchio / PRF (Rocchio, JASIS 1971; relevance models in IR textbooks)
selects high-IDF terms from top-ranked documents and appends them to the
original query to improve recall.  This implementation is **pure stdlib**:
whitespace tokenisation, document-frequency statistics across the candidate
set, and deterministic tie-breaking.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.ASCII)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass(frozen=True)
class PRFExpansionResult:
    """Expanded query string plus the injected terms (for attribution)."""

    query: str
    added_terms: tuple[str, ...]


class PRFQueryExpander:
    """Expand a query using pseudo-relevance feedback over document strings."""

    def __init__(self, *, min_doc_freq: int = 1, max_term_len: int = 64) -> None:
        if min_doc_freq < 1:
            raise ValueError("min_doc_freq must be >= 1")
        if max_term_len < 1:
            raise ValueError("max_term_len must be >= 1")
        self._min_doc_freq = min_doc_freq
        self._max_term_len = max_term_len

    def expand(
        self,
        query: str,
        documents: list[str],
        *,
        num_terms: int = 8,
    ) -> PRFExpansionResult:
        """Return a new query with up to ``num_terms`` PRF terms appended."""
        if not isinstance(query, str):
            raise TypeError("query must be str")
        if not isinstance(documents, list):
            raise TypeError("documents must be a list of str")
        if num_terms < 0:
            raise ValueError("num_terms must be >= 0")
        if num_terms > 256:
            raise ValueError("num_terms must be <= 256")

        q_terms = set(_tokens(query))
        if not documents:
            if num_terms > 0:
                raise RuntimeError("prf_query_expander: empty documents with num_terms>0")
            return PRFExpansionResult(query=query.strip(), added_terms=())

        doc_freq: Counter[str] = Counter()
        collection_tf: Counter[str] = Counter()
        for doc in documents:
            if not isinstance(doc, str):
                raise TypeError("every document must be str")
            toks = _tokens(doc)
            collection_tf.update(toks)
            doc_freq.update(set(toks))

        scored: list[tuple[str, int, int]] = []
        for tok, df in doc_freq.items():
            if df < self._min_doc_freq:
                continue
            if tok in q_terms:
                continue
            if len(tok) > self._max_term_len:
                continue
            tf = collection_tf[tok]
            scored.append((tok, tf, df))

        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        added = tuple(t for t, _, _ in scored[:num_terms])
        if not added:
            return PRFExpansionResult(query=query, added_terms=())
        suffix = " ".join(added)
        sep = "" if query.endswith((" ", "\n", "\t")) else " "
        return PRFExpansionResult(query=f"{query}{sep}{suffix}", added_terms=added)


__all__ = ["PRFExpansionResult", "PRFQueryExpander"]
