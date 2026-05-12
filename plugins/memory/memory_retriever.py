"""BM25-backed retrieval over LTM and episodic memory entries.

Combines BM25 lexical scoring with LTM importance scores for re-ranking.
Inspired by memory retrieval in Generative Agents (Park et al. 2303.17580)
and MemGPT (Packer et al. 2310.08560). License: MIT.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from math import log
from typing import Any

_MAX_QUERY_LEN = 2048
_MAX_TOP_K = 500


@dataclass
class RetrievalResult:
    key: str
    value: Any
    bm25_score: float
    importance: float
    combined_score: float
    snippet: str = ""  # first 200 chars of str(value)


class MemoryRetriever:
    """BM25 index over text representations of memory entries.

    Indexes entries by concatenating key + str(value)[:512].
    Combined score = alpha * bm25_norm + (1 - alpha) * importance.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, alpha: float = 0.7) -> None:
        """alpha: weight on BM25 score (1-alpha goes to importance)."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.alpha = alpha
        self._tf: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        self._df: dict[str, int] = defaultdict(int)
        self._importance: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _text_for(self, key: str, value: Any) -> str:
        return f"{key} {str(value)[:512]}"

    @property
    def _avgdl(self) -> float:
        if not self._lengths:
            return 0.0
        return sum(self._lengths.values()) / len(self._lengths)

    def index(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Add or update a memory entry in the index."""
        if key in self._tf:
            self.remove(key)
        text = self._text_for(key, value)
        tokens = self._tokenize(text)
        if not tokens:
            return
        counts: dict[str, int] = defaultdict(int)
        for tok in tokens:
            counts[tok] += 1
        self._tf[key] = dict(counts)
        self._lengths[key] = len(tokens)
        self._importance[key] = max(0.0, min(1.0, importance))
        for tok in counts:
            self._df[tok] += 1

    def remove(self, key: str) -> None:
        """Remove an entry from the index."""
        if key not in self._tf:
            return
        for tok in self._tf[key]:
            self._df[tok] -= 1
            if self._df[tok] <= 0:
                del self._df[tok]
        del self._tf[key]
        del self._lengths[key]
        del self._importance[key]

    def _bm25_score(self, key: str, query_tokens: list[str]) -> float:
        tf_doc = self._tf.get(key, {})
        dl = self._lengths.get(key, 0)
        avgdl = self._avgdl
        N = len(self._tf)
        score = 0.0
        for tok in query_tokens:
            tf = tf_doc.get(tok, 0)
            df = self._df.get(tok, 0)
            idf = log((N - df + 0.5) / (df + 0.5) + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1))
            score += idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)
        return score

    def query(
        self, text: str, top_k: int = 10, importance_floor: float = 0.0
    ) -> list[RetrievalResult]:
        """Query the index. Returns top_k results by combined score."""
        if len(text) > _MAX_QUERY_LEN:
            raise ValueError(f"query exceeds {_MAX_QUERY_LEN} chars")
        if top_k > _MAX_TOP_K:
            raise ValueError(f"top_k exceeds {_MAX_TOP_K}")
        if not self._tf:
            return []
        tokens = self._tokenize(text)
        if not tokens:
            return []

        raw_scores = {k: self._bm25_score(k, tokens) for k in self._tf}
        max_bm25 = max(raw_scores.values()) if raw_scores else 1.0

        results = []
        for key, bm25 in raw_scores.items():
            imp = self._importance.get(key, 0.5)
            if imp < importance_floor:
                continue
            bm25_norm = bm25 / max(max_bm25, 1e-9)
            combined = self.alpha * bm25_norm + (1 - self.alpha) * imp
            results.append(
                RetrievalResult(
                    key=key,
                    value=None,  # caller fills value from LTM
                    bm25_score=bm25,
                    importance=imp,
                    combined_score=combined,
                    snippet="",
                )
            )
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._tf)


MEMORY_RETRIEVER = MemoryRetriever()
