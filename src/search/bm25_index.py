"""BM25 lexical index.

Inspired by Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework:
BM25 and Beyond"; clean-room implementation. License: MIT.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

_MAX_DOC_ID_LEN = 512
_MAX_TEXT_LEN = 1_000_000
_MAX_TOP_K = 1000


@dataclass
class BM25Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: dict[str, dict[str, int]] = {}       # doc_id -> {token: count}
        self._lengths: dict[str, int] = {}             # doc_id -> token count
        self._df: dict[str, int] = defaultdict(int)    # token -> doc freq
        self._metadata: dict[str, dict] = {}

    @property
    def _avgdl(self) -> float:
        if not self._lengths:
            return 0.0
        return sum(self._lengths.values()) / len(self._lengths)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        if len(doc_id) > _MAX_DOC_ID_LEN:
            raise ValueError(f"doc_id exceeds {_MAX_DOC_ID_LEN} chars")
        if len(text) > _MAX_TEXT_LEN:
            raise ValueError(f"text exceeds {_MAX_TEXT_LEN} chars")
        if doc_id in self._tf:
            raise ValueError(f"doc_id already exists: {doc_id!r}")
        tokens = self._tokenize(text)
        if not tokens:
            raise ValueError("text is empty after tokenization")
        counts: dict[str, int] = defaultdict(int)
        for tok in tokens:
            counts[tok] += 1
        self._tf[doc_id] = dict(counts)
        self._lengths[doc_id] = len(tokens)
        self._metadata[doc_id] = dict(metadata) if metadata else {}
        for tok in counts:
            self._df[tok] += 1

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._tf:
            raise KeyError(f"doc_id not found: {doc_id!r}")
        for tok in self._tf[doc_id]:
            self._df[tok] -= 1
            if self._df[tok] <= 0:
                del self._df[tok]
        del self._tf[doc_id]
        del self._lengths[doc_id]
        del self._metadata[doc_id]

    def _idf(self, token: str) -> float:
        N = len(self._tf)
        df = self._df.get(token, 0)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _score(self, doc_id: str, tokens: list[str]) -> float:
        tf_doc = self._tf.get(doc_id, {})
        dl = self._lengths.get(doc_id, 0)
        avgdl = self._avgdl
        score = 0.0
        for tok in tokens:
            tf = tf_doc.get(tok, 0)
            idf = self._idf(tok)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1))
            score += idf * (tf * (self.k1 + 1)) / denom if denom > 0 else 0.0
        return score

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > _MAX_TOP_K:
            raise ValueError(f"top_k must be <= {_MAX_TOP_K}")
        if not self._tf:
            return []
        tokens = self._tokenize(text)
        if not tokens:
            return []
        scored = [(doc_id, self._score(doc_id, tokens)) for doc_id in self._tf]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._tf)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._tf


BM25_INDEX = BM25Index()
