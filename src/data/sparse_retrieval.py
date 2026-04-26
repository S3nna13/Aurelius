"""
sparse_retrieval.py — BM25 and TF-IDF sparse retrieval for RAG pipelines.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25  # IDF floor


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def tokenize_bm25(text: str) -> list[str]:
    """Lowercase and split on whitespace/punctuation, filtering empty strings."""
    tokens = re.split(r'[\s\.,!?;:()\[\]{}"\']+', text.lower())
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


class BM25Index:
    def __init__(self, config: BM25Config = None) -> None:
        self.config = config or BM25Config()
        self._documents: list[str] = []
        self._doc_freqs: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    def build(self, documents: list[str]) -> None:
        self._documents = list(documents)
        self._doc_freqs = []
        self._doc_lengths = []

        # Tokenize and count
        for doc in self._documents:
            tokens = tokenize_bm25(doc)
            self._doc_freqs.append(Counter(tokens))
            self._doc_lengths.append(len(tokens))

        N = len(self._documents)
        self._avgdl = sum(self._doc_lengths) / N if N > 0 else 0.0

        # Compute per-term document frequency
        df: dict[str, int] = {}
        for freq_counter in self._doc_freqs:
            for term in freq_counter:
                df[term] = df.get(term, 0) + 1

        # Compute IDF (Robertson-Sparck Jones variant, clipped to epsilon)
        self._idf = {}
        for term, doc_freq in df.items():
            idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            self._idf[term] = max(idf, self.config.epsilon)

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for query against the document at doc_idx."""
        query_terms = tokenize_bm25(query)
        freq_counter = self._doc_freqs[doc_idx]
        dl = self._doc_lengths[doc_idx]
        k1 = self.config.k1
        b = self.config.b
        avgdl = self._avgdl

        score = 0.0
        for term in query_terms:
            if term not in self._idf:
                continue
            tf = freq_counter.get(term, 0)
            idf = self._idf[term]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else tf + k1
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """Return top_k (score, document) pairs sorted by score descending."""
        scores = [(self.score(query, i), doc) for i, doc in enumerate(self._documents)]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    def __len__(self) -> int:
        return len(self._documents)


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------


class TF_IDF_Index:
    def __init__(self) -> None:
        self._documents: list[str] = []
        self._tfidf_vecs: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def build(self, documents: list[str]) -> None:
        self._documents = list(documents)
        N = len(self._documents)

        # Tokenize
        tokenized = [tokenize_bm25(doc) for doc in self._documents]

        # Document frequency
        df: dict[str, int] = {}
        for tokens in tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        # IDF (same formula as BM25, clipped at 0)
        self._idf = {}
        for term, doc_freq in df.items():
            self._idf[term] = max(math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1), 0.0)

        # TF-IDF vectors
        self._tfidf_vecs = []
        for tokens in tokenized:
            doc_len = len(tokens)
            counter = Counter(tokens)
            vec: dict[str, float] = {}
            for term, cnt in counter.items():
                tf = cnt / doc_len if doc_len > 0 else 0.0
                vec[term] = tf * self._idf.get(term, 0.0)
            self._tfidf_vecs.append(vec)

    def _vec_norm(self, vec: dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values()))

    def _cosine(self, q_vec: dict[str, float], d_vec: dict[str, float]) -> float:
        dot = sum(q_vec.get(t, 0.0) * d_vec.get(t, 0.0) for t in q_vec)
        denom = self._vec_norm(q_vec) * self._vec_norm(d_vec)
        return dot / denom if denom > 0 else 0.0

    def _query_vec(self, query: str) -> dict[str, float]:
        tokens = tokenize_bm25(query)
        doc_len = len(tokens)
        counter = Counter(tokens)
        vec: dict[str, float] = {}
        for term, cnt in counter.items():
            tf = cnt / doc_len if doc_len > 0 else 0.0
            vec[term] = tf * self._idf.get(term, 0.0)
        return vec

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        q_vec = self._query_vec(query)
        results = []
        for i, doc in enumerate(self._documents):
            sim = self._cosine(q_vec, self._tfidf_vecs[i])
            results.append((sim, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    def __init__(self, bm25_weight: float = 0.6, tfidf_weight: float = 0.4) -> None:
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        self._bm25 = BM25Index()
        self._tfidf = TF_IDF_Index()
        self._documents: list[str] = []

    def build(self, documents: list[str]) -> None:
        self._documents = list(documents)
        self._bm25.build(documents)
        self._tfidf.build(documents)

    def _normalize(self, scores: list[float]) -> list[float]:
        mn, mx = min(scores), max(scores)
        rng = mx - mn
        if rng == 0:
            return [0.0] * len(scores)
        return [(s - mn) / rng for s in scores]

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        n = len(self._documents)
        if n == 0:
            return []

        bm25_raw = [self._bm25.score(query, i) for i in range(n)]
        tfidf_results = {doc: sc for sc, doc in self._tfidf.search(query, top_k=n)}
        tfidf_raw = [tfidf_results.get(doc, 0.0) for doc in self._documents]

        bm25_norm = self._normalize(bm25_raw)
        tfidf_norm = self._normalize(tfidf_raw)

        combined = [
            (self.bm25_weight * b + self.tfidf_weight * t, doc)
            for b, t, doc in zip(bm25_norm, tfidf_norm, self._documents)
        ]
        combined.sort(key=lambda x: x[0], reverse=True)
        return combined[:top_k]


# ---------------------------------------------------------------------------
# RAGContext
# ---------------------------------------------------------------------------


@dataclass
class RAGContext:
    query: str
    retrieved_docs: list[str]
    scores: list[float]


def build_rag_context(
    query: str,
    retriever: Any,
    prompt_template: str = "Context:\n{context}\n\nQuestion: {query}",
) -> str:
    """Retrieve docs and format into a prompt string."""
    results = retriever.search(query)
    docs = [doc for _, doc in results]
    context = "\n\n".join(docs)
    return prompt_template.format(context=context, query=query)
