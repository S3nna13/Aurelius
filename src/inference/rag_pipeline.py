"""Full RAG pipeline: chunking, BM25-style sparse retrieval, dense fusion, and generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    chunk_size: int = 256        # tokens (words) per chunk
    chunk_overlap: int = 32      # overlap words between consecutive chunks
    n_retrieve: int = 5          # docs to retrieve (sparse + dense each)
    n_rerank: int = 3            # top docs after reranking / RRF fusion
    fusion_alpha: float = 0.5    # weight of dense vs sparse score
    max_context_len: int = 1024  # max context length in words


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-level chunks.

    Each chunk has at most chunk_size words with overlap words shared with the
    previous chunk. Returns an empty list if text is empty.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------

class BM25Index:
    """Simple BM25 scoring index (Robertson & Zaragoza 2009).

    Parameters
    ----------
    k1 : float
        Term saturation parameter (default 1.5).
    b : float
        Length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: list[str] = []
        self._tf: list[dict[str, int]] = []   # per-doc term frequency
        self._df: dict[str, int] = {}          # document frequency per term
        self._avgdl: float = 0.0
        self._n: int = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def index(self, documents: list[str]) -> None:
        """Tokenize documents and compute IDF statistics."""
        self._documents = list(documents)
        self._tf = []
        self._df = {}
        self._n = len(documents)

        total_len = 0
        for doc in documents:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)
            for tok in set(tokens):
                self._df[tok] = self._df.get(tok, 0) + 1

        self._avgdl = total_len / self._n if self._n > 0 else 0.0

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against document at doc_idx."""
        if not self._documents or doc_idx >= self._n:
            return 0.0

        query_tokens = self._tokenize(query)
        tf_i = self._tf[doc_idx]
        dl = sum(tf_i.values())
        result = 0.0

        for tok in query_tokens:
            tf_tok = tf_i.get(tok, 0)
            df_tok = self._df.get(tok, 0)
            if df_tok == 0:
                continue
            idf = math.log((self._n - df_tok + 0.5) / (df_tok + 0.5) + 1)
            numerator = tf_tok * (self.k1 + 1)
            denominator = tf_tok + self.k1 * (
                1 - self.b + self.b * dl / self._avgdl
            ) if self._avgdl > 0 else tf_tok + self.k1
            result += idf * numerator / denominator

        return result

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return top-k (doc_idx, score) pairs sorted by score descending."""
        if not self._documents:
            return []

        scores = [(i, self.score(query, i)) for i in range(self._n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Combine multiple ranked lists via Reciprocal Rank Fusion.

    score(d) = sum_r 1 / (k + rank_r(d))   where rank is 1-based.

    Parameters
    ----------
    rankings : list of ordered doc_idx lists (each is a ranked result).
    k : RRF constant (default 60).

    Returns
    -------
    List of doc indices sorted by RRF score descending.
    """
    rrf_scores: dict[int, float] = {}

    for ranked_list in rankings:
        for rank, doc_idx in enumerate(ranked_list, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    return sorted_docs


# ---------------------------------------------------------------------------
# Dense retriever (lightweight, no AureliusTransformer dependency)
# ---------------------------------------------------------------------------

class DenseRetriever(nn.Module):
    """Lightweight dense retriever using a learnable projection matrix.

    Encodes texts as mean-of-character-codes projected through a weight matrix.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (default 64).
    """

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.W = nn.Parameter(torch.eye(embed_dim))

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of texts into embeddings of shape (N, embed_dim).

        Each text is represented as the mean of its character ordinals,
        broadcast to embed_dim and projected by W.
        """
        vectors = []
        for text in texts:
            if text:
                scalar = torch.tensor(
                    [ord(c) for c in text], dtype=torch.float
                ).mean()
            else:
                scalar = torch.zeros(1, dtype=torch.float).squeeze()
            vec = scalar.expand(self.embed_dim).clone()  # (embed_dim,)
            projected = vec @ self.W  # (embed_dim,)
            vectors.append(projected)

        return torch.stack(vectors)  # (N, embed_dim)

    def search(
        self, query: str, doc_embeddings: torch.Tensor, top_k: int
    ) -> list[tuple[int, float]]:
        """Dot-product similarity search.

        Parameters
        ----------
        query : query string.
        doc_embeddings : (N, embed_dim) tensor of document embeddings.
        top_k : number of results.

        Returns
        -------
        List of (doc_idx, score) tuples sorted by score descending.
        """
        query_emb = self.encode([query])[0]  # (embed_dim,)
        scores = doc_embeddings @ query_emb  # (N,)
        k = min(top_k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k)
        return list(zip(top_indices.tolist(), top_scores.tolist()))


# ---------------------------------------------------------------------------
# Full RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Full RAG pipeline: chunk, index, retrieve via sparse+dense fusion, format context.

    Parameters
    ----------
    config : RAGConfig
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._chunks: list[str] = []
        self._bm25 = BM25Index()
        self._dense = DenseRetriever(embed_dim=64)
        self._doc_embeddings: torch.Tensor | None = None

    def index_documents(self, documents: list[str]) -> None:
        """Chunk each document, then index with BM25 and DenseRetriever.

        All chunks across all documents are pooled into a single flat list.
        """
        all_chunks: list[str] = []
        for doc in documents:
            chunks = chunk_text(doc, self.config.chunk_size, self.config.chunk_overlap)
            all_chunks.extend(chunks)

        self._chunks = all_chunks

        if all_chunks:
            self._bm25.index(all_chunks)
            with torch.no_grad():
                self._doc_embeddings = self._dense.encode(all_chunks)  # (N, 64)
        else:
            self._doc_embeddings = None

    def retrieve(self, query: str) -> list[str]:
        """Retrieve top chunks for a query using BM25 + dense RRF fusion.

        Steps:
        1. BM25 search -> top n_retrieve chunk indices.
        2. Dense search -> top n_retrieve chunk indices.
        3. RRF fusion -> top n_rerank chunk indices.
        4. Return corresponding chunk strings.
        """
        if not self._chunks or self._doc_embeddings is None:
            return []

        cfg = self.config

        # BM25 ranked list (doc indices)
        bm25_results = self._bm25.search(query, cfg.n_retrieve)
        bm25_ranking = [idx for idx, _ in bm25_results]

        # Dense ranked list (doc indices)
        dense_results = self._dense.search(query, self._doc_embeddings, cfg.n_retrieve)
        dense_ranking = [idx for idx, _ in dense_results]

        # RRF fusion
        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
        top_indices = fused[: cfg.n_rerank]

        return [self._chunks[i] for i in top_indices]

    def format_context(self, query: str, chunks: list[str]) -> str:
        """Format retrieved chunks and query into a context string.

        Format:
            Context:
            {chunk1}
            {chunk2}
            ...
            Query: {query}
        """
        chunks_text = "\n".join(chunks)
        return f"Context:\n{chunks_text}\nQuery: {query}"
