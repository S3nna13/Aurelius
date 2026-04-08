"""RAG Pipeline: higher-level orchestration layer for Retrieval-Augmented Generation.

Provides document ingestion (chunking + indexing), semantic retrieval (dense or
sparse BM25), and augmented generation — all without external dependencies beyond
PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    chunk_size: int = 256           # characters per chunk
    chunk_overlap: int = 64         # overlap between consecutive chunks
    top_k: int = 3                  # number of chunks to retrieve
    max_context_tokens: int = 512   # max tokens for retrieved context
    rerank: bool = False            # whether to rerank retrieved chunks


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str           # "{doc_id}_{chunk_idx}"
    doc_id: str
    text: str
    start_char: int
    end_char: int
    embedding: torch.Tensor | None = None


@dataclass
class RetrievalResult:
    query: str
    retrieved_chunks: list[Chunk]
    scores: list[float]     # similarity scores, descending


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

class DocumentChunker:
    """Split documents into overlapping chunks (character-based).

    Args:
        chunk_size: maximum characters per chunk.
        chunk_overlap: overlap in characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split doc.text into overlapping chunks.

        Tries to break at the last whitespace boundary within chunk_size;
        falls back to a hard cut at exactly chunk_size characters.
        """
        text = doc.text
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last (possibly short) chunk
                chunk_text = text[start:]
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=len(text),
                ))
                break
            else:
                # Try to find a whitespace boundary within [start, end]
                boundary = text.rfind(" ", start, end)
                if boundary == -1 or boundary <= start:
                    # No suitable space found; hard-cut at chunk_size
                    boundary = end

                chunk_text = text[start:boundary]
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=boundary,
                ))
                # Advance start by (chunk_size - overlap), but at least 1 to avoid infinite loop
                step = max(1, self.chunk_size - self.chunk_overlap)
                start += step
                idx += 1

        return chunks

    def chunk_corpus(self, documents: list[Document]) -> list[Chunk]:
        """Chunk all documents and return a flat list of all chunks."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retrieval using cosine similarity over chunk embeddings.

    Args:
        embed_fn: callable (text: str) -> Tensor(D)
        top_k: default number of chunks to retrieve.
    """

    def __init__(self, embed_fn: Callable[[str], torch.Tensor], top_k: int = 3) -> None:
        self.embed_fn = embed_fn
        self.top_k = top_k
        self._chunks: list[Chunk] = []
        self._embeddings: torch.Tensor | None = None  # (N, D)

    def index(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and store for retrieval."""
        self._chunks = list(chunks)
        if not chunks:
            self._embeddings = None
            return

        embs = []
        for chunk in chunks:
            emb = self.embed_fn(chunk.text)
            embs.append(emb.detach().cpu())

        self._embeddings = torch.stack(embs)  # (N, D)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Embed query and return top-k most similar chunks (cosine similarity)."""
        k = top_k if top_k is not None else self.top_k

        if not self._chunks or self._embeddings is None:
            return RetrievalResult(query=query, retrieved_chunks=[], scores=[])

        query_emb = self.embed_fn(query).detach().cpu().float()
        query_norm = F.normalize(query_emb, dim=-1)
        corpus_norm = F.normalize(self._embeddings.float(), dim=-1)  # (N, D)

        scores = corpus_norm @ query_norm  # (N,)
        k_actual = min(k, len(self._chunks))
        top_scores, top_indices = torch.topk(scores, k_actual)

        retrieved = [self._chunks[i] for i in top_indices.tolist()]
        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved,
            scores=top_scores.tolist(),
        )

    def update(self, new_chunks: list[Chunk]) -> None:
        """Append new chunks to the existing index."""
        if not new_chunks:
            return

        new_embs = []
        for chunk in new_chunks:
            emb = self.embed_fn(chunk.text)
            new_embs.append(emb.detach().cpu())

        new_tensor = torch.stack(new_embs)  # (M, D)
        self._chunks.extend(new_chunks)

        if self._embeddings is None:
            self._embeddings = new_tensor
        else:
            self._embeddings = torch.cat([self._embeddings, new_tensor], dim=0)


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Sparse BM25 retrieval (Robertson & Zaragoza 2009).

    BM25 score(d, q) = sum_t IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1 - b + b*dl/avgdl))

    Args:
        k1: term saturation parameter (default 1.5).
        b: length normalisation parameter (default 0.75).
        top_k: default number of chunks to retrieve.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, top_k: int = 3) -> None:
        self.k1 = k1
        self.b = b
        self.top_k = top_k
        self._chunks: list[Chunk] = []
        self._vocab: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []   # per-chunk term frequencies
        self._df: dict[str, int] = {}          # document (chunk) frequencies
        self._avgdl: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase whitespace tokenization."""
        return text.lower().split()

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = list(chunks)
        self._vocab = {}
        self._tf = []
        self._df = {}

        total_len = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            total_len += len(tokens)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)
            for tok in set(tokens):
                self._df[tok] = self._df.get(tok, 0) + 1

        self._avgdl = total_len / len(chunks) if chunks else 0.0

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """BM25 retrieval."""
        k = top_k if top_k is not None else self.top_k

        if not self._chunks:
            return RetrievalResult(query=query, retrieved_chunks=[], scores=[])

        query_tokens = self._tokenize(query)
        n = len(self._chunks)
        scores: list[float] = []

        for i, chunk in enumerate(self._chunks):
            tf_i = self._tf[i]
            dl = sum(tf_i.values())
            score = 0.0
            for tok in query_tokens:
                tf_tok = tf_i.get(tok, 0)
                df_tok = self._df.get(tok, 0)
                if df_tok == 0:
                    continue
                idf = math.log((n - df_tok + 0.5) / (df_tok + 0.5) + 1)
                numerator = tf_tok * (self.k1 + 1)
                denominator = tf_tok + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * numerator / denominator
            scores.append(score)

        k_actual = min(k, n)
        # argsort descending
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k_actual]
        retrieved = [self._chunks[i] for i in ranked]
        top_scores = [scores[i] for i in ranked]

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved,
            scores=top_scores,
        )


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Full RAG pipeline: chunk → index → retrieve → augment → generate.

    Args:
        model: AureliusTransformer instance.
        tokenizer_encode: callable (text: str) -> Tensor of token ids (1D).
        tokenizer_decode: callable (token_ids: Tensor) -> str.
        retriever: DenseRetriever or BM25Retriever.
        chunker: DocumentChunker.
        config: RAGConfig.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_encode: Callable[[str], torch.Tensor],
        tokenizer_decode: Callable[[torch.Tensor], str],
        retriever: DenseRetriever | BM25Retriever,
        chunker: DocumentChunker,
        config: RAGConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.retriever = retriever
        self.chunker = chunker
        self.config = config or RAGConfig()

    def ingest(self, documents: list[Document]) -> int:
        """Chunk and index documents. Returns total chunk count."""
        chunks = self.chunker.chunk_corpus(documents)
        self.retriever.index(chunks)
        return len(chunks)

    def query(self, question: str) -> dict:
        """Full RAG pipeline: retrieve → format → generate.

        Returns:
            dict with keys 'answer', 'retrieved_chunks', 'scores'.
        """
        # 1. Retrieve top-k chunks
        result = self.retriever.retrieve(question, top_k=self.config.top_k)

        # 2. Build context string, truncated to max_context_tokens
        #    Rough estimate: 1 token ≈ 4 characters
        max_chars = self.config.max_context_tokens * 4
        context_parts: list[str] = []
        used = 0
        for chunk in result.retrieved_chunks:
            if used + len(chunk.text) > max_chars:
                remaining = max_chars - used
                if remaining > 0:
                    context_parts.append(chunk.text[:remaining])
                break
            context_parts.append(chunk.text)
            used += len(chunk.text)

        context_str = "\n".join(context_parts)
        prompt = f"Context:\n{context_str}\nQuestion: {question}\nAnswer:"

        # 3. Tokenize and generate
        input_ids = self.tokenizer_encode(prompt)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (1, S)

        with torch.no_grad():
            _, logits, _ = self.model(input_ids)

        # Greedy next-token prediction (single step)
        next_token_id = logits[0, -1].argmax(dim=-1, keepdim=True)
        answer = self.tokenizer_decode(next_token_id)

        return {
            "answer": answer,
            "retrieved_chunks": result.retrieved_chunks,
            "scores": result.scores,
        }

    def batch_query(self, questions: list[str]) -> list[dict]:
        """Run query() for each question. Returns list of result dicts."""
        return [self.query(q) for q in questions]
