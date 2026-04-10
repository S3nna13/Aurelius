"""Full RAG pipeline: chunking, BM25-style sparse retrieval, dense fusion, and generation.

Extended with dense DocumentStore, QueryEncoder, augmented input builder, generator-integrated
RAGPipeline, and RAGTrainer for fine-tuning on RAG-augmented inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


# ---------------------------------------------------------------------------
# Dense RAG: DocumentStore, QueryEncoder, augmented input, generator pipeline
# ---------------------------------------------------------------------------

@dataclass
class DenseRAGConfig:
    """Configuration for the dense-retrieval RAG pipeline."""

    n_docs: int = 3                 # documents to retrieve per query
    max_doc_len: int = 64           # max tokens per document
    max_answer_len: int = 32
    score_method: str = "dot"       # "dot" | "cosine"
    prepend_docs: bool = True       # prepend retrieved docs to query


class DocumentStore:
    """In-memory document store with dense embeddings."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim
        self._doc_ids: list[Tensor] = []
        self._embeddings: list[Tensor] = []

    def add(self, doc_ids: Tensor, embedding: Tensor) -> None:
        """Add a document: doc_ids (T,) and its dense embedding (embed_dim,)."""
        self._doc_ids.append(doc_ids.detach())
        self._embeddings.append(embedding.detach())

    def add_batch(self, doc_ids_list: list[Tensor], embeddings: Tensor) -> None:
        """Add multiple docs. embeddings shape (n_docs, embed_dim)."""
        assert embeddings.shape[0] == len(doc_ids_list), (
            "Number of embeddings must match number of doc_ids entries."
        )
        for i, doc_ids in enumerate(doc_ids_list):
            self.add(doc_ids, embeddings[i])

    def retrieve(
        self,
        query_emb: Tensor,
        n_docs: int,
        method: str = "dot",
    ) -> list[tuple[Tensor, float]]:
        """Retrieve top-n_docs documents by query_emb similarity.

        Returns list of (doc_ids, score) sorted by score descending.
        """
        if not self._embeddings:
            return []

        doc_matrix = torch.stack(self._embeddings)  # (N, embed_dim)
        q = query_emb.detach().float()

        if method == "cosine":
            q_norm = F.normalize(q.unsqueeze(0), dim=-1)              # (1, D)
            d_norm = F.normalize(doc_matrix.float(), dim=-1)          # (N, D)
            scores = (d_norm @ q_norm.T).squeeze(-1)                  # (N,)
        else:  # dot
            scores = doc_matrix.float() @ q.float()                   # (N,)

        k = min(n_docs, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k)

        results: list[tuple[Tensor, float]] = []
        for i in range(k):
            idx = top_indices[i].item()
            score = top_scores[i].item()
            results.append((self._doc_ids[idx], float(score)))

        return results

    def __len__(self) -> int:
        return len(self._doc_ids)


class QueryEncoder(nn.Module):
    """Encode token ids into a dense query embedding."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        """input_ids (B, T) -> embeddings (B, embed_dim) — mean-pool over T."""
        # (B, T, embed_dim) -> mean over T -> (B, embed_dim)
        token_embeds = self.embed(input_ids)        # (B, T, D)
        return token_embeds.mean(dim=1)             # (B, D)


def build_augmented_input(
    query_ids: Tensor,
    doc_ids_list: list[Tensor],
    max_total_len: int,
) -> Tensor:
    """Concatenate retrieved docs before query, truncated to max_total_len.

    Concatenation order: [doc1_ids, doc2_ids, ..., query_ids].
    When total length exceeds max_total_len, truncate from the LEFT
    (keep the end / query part).

    Returns (max_total_len,) padded with 0s on the left if shorter.
    """
    parts: list[Tensor] = []
    for doc_ids in doc_ids_list:
        parts.append(doc_ids.flatten())
    parts.append(query_ids.flatten())

    if parts:
        combined = torch.cat(parts, dim=0)
    else:
        combined = query_ids.flatten()

    # Truncate from left — keep tail (query end)
    if combined.shape[0] > max_total_len:
        combined = combined[-max_total_len:]

    # Pad left with zeros if shorter
    pad_len = max_total_len - combined.shape[0]
    if pad_len > 0:
        combined = torch.cat(
            [torch.zeros(pad_len, dtype=combined.dtype), combined], dim=0
        )

    return combined


class DenseRAGPipeline:
    """Full RAG pipeline: encode query -> retrieve docs -> augment -> generate.

    Uses a separate nn.Module generator (e.g. AureliusTransformer) whose
    forward signature must be:
        loss, logits, past_key_values = generator(input_ids, labels=labels)
    """

    def __init__(
        self,
        generator: nn.Module,
        query_encoder: QueryEncoder,
        doc_store: DocumentStore,
        config: DenseRAGConfig,
    ) -> None:
        self.generator = generator
        self.query_encoder = query_encoder
        self.doc_store = doc_store
        self.config = config

    def retrieve(self, query_ids: Tensor) -> list[tuple[Tensor, float]]:
        """Encode query and retrieve top-n_docs documents."""
        with torch.no_grad():
            q_emb = self.query_encoder(query_ids.unsqueeze(0))  # (1, D)
            q_emb = q_emb.squeeze(0)                            # (D,)
        return self.doc_store.retrieve(
            q_emb, n_docs=self.config.n_docs, method=self.config.score_method
        )

    def generate(self, query_ids: Tensor) -> tuple[Tensor, list[tuple[Tensor, float]]]:
        """Full RAG: retrieve docs, build augmented input, greedy decode one token.

        Returns (next_token_id (1,), retrieved_docs).
        """
        retrieved = self.retrieve(query_ids)
        doc_ids_list = [doc_ids for doc_ids, _ in retrieved]

        max_seq = self.generator.config.max_seq_len if hasattr(self.generator, "config") else 512
        max_total_len = max_seq

        aug_ids = build_augmented_input(query_ids, doc_ids_list, max_total_len)
        input_ids = aug_ids.unsqueeze(0)  # (1, max_total_len)

        with torch.no_grad():
            _, logits, _ = self.generator(input_ids)  # (1, T, V)

        next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)  # (1,)
        return next_token, retrieved

    def score_answer(self, query_ids: Tensor, answer_ids: Tensor) -> float:
        """Score answer relevance: mean log prob of answer_ids given augmented context.

        Builds augmented context, then computes log-softmax over logits at each
        answer position and averages.
        """
        retrieved = self.retrieve(query_ids)
        doc_ids_list = [d for d, _ in retrieved]

        max_seq = self.generator.config.max_seq_len if hasattr(self.generator, "config") else 512
        aug_ids = build_augmented_input(query_ids, doc_ids_list, max_seq)

        # Truncate aug to leave room for answer
        context_len = max_seq - answer_ids.shape[0]
        context_ids = aug_ids[:context_len]
        full_ids = torch.cat([context_ids, answer_ids.flatten()], dim=0).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            _, logits, _ = self.generator(full_ids)  # (1, T, V)

        log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V)

        # Compute log prob for each answer token
        ans_len = answer_ids.shape[0]
        # Answer tokens start at position context_len; we predict from context_len-1
        pred_start = context_len - 1
        ans_log_probs = []
        for i in range(ans_len):
            tok = answer_ids.flatten()[i].item()
            lp = log_probs[pred_start + i, tok].item()
            ans_log_probs.append(lp)

        return float(sum(ans_log_probs) / len(ans_log_probs)) if ans_log_probs else 0.0


class DenseRAGTrainer:
    """Fine-tune generator on RAG-augmented inputs."""

    def __init__(
        self,
        pipeline: DenseRAGPipeline,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.pipeline = pipeline
        self.optimizer = optimizer

    def train_step(self, query_ids: Tensor, answer_ids: Tensor) -> dict:
        """Retrieve docs, build augmented input, compute CE loss on answer tokens.

        Returns dict with: 'loss', 'n_docs_retrieved', 'mean_doc_score'.
        """
        # Retrieve (no grad needed for retrieval)
        retrieved = self.pipeline.retrieve(query_ids)
        n_docs = len(retrieved)
        mean_score = (
            float(sum(s for _, s in retrieved) / n_docs) if n_docs > 0 else 0.0
        )
        doc_ids_list = [d for d, _ in retrieved]

        cfg = self.pipeline.config
        generator = self.pipeline.generator
        max_seq = generator.config.max_seq_len if hasattr(generator, "config") else 512

        aug_ids = build_augmented_input(query_ids, doc_ids_list, max_seq)

        # Context: everything before answer; labels: answer tokens
        context_len = max_seq - answer_ids.shape[0]
        context_ids = aug_ids[:context_len]
        full_ids = torch.cat([context_ids, answer_ids.flatten()], dim=0).unsqueeze(0)  # (1, T)

        # Build labels: mask context positions with -100, supervise answer positions
        labels = full_ids.clone()
        labels[0, :context_len] = -100  # ignore context in loss

        self.optimizer.zero_grad()
        loss, _, _ = generator(full_ids, labels=labels)

        if loss is None:
            # If labels weren't accepted, compute manually
            _, logits, _ = generator(full_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_docs_retrieved": n_docs,
            "mean_doc_score": mean_score,
        }
