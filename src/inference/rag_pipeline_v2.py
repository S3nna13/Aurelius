"""RAG pipeline v2: EmbeddingIndex-based dense retrieval with pluggable encoder/generator.

Implements:
  - RAGConfig
  - Document
  - EmbeddingIndex
  - build_rag_prompt
  - score_document_relevance
  - deduplicate_docs
  - RAGPipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    n_docs: int = 5
    max_doc_len: int = 256
    query_prefix: str = "Query: "
    doc_prefix: str = "Document: "
    rerank: bool = False
    deduplicate: bool = True


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    text: str
    embedding: Optional[Tensor] = field(default=None)
    score: float = 0.0


# ---------------------------------------------------------------------------
# EmbeddingIndex
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """In-memory dense embedding index using cosine similarity search."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._doc_ids: List[str] = []
        self._texts: List[str] = []
        self._embeddings: List[Tensor] = []

    def add(self, doc_id: str, text: str, embedding: Tensor) -> None:
        """Add a document embedding to the index."""
        self._doc_ids.append(doc_id)
        self._texts.append(text)
        self._embeddings.append(embedding.detach().float())

    def search(self, query_embedding: Tensor, k: int) -> List[Document]:
        """Cosine similarity search; return top-k Documents sorted by score desc."""
        if not self._embeddings:
            return []

        q = query_embedding.detach().float()
        # Stack all embeddings: (N, dim)
        doc_matrix = torch.stack(self._embeddings)  # (N, dim)

        # Cosine similarity: normalize both
        q_norm = F.normalize(q.unsqueeze(0), dim=-1)          # (1, dim)
        d_norm = F.normalize(doc_matrix, dim=-1)               # (N, dim)
        similarities = (d_norm @ q_norm.T).squeeze(-1)         # (N,)

        actual_k = min(k, similarities.shape[0])
        top_scores, top_indices = torch.topk(similarities, actual_k)

        results: List[Document] = []
        for i in range(actual_k):
            idx = top_indices[i].item()
            score = float(top_scores[i].item())
            doc = Document(
                doc_id=self._doc_ids[idx],
                text=self._texts[idx],
                embedding=self._embeddings[idx],
                score=score,
            )
            results.append(doc)

        return results

    def __len__(self) -> int:
        return len(self._doc_ids)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_rag_prompt(query: str, docs: List[Document], config: RAGConfig) -> str:
    """Format retrieved docs and query into a prompt string.

    Each document is prefixed with config.doc_prefix; the query follows at the
    end prefixed with config.query_prefix.
    """
    parts: List[str] = []
    for doc in docs:
        text = doc.text[: config.max_doc_len] if len(doc.text) > config.max_doc_len else doc.text
        parts.append(f"{config.doc_prefix}{text}")
    parts.append(f"{config.query_prefix}{query}")
    return "\n".join(parts)


def score_document_relevance(query_embedding: Tensor, doc_embedding: Tensor) -> float:
    """Cosine similarity between query and document embeddings, returned as float."""
    q = query_embedding.detach().float()
    d = doc_embedding.detach().float()
    q_norm = F.normalize(q.unsqueeze(0), dim=-1)  # (1, dim)
    d_norm = F.normalize(d.unsqueeze(0), dim=-1)  # (1, dim)
    sim = (q_norm * d_norm).sum(dim=-1)           # (1,)
    return float(sim.item())


def deduplicate_docs(docs: List[Document], threshold: float = 0.95) -> List[Document]:
    """Remove near-duplicate documents based on embedding cosine similarity.

    Among a cluster of near-duplicates (cosine similarity >= threshold), keep
    the one with the highest score. Documents without embeddings are always kept.
    """
    if not docs:
        return []

    # Sort by score descending so we always keep the highest-scored of a cluster
    sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)

    kept: List[Document] = []
    for candidate in sorted_docs:
        if candidate.embedding is None:
            kept.append(candidate)
            continue

        is_duplicate = False
        for existing in kept:
            if existing.embedding is None:
                continue
            sim = score_document_relevance(candidate.embedding, existing.embedding)
            if sim >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Retrieval-augmented generation pipeline.

    Parameters
    ----------
    encoder_fn : Callable[[str], Tensor]
        Maps a text string to a raw embedding tensor.
    generator_fn : Callable[[str], str]
        Maps a prompt string to a generated response string.
    index : EmbeddingIndex
        Pre-populated document index.
    config : RAGConfig
    """

    def __init__(
        self,
        encoder_fn: Callable[[str], Tensor],
        generator_fn: Callable[[str], str],
        index: EmbeddingIndex,
        config: RAGConfig,
    ) -> None:
        self.encoder_fn = encoder_fn
        self.generator_fn = generator_fn
        self.index = index
        self.config = config

    def encode_query(self, query: str) -> Tensor:
        """Encode query text and L2-normalize the resulting embedding."""
        raw = self.encoder_fn(query).float()
        return F.normalize(raw.unsqueeze(0), dim=-1).squeeze(0)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Encode query, search index, optionally deduplicate results."""
        if k is None:
            k = self.config.n_docs

        query_emb = self.encode_query(query)
        docs = self.index.search(query_emb, k)

        if self.config.deduplicate:
            docs = deduplicate_docs(docs)

        return docs

    def generate(self, query: str) -> Tuple[str, List[Document]]:
        """Retrieve documents, build prompt, generate response.

        Returns
        -------
        (response, retrieved_docs)
        """
        docs = self.retrieve(query)
        prompt = build_rag_prompt(query, docs, self.config)
        response = self.generator_fn(prompt)
        return response, docs

    def get_retrieval_stats(self, docs: List[Document]) -> Dict[str, float]:
        """Compute retrieval statistics over a list of Documents.

        Returns dict with keys: n_docs, mean_score, max_score, min_score.
        """
        n = float(len(docs))
        if docs:
            scores = [d.score for d in docs]
            mean_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            mean_score = 0.0
            max_score = 0.0
            min_score = 0.0

        return {
            "n_docs": n,
            "mean_score": mean_score,
            "max_score": max_score,
            "min_score": min_score,
        }
