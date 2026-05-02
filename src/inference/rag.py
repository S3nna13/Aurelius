"""Simple in-memory RAG: VectorStore + RAGRetriever using existing EmbeddingExtractor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.inference.embeddings import EmbeddingConfig, EmbeddingExtractor


@dataclass
class RetrievalResult:
    texts: list[str]
    scores: list[float]
    indices: list[int]


class VectorStore:
    """In-memory vector store for text + embedding pairs.

    Flat cosine similarity search — no external dependencies.
    """

    def __init__(self) -> None:
        self._embeddings: list[torch.Tensor] = []  # each (D,) L2-normalized
        self._texts: list[str] = []

    def add(self, embedding: torch.Tensor, text: str) -> None:
        """Add a single embedding+text pair. Embedding should be L2-normalized (D,)."""
        self._embeddings.append(embedding.detach().cpu())
        self._texts.append(text)

    def add_batch(self, embeddings: torch.Tensor, texts: list[str]) -> None:
        """Add a batch of embeddings. embeddings: (N, D)."""
        for i, text in enumerate(texts):
            self.add(embeddings[i], text)

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> RetrievalResult:
        """Find top-k most similar texts to query.

        Args:
            query_embedding: (D,) L2-normalized query embedding
            top_k: number of results

        Returns:
            RetrievalResult with texts, scores, indices sorted by score descending.
        """
        if not self._embeddings:
            return RetrievalResult([], [], [])

        corpus = torch.stack(self._embeddings)  # (N, D)
        query = F.normalize(query_embedding.float().cpu(), dim=-1)
        corpus_norm = F.normalize(corpus.float(), dim=-1)

        scores = (corpus_norm @ query).squeeze()  # (N,)
        k = min(top_k, len(self._embeddings))
        top_scores, top_indices = torch.topk(scores, k)

        return RetrievalResult(
            texts=[self._texts[i] for i in top_indices.tolist()],
            scores=top_scores.tolist(),
            indices=top_indices.tolist(),
        )

    def __len__(self) -> int:
        return len(self._texts)

    def clear(self) -> None:
        self._embeddings.clear()
        self._texts.clear()


class RAGRetriever:
    """End-to-end retriever: encodes query -> searches VectorStore -> returns context.

    Usage:
        retriever = RAGRetriever(model, store)
        results = retriever.retrieve(input_ids, top_k=3)
        context = "\\n".join(results.texts)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        store: VectorStore,
        embedding_cfg: EmbeddingConfig | None = None,
    ) -> None:
        self.extractor = EmbeddingExtractor(model, embedding_cfg or EmbeddingConfig())
        self.store = store

    def retrieve(
        self,
        input_ids: torch.Tensor,
        top_k: int = 5,
        attention_mask: torch.Tensor | None = None,
    ) -> RetrievalResult:
        """Encode input_ids and retrieve top-k relevant texts.

        Args:
            input_ids: (1, S) or (B, S) query token IDs
            top_k: number of results per query (returns first query's results if batch)
        """
        emb = self.extractor.encode(input_ids, attention_mask)  # (B, D)
        query = emb[0]  # (D,) — use first sequence as query
        return self.store.search(query, top_k=top_k)

    def index(
        self,
        input_ids: torch.Tensor,
        texts: list[str],
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Encode input_ids and add to store with corresponding texts."""
        embs = self.extractor.encode(input_ids, attention_mask)  # (B, D)
        self.store.add_batch(embs, texts)
