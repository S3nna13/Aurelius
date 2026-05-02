"""Dense reranking for RAG pipelines.

Implements cross-encoder reranking, reciprocal rank fusion (RRF),
and score normalization for retrieved document sets.

References:
    Nogueira & Cho 2019 (monoBERT reranking)
    Cormack et al. 2009 (Reciprocal Rank Fusion)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A retrieved document with its dense embedding and optional token ids."""

    doc_id: int
    embedding: Tensor  # (d_model,)
    text_tokens: Tensor | None = None  # (T,) token ids, may be None
    score: float = 0.0  # retrieval score


# ---------------------------------------------------------------------------
# CrossEncoderScorer
# ---------------------------------------------------------------------------


class CrossEncoderScorer(nn.Module):
    """Scores (query, document) pairs jointly via a small MLP.

    Architecture:
        Linear(2*d_model, hidden_size) -> GELU -> Linear(hidden_size, 1)
    """

    def __init__(self, d_model: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def score(self, query_emb: Tensor, doc_embs: Tensor) -> Tensor:
        """Compute relevance scores for a query against N documents.

        Args:
            query_emb: (d_model,) query embedding.
            doc_embs:  (N, d_model) document embeddings.

        Returns:
            (N,) relevance scores.
        """
        # Expand query to (N, d_model) and concatenate with docs -> (N, 2*d_model)
        n = doc_embs.shape[0]
        q_expanded = query_emb.unsqueeze(0).expand(n, -1)  # (N, d_model)
        pairs = torch.cat([q_expanded, doc_embs], dim=-1)  # (N, 2*d_model)
        return self.mlp(pairs).squeeze(-1)  # (N,)


# ---------------------------------------------------------------------------
# BiEncoderRetriever
# ---------------------------------------------------------------------------


class BiEncoderRetriever:
    """Fast retrieval via dot-product similarity over a dense corpus.

    Args:
        doc_embeddings: (N_docs, d_model) corpus of pre-computed embeddings.
        doc_ids:        list of length N_docs, integer id for each document.
    """

    def __init__(self, doc_embeddings: Tensor, doc_ids: list[int]) -> None:
        self.doc_embeddings = doc_embeddings.float()  # (N_docs, d_model)
        self.doc_ids = doc_ids

    def retrieve(self, query_emb: Tensor, top_k: int = 10) -> list[Document]:
        """Return top_k Documents sorted by dot-product score descending.

        Args:
            query_emb: (d_model,) query embedding.
            top_k:     number of documents to return.

        Returns:
            List of up to top_k Documents with .score set.
        """
        q = query_emb.float()  # (d_model,)
        scores = self.doc_embeddings @ q  # (N_docs,)

        actual_k = min(top_k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, actual_k)

        results: list[Document] = []
        for i in range(actual_k):
            idx = int(top_indices[i].item())
            doc = Document(
                doc_id=self.doc_ids[idx],
                embedding=self.doc_embeddings[idx],
                score=float(top_scores[i].item()),
            )
            results.append(doc)

        return results


# ---------------------------------------------------------------------------
# RecipRankFusion
# ---------------------------------------------------------------------------


class RecipRankFusion:
    """Combines multiple ranked lists via Reciprocal Rank Fusion (RRF).

    RRF score for document d:
        score(d) = sum_r  1 / (k + rank_r(d))
    where rank is 1-indexed and the sum is over all ranked lists r.

    Reference:
        Cormack, Clarke & Buettcher 2009, SIGIR.
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def _compute_rrf_scores(self, ranked_lists: list[list[int]]) -> dict[int, float]:
        rrf_scores: dict[int, float] = {}
        for ranking in ranked_lists:
            for rank_idx, doc_id in enumerate(ranking):
                rank = rank_idx + 1  # 1-indexed
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        return rrf_scores

    def fuse(self, ranked_lists: list[list[int]]) -> list[int]:
        """Fuse ranked lists and return doc ids sorted by RRF score descending.

        Args:
            ranked_lists: list of ranked doc_id lists (earlier = higher rank).

        Returns:
            doc ids sorted by descending RRF score.
        """
        rrf_scores = self._compute_rrf_scores(ranked_lists)
        return sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    def fuse_with_scores(self, ranked_lists: list[list[int]]) -> list[tuple[int, float]]:
        """Fuse ranked lists and return (doc_id, rrf_score) tuples.

        Returns:
            List of (doc_id, rrf_score) sorted by score descending.
        """
        rrf_scores = self._compute_rrf_scores(ranked_lists)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# RAGReranker
# ---------------------------------------------------------------------------


class RAGReranker:
    """Full two-stage RAG pipeline: bi-encoder retrieval + cross-encoder reranking.

    Args:
        retriever:        BiEncoderRetriever for fast first-stage retrieval.
        scorer:           CrossEncoderScorer for expensive second-stage scoring.
        top_k_retrieve:   number of documents to pull from bi-encoder.
        top_k_rerank:     number of documents to return after reranking.
    """

    def __init__(
        self,
        retriever: BiEncoderRetriever,
        scorer: CrossEncoderScorer,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
    ) -> None:
        self.retriever = retriever
        self.scorer = scorer
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

    def retrieve_and_rerank(self, query_emb: Tensor) -> list[Document]:
        """Two-stage retrieval: bi-encoder then cross-encoder reranking.

        Step 1: Retrieve top_k_retrieve documents via bi-encoder dot product.
        Step 2: Score all retrieved documents via cross-encoder MLP.
        Step 3: Sort by cross-encoder score, return top top_k_rerank.

        Args:
            query_emb: (d_model,) query embedding.

        Returns:
            Up to top_k_rerank Documents sorted by cross-encoder score descending.
        """
        # Stage 1 — fast bi-encoder retrieval
        candidates: list[Document] = self.retriever.retrieve(query_emb, top_k=self.top_k_retrieve)
        if not candidates:
            return []

        # Stage 2 — cross-encoder scoring
        scores = self.score_documents(query_emb, candidates)  # (N,)

        # Sort and truncate
        scored_pairs = sorted(
            zip(scores.tolist(), candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = []
        for new_score, doc in scored_pairs[: self.top_k_rerank]:
            doc.score = float(new_score)
            reranked.append(doc)

        return reranked

    def score_documents(self, query_emb: Tensor, documents: list[Document]) -> Tensor:
        """Score a list of documents against a query.

        Args:
            query_emb: (d_model,) query embedding.
            documents: list of Documents to score.

        Returns:
            (len(documents),) float tensor of scores.
        """
        if not documents:
            return torch.zeros(0)

        doc_embs = torch.stack([doc.embedding.float() for doc in documents])  # (N, d_model)
        with torch.no_grad():
            scores = self.scorer.score(query_emb.float(), doc_embs)  # (N,)
        return scores
