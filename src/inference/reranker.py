"""Cross-encoder re-ranker for retrieval-augmented generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RerankConfig:
    max_length: int = 256
    batch_size: int = 8
    score_aggregation: str = "max"  # "max" | "mean" | "first"


class CrossEncoderScorer(nn.Module):
    """Cross-encoder that jointly scores a query and document for relevance."""

    def __init__(self, d_model: int, vocab_size: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, query_ids: Tensor, doc_ids: Tensor) -> Tensor:
        """Score query-document pairs.

        Args:
            query_ids: (B, T_q) token ids
            doc_ids:   (B, T_d) token ids

        Returns:
            (B, 1) relevance scores
        """
        combined = torch.cat([query_ids, doc_ids], dim=1)  # (B, T_q+T_d)
        emb = self.embedding(combined)                      # (B, T_q+T_d, d_model)
        pooled = emb.mean(dim=1)                            # (B, d_model)
        encoded = self.encoder(pooled)                      # (B, d_model)
        return self.scorer(encoded)                         # (B, 1)


class DocumentReranker:
    """Reranks a list of documents against a query using a CrossEncoderScorer."""

    def __init__(self, scorer: CrossEncoderScorer, config: RerankConfig) -> None:
        self.scorer = scorer
        self.config = config

    def score_documents(self, query_ids: Tensor, doc_ids_list: list[Tensor]) -> Tensor:
        """Score each document against the query.

        Args:
            query_ids:    (1, T_q) query token ids
            doc_ids_list: list of n_docs tensors each (1, T_d)

        Returns:
            (n_docs,) relevance scores
        """
        all_scores: list[Tensor] = []
        batch_size = self.config.batch_size

        for start in range(0, len(doc_ids_list), batch_size):
            batch_docs = doc_ids_list[start : start + batch_size]
            # Pad docs in the batch to the same length
            max_len = max(d.shape[1] for d in batch_docs)
            padded = [
                F.pad(d, (0, max_len - d.shape[1])) for d in batch_docs
            ]
            doc_batch = torch.cat(padded, dim=0)  # (batch, max_len)
            query_batch = query_ids.expand(doc_batch.shape[0], -1)  # (batch, T_q)

            with torch.no_grad():
                scores = self.scorer(query_batch, doc_batch)  # (batch, 1)
            all_scores.append(scores.squeeze(-1))  # (batch,)

        return torch.cat(all_scores, dim=0)  # (n_docs,)

    def rerank(
        self,
        query_ids: Tensor,
        doc_ids_list: list[Tensor],
        top_k: int | None = None,
    ) -> tuple[list[Tensor], Tensor]:
        """Score and sort documents by relevance descending.

        Returns:
            (sorted_docs, sorted_scores) optionally trimmed to top_k
        """
        scores = self.score_documents(query_ids, doc_ids_list)  # (n_docs,)
        order = torch.argsort(scores, descending=True)

        if top_k is not None:
            order = order[:top_k]

        sorted_docs = [doc_ids_list[i] for i in order.tolist()]
        sorted_scores = scores[order]
        return sorted_docs, sorted_scores

    def rerank_texts(
        self,
        query: str,
        documents: list[str],
        tokenize_fn: Callable[[str], list[int]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Tokenize, rerank, and return (doc_text, score) pairs sorted by relevance.

        Args:
            query:        query string
            documents:    list of document strings
            tokenize_fn:  function mapping str -> list[int]
            top_k:        optional limit on returned results

        Returns:
            list of (doc_text, score) tuples sorted by score descending
        """
        query_ids = torch.tensor(tokenize_fn(query), dtype=torch.long).unsqueeze(0)
        doc_ids_list = [
            torch.tensor(tokenize_fn(doc), dtype=torch.long).unsqueeze(0)
            for doc in documents
        ]

        sorted_docs, sorted_scores = self.rerank(query_ids, doc_ids_list, top_k=top_k)

        # Map sorted doc tensors back to original text
        # Build index from tensor data_ptr for lookup
        tensor_to_text = {
            id(doc_ids_list[i]): documents[i] for i in range(len(documents))
        }

        results: list[tuple[str, float]] = []
        for doc_tensor, score in zip(sorted_docs, sorted_scores.tolist()):
            text = tensor_to_text[id(doc_tensor)]
            results.append((text, float(score)))

        return results


def listwise_rerank_loss(scores: Tensor, relevance_labels: Tensor) -> Tensor:
    """ListNet listwise ranking loss: KL divergence between label and score distributions.

    Args:
        scores:           (n_docs,) predicted relevance scores
        relevance_labels: (n_docs,) ground-truth relevance (0/1 or continuous)

    Returns:
        scalar KL divergence loss
    """
    label_probs = F.softmax(relevance_labels.float(), dim=0)
    score_log_probs = F.log_softmax(scores.float(), dim=0)
    return F.kl_div(score_log_probs, label_probs, reduction="sum")


def pairwise_rerank_loss(pos_scores: Tensor, neg_scores: Tensor, margin: float = 1.0) -> Tensor:
    """Pairwise margin ranking loss.

    Args:
        pos_scores: (B,) scores for positive (relevant) documents
        neg_scores: (B,) scores for negative (irrelevant) documents
        margin:     minimum required score gap

    Returns:
        scalar mean loss
    """
    return torch.clamp(margin - (pos_scores - neg_scores), min=0.0).mean()


class RerankTrainer:
    """Trains a CrossEncoderScorer with pairwise ranking loss."""

    def __init__(
        self,
        scorer: CrossEncoderScorer,
        optimizer: torch.optim.Optimizer,
        config: RerankConfig,
    ) -> None:
        self.scorer = scorer
        self.optimizer = optimizer
        self.config = config

    def train_step(
        self,
        query_ids: Tensor,
        pos_doc_ids: Tensor,
        neg_doc_ids: Tensor,
    ) -> dict[str, float]:
        """One training step: score pos/neg docs, compute pairwise loss, update.

        Args:
            query_ids:   (1, T_q)
            pos_doc_ids: (1, T_d) positive document token ids
            neg_doc_ids: (1, T_d) negative document token ids

        Returns:
            {"loss": float, "pos_score": float, "neg_score": float}
        """
        self.scorer.train()
        self.optimizer.zero_grad()

        pos_score = self.scorer(query_ids, pos_doc_ids)  # (1, 1)
        neg_score = self.scorer(query_ids, neg_doc_ids)  # (1, 1)

        loss = pairwise_rerank_loss(pos_score.squeeze(), neg_score.squeeze())
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "pos_score": pos_score.item(),
            "neg_score": neg_score.item(),
        }
