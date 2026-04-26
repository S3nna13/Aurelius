"""BERTScore-style token-level embedding similarity metric.

Uses token embeddings from any model's embedding table as representations.
Computes soft token-level overlap between candidate and reference sequences.
No external dependencies — pure PyTorch.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between all pairs of vectors.

    Args:
        a: (M, d)
        b: (N, d)

    Returns:
        (M, N) similarity matrix
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return torch.matmul(a_norm, b_norm.t())


def bert_precision(
    candidate_embeddings: torch.Tensor,  # (C, d)
    reference_embeddings: torch.Tensor,  # (R, d)
) -> float:
    """
    For each candidate token, find its max similarity to any reference token.
    Precision = mean of max similarities.
    """
    sim = cosine_similarity_matrix(candidate_embeddings, reference_embeddings)  # (C, R)
    max_sims = sim.max(dim=-1).values  # (C,)
    return float(max_sims.mean().item())


def bert_recall(
    candidate_embeddings: torch.Tensor,  # (C, d)
    reference_embeddings: torch.Tensor,  # (R, d)
) -> float:
    """
    For each reference token, find its max similarity to any candidate token.
    Recall = mean of max similarities.
    """
    sim = cosine_similarity_matrix(candidate_embeddings, reference_embeddings)  # (C, R)
    max_sims = sim.max(dim=0).values  # (R,)
    return float(max_sims.mean().item())


def bert_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall. 0 if both are 0."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class EmbeddingBERTScore:
    """
    BERTScore-style evaluation using any embedding model.

    Instead of BERT, uses the model's token embeddings as representations.
    Computes token-level soft overlap between candidate and reference.

    Args:
        embeddings: (vocab_size, d_model) embedding matrix (e.g. model.embed.weight)
        idf_weights: optional (vocab_size,) IDF weights for weighted scoring
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        idf_weights: torch.Tensor | None = None,
    ):
        # L2-normalize embeddings for cosine similarity
        self.embeddings = F.normalize(embeddings.float(), dim=-1)
        self.idf_weights = idf_weights

    def encode(self, token_ids: list[int]) -> torch.Tensor:
        """Look up normalized embeddings for token IDs. Returns (N, d)."""
        idx = torch.tensor(token_ids, dtype=torch.long)
        return self.embeddings[idx]

    def score(
        self,
        candidate_ids: list[int],
        reference_ids: list[int],
    ) -> dict:
        """
        Compute BERTScore between candidate and reference.

        Returns: {'precision': float, 'recall': float, 'f1': float}
        """
        cand_emb = self.encode(candidate_ids)  # (C, d)
        ref_emb = self.encode(reference_ids)  # (R, d)

        if self.idf_weights is not None:
            cand_idf = self.idf_weights[torch.tensor(candidate_ids, dtype=torch.long)]
            ref_idf = self.idf_weights[torch.tensor(reference_ids, dtype=torch.long)]
        else:
            cand_idf = None
            ref_idf = None

        sim = cosine_similarity_matrix(cand_emb, ref_emb)  # (C, R)

        # Precision: for each candidate token, max sim to any reference token
        p_sims = sim.max(dim=-1).values  # (C,)
        if cand_idf is not None:
            precision = float((p_sims * cand_idf).sum() / cand_idf.sum().clamp(min=1e-8))
        else:
            precision = float(p_sims.mean().item())

        # Recall: for each reference token, max sim to any candidate token
        r_sims = sim.max(dim=0).values  # (R,)
        if ref_idf is not None:
            recall = float((r_sims * ref_idf).sum() / ref_idf.sum().clamp(min=1e-8))
        else:
            recall = float(r_sims.mean().item())

        f1 = bert_f1(precision, recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    def batch_score(
        self,
        candidates: list[list[int]],
        references: list[list[int]],
    ) -> list[dict]:
        """Score multiple (candidate, reference) pairs."""
        return [self.score(cand, ref) for cand, ref in zip(candidates, references)]

    def corpus_score(
        self,
        candidates: list[list[int]],
        references: list[list[int]],
    ) -> dict:
        """Mean P, R, F1 across all pairs."""
        results = self.batch_score(candidates, references)
        n = len(results)
        if n == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        mean_p = sum(r["precision"] for r in results) / n
        mean_r = sum(r["recall"] for r in results) / n
        mean_f1 = sum(r["f1"] for r in results) / n

        return {"precision": mean_p, "recall": mean_r, "f1": mean_f1}


def compute_idf_weights(corpus: list[list[int]], vocab_size: int) -> torch.Tensor:
    """
    Compute Inverse Document Frequency weights.

    idf(t) = log((N+1) / (df(t)+1))

    N: number of documents
    df(t): number of documents containing token t

    Returns: (vocab_size,) tensor
    """
    n = len(corpus)
    doc_freq = torch.zeros(vocab_size, dtype=torch.float)

    for doc in corpus:
        unique_tokens = set(doc)
        for token_id in unique_tokens:
            if 0 <= token_id < vocab_size:
                doc_freq[token_id] += 1.0

    idf = torch.log((n + 1.0) / (doc_freq + 1.0))
    return idf
