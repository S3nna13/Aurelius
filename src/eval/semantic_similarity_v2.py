"""Semantic similarity metrics for evaluating embedding quality and generation outputs.

Pure PyTorch implementation — no HuggingFace, no scipy, no sklearn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimilarityConfig:
    """Configuration for semantic similarity evaluation."""

    pooling: str = "mean"          # "mean" | "cls" | "max"
    normalize: bool = True
    similarity_metric: str = "cosine"  # "cosine" | "dot" | "l2"

    def __post_init__(self) -> None:
        valid_pooling = {"mean", "cls", "max"}
        if self.pooling not in valid_pooling:
            raise ValueError(
                f"pooling must be one of {valid_pooling}, got {self.pooling!r}"
            )
        valid_metrics = {"cosine", "dot", "l2"}
        if self.similarity_metric not in valid_metrics:
            raise ValueError(
                f"similarity_metric must be one of {valid_metrics}, "
                f"got {self.similarity_metric!r}"
            )


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool_hidden_states(
    hidden: Tensor,
    attention_mask: Optional[Tensor] = None,
    mode: str = "mean",
) -> Tensor:
    """Pool a sequence of hidden states to a single vector per example.

    Args:
        hidden: (B, T, D) hidden state tensor.
        attention_mask: Optional (B, T) boolean or 0/1 mask where 1/True
            marks *real* (non-padded) positions.  If None, all positions
            are treated as real.
        mode: Pooling strategy — "mean" (masked mean), "cls" (first token),
              or "max" (masked max).

    Returns:
        (B, D) pooled tensor.
    """
    B, T, D = hidden.shape

    if mode == "cls":
        return hidden[:, 0, :]

    if attention_mask is None:
        # All positions are real.
        if mode == "mean":
            return hidden.mean(dim=1)
        elif mode == "max":
            return hidden.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {mode!r}")

    # Convert mask to float (B, T, 1) for broadcasting.
    mask_f = attention_mask.float().unsqueeze(-1)  # (B, T, 1)

    if mode == "mean":
        # Masked mean: sum real tokens / count of real tokens.
        summed = (hidden * mask_f).sum(dim=1)             # (B, D)
        counts = mask_f.sum(dim=1).clamp(min=1.0)         # (B, 1)
        return summed / counts

    elif mode == "max":
        # Masked max: set padded positions to -inf before taking max.
        mask_bool = attention_mask.bool().unsqueeze(-1).expand_as(hidden)  # (B, T, D)
        hidden_masked = hidden.masked_fill(~mask_bool, float("-inf"))
        return hidden_masked.max(dim=1).values

    else:
        raise ValueError(f"Unknown pooling mode: {mode!r}")


# ---------------------------------------------------------------------------
# Pairwise cosine similarity
# ---------------------------------------------------------------------------

def compute_cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    """Compute element-wise cosine similarity between paired embeddings.

    Args:
        a: (B, D) tensor.
        b: (B, D) tensor.

    Returns:
        (B,) cosine similarity per pair.
    """
    a_norm = F.normalize(a, p=2, dim=-1)  # (B, D)
    b_norm = F.normalize(b, p=2, dim=-1)  # (B, D)
    return (a_norm * b_norm).sum(dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# Pairwise similarity matrix
# ---------------------------------------------------------------------------

def compute_pairwise_similarity(
    embeddings: Tensor,
    metric: str = "cosine",
) -> Tensor:
    """Compute an all-pairs similarity matrix.

    Args:
        embeddings: (N, D) embedding matrix.
        metric: "cosine", "dot", or "l2" (returns negative L2 distance).

    Returns:
        (N, N) similarity matrix.
    """
    if metric == "cosine":
        normed = F.normalize(embeddings, p=2, dim=-1)  # (N, D)
        return normed @ normed.T                        # (N, N)

    elif metric == "dot":
        return embeddings @ embeddings.T                # (N, N)

    elif metric == "l2":
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        sq_norm = (embeddings ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
        sq_dist = sq_norm + sq_norm.T - 2.0 * (embeddings @ embeddings.T)
        sq_dist = sq_dist.clamp(min=0.0)
        return -sq_dist.sqrt()  # negative L2 distance (N, N)

    else:
        raise ValueError(f"Unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# Alignment score
# ---------------------------------------------------------------------------

def embedding_alignment_score(
    source_embs: Tensor,
    target_embs: Tensor,
) -> float:
    """Mean cosine similarity between paired source and target embeddings.

    Args:
        source_embs: (N, D)
        target_embs: (N, D)

    Returns:
        Scalar float in [-1, 1].
    """
    sims = compute_cosine_similarity(source_embs, target_embs)  # (N,)
    return float(sims.mean().item())


# ---------------------------------------------------------------------------
# Uniformity score  (Wang & Isola 2020)
# ---------------------------------------------------------------------------

def embedding_uniformity_score(embeddings: Tensor) -> float:
    """Uniformity loss from Wang & Isola 2020.

    uniformity = log( mean_{i<j} exp(-2 * ||z_i - z_j||^2) )

    More negative value => more uniformly distributed on the hypersphere.

    Args:
        embeddings: (N, D)

    Returns:
        Float scalar (more negative = more uniform).
    """
    N = embeddings.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 embeddings for uniformity score.")

    # Squared pairwise distances via the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b.
    sq_norm = (embeddings ** 2).sum(dim=-1, keepdim=True)          # (N, 1)
    sq_dist = sq_norm + sq_norm.T - 2.0 * (embeddings @ embeddings.T)  # (N, N)
    sq_dist = sq_dist.clamp(min=0.0)

    # Collect upper-triangular pairs (i < j).
    i_idx, j_idx = torch.triu_indices(N, N, offset=1)
    pairwise_sq = sq_dist[i_idx, j_idx]           # (N*(N-1)/2,)

    # Uniformity loss.
    loss = torch.log(torch.exp(-2.0 * pairwise_sq).mean())
    return float(loss.item())


# ---------------------------------------------------------------------------
# SemanticSimilarityScorer
# ---------------------------------------------------------------------------

class SemanticSimilarityScorer:
    """Encode hidden states and compute semantic similarity metrics."""

    def __init__(self, config: SimilarityConfig) -> None:
        self.config = config

    def encode(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool hidden states and optionally L2-normalize.

        Args:
            hidden_states: (B, T, D)
            attention_mask: Optional (B, T) mask (1=real token).

        Returns:
            (B, D) embedding tensor.
        """
        emb = pool_hidden_states(
            hidden_states,
            attention_mask=attention_mask,
            mode=self.config.pooling,
        )  # (B, D)

        if self.config.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb

    def similarity(
        self,
        hidden_a: Tensor,
        hidden_b: Tensor,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute element-wise similarity for paired sequences.

        Args:
            hidden_a: (B, T, D)
            hidden_b: (B, T, D)
            mask_a: Optional (B, T) mask for a.
            mask_b: Optional (B, T) mask for b.

        Returns:
            (B,) similarity per pair.
        """
        emb_a = self.encode(hidden_a, mask_a)  # (B, D)
        emb_b = self.encode(hidden_b, mask_b)  # (B, D)

        metric = self.config.similarity_metric

        if metric == "cosine":
            # Embeddings may already be normalized; normalize again for safety.
            return compute_cosine_similarity(emb_a, emb_b)

        elif metric == "dot":
            return (emb_a * emb_b).sum(dim=-1)

        elif metric == "l2":
            return -(emb_a - emb_b).norm(p=2, dim=-1)

        else:
            raise ValueError(f"Unknown similarity metric: {metric!r}")

    def similarity_matrix(
        self,
        hidden: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute all-pairs similarity matrix.

        Args:
            hidden: (N, T, D)
            mask: Optional (N, T) mask.

        Returns:
            (N, N) similarity matrix.
        """
        emb = self.encode(hidden, mask)  # (N, D)
        return compute_pairwise_similarity(emb, metric=self.config.similarity_metric)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    query_embs: Tensor,
    key_embs: Tensor,
    top_k: int = 1,
) -> dict:
    """Compute recall@k and MRR for embedding retrieval.

    For each query i, the correct key is key_embs[i].  We rank all keys by
    cosine similarity to the query and check whether key i appears in the
    top-k results.

    Args:
        query_embs: (N, D)
        key_embs:   (N, D)
        top_k:      Number of top results to consider for recall.

    Returns:
        {"recall@k": float, "mrr": float}
    """
    N = query_embs.shape[0]

    # Compute cosine similarity matrix: (N_q, N_k).
    q_norm = F.normalize(query_embs, p=2, dim=-1)
    k_norm = F.normalize(key_embs, p=2, dim=-1)
    sim_matrix = q_norm @ k_norm.T  # (N, N)

    # Rank keys for each query (descending similarity).
    # sorted_indices[i] = indices of keys sorted from most to least similar.
    sorted_indices = sim_matrix.argsort(dim=-1, descending=True)  # (N, N)

    recall_hits = 0
    reciprocal_ranks: list[float] = []

    for i in range(N):
        ranked = sorted_indices[i]  # (N,) indices of keys, best first

        # Recall@k: is the correct key (index i) in the top-k?
        top_k_indices = ranked[:top_k].tolist()
        if i in top_k_indices:
            recall_hits += 1

        # MRR: rank is 1-based position of the correct key.
        rank_position = (ranked == i).nonzero(as_tuple=True)[0]
        if len(rank_position) > 0:
            rank = int(rank_position[0].item()) + 1  # 1-based
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = recall_hits / N if N > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {"recall@k": recall_at_k, "mrr": mrr}
