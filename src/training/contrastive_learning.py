"""SimCSE-style contrastive learning for sentence embeddings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive (SimCSE-style) training."""

    temperature: float = 0.05  # NT-Xent temperature
    hard_negative_weight: float = 0.0  # weight for hard negatives (0 = disabled)
    pooling: str = "mean"  # "mean" | "cls" | "last"


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def pool_hidden_states(
    hidden: Tensor,
    attention_mask: Tensor | None = None,
    pooling: str = "mean",
) -> Tensor:
    """Pool (B, T, D) hidden states to (B, D) embeddings.

    Args:
        hidden: shape (B, T, D)
        attention_mask: shape (B, T), 1 for real tokens, 0 for padding.
                        Only used by "mean" and "last" modes.
        pooling: "mean", "cls", or "last"

    Returns:
        Tensor of shape (B, D)
    """
    if pooling == "cls":
        return hidden[:, 0, :]

    if pooling == "mean":
        if attention_mask is None:
            return hidden.mean(dim=1)
        # masked average
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        sum_hidden = (hidden * mask).sum(dim=1)  # (B, D)
        count = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return sum_hidden / count

    if pooling == "last":
        if attention_mask is None:
            return hidden[:, -1, :]
        # index of last real token per sequence
        lengths = attention_mask.sum(dim=1).long() - 1  # (B,)
        lengths = lengths.clamp(min=0)
        batch_size = hidden.size(0)
        idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)  # (B, D)

    raise ValueError(f"Unknown pooling mode: {pooling!r}. Choose 'mean', 'cls', or 'last'.")


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def normalize_embeddings(embeddings: Tensor, eps: float = 1e-8) -> Tensor:
    """L2-normalize embeddings along the last dimension.

    Args:
        embeddings: (..., D)
        eps: small constant for numerical stability

    Returns:
        Tensor of same shape with unit norm along last dim.
    """
    return F.normalize(embeddings, p=2, dim=-1, eps=eps)


def cosine_similarity_matrix(a: Tensor, b: Tensor) -> Tensor:
    """Compute pairwise cosine similarity between two sets of L2-normalized embeddings.

    Both ``a`` and ``b`` must already be L2-normalized.

    Args:
        a: (N, D) normalized embeddings
        b: (N, D) normalized embeddings

    Returns:
        (N, N) cosine similarity matrix where entry [i, j] = a[i] · b[j]
    """
    return torch.mm(a, b.t())


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def nt_xent_loss(similarity_matrix: Tensor, temperature: float) -> Tensor:
    """Normalized temperature-scaled cross-entropy (NT-Xent) loss.

    Diagonal entries are treated as positives; off-diagonal as negatives.

    Args:
        similarity_matrix: (N, N) cosine similarity matrix
        temperature: scaling factor

    Returns:
        Scalar loss tensor.
    """
    n = similarity_matrix.size(0)
    logits = similarity_matrix / temperature  # (N, N)
    labels = torch.arange(n, device=similarity_matrix.device)
    return F.cross_entropy(logits, labels)


def in_batch_negatives_loss(
    anchor: Tensor,
    positive: Tensor,
    temperature: float,
) -> Tensor:
    """SimCSE unsupervised contrastive loss with in-batch negatives.

    ``anchor`` and ``positive`` are two views of the same batch of sentences
    (same input encoded twice with dropout).  Both are already L2-normalized.

    Args:
        anchor:   (B, D) normalized embeddings
        positive: (B, D) normalized embeddings
        temperature: NT-Xent temperature

    Returns:
        Scalar loss tensor.
    """
    sim = cosine_similarity_matrix(anchor, positive)  # (B, B)
    return nt_xent_loss(sim, temperature)


def hard_negative_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    temperature: float,
) -> Tensor:
    """Contrastive loss with in-batch negatives *and* hard negatives.

    For each anchor the candidates are:
      - The corresponding positive (target)
      - All other positives in the batch (in-batch negatives)
      - All hard negatives in the batch

    Args:
        anchor:   (B, D) normalized embeddings
        positive: (B, D) normalized embeddings
        negative: (B, D) hard-negative normalized embeddings
        temperature: NT-Xent temperature

    Returns:
        Scalar loss tensor.
    """
    # Concatenate positives and hard-negatives as columns
    # candidates: (B, 2*B) — first B cols are positives, next B cols are negatives
    candidates = torch.cat([positive, negative], dim=0)  # (2B, D)
    sim = cosine_similarity_matrix(anchor, candidates)  # (B, 2B)
    logits = sim / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# ContrastiveLearner
# ---------------------------------------------------------------------------


class ContrastiveLearner:
    """High-level wrapper that encodes, pools, normalizes and computes loss."""

    def __init__(
        self,
        encoder_fn: Callable[[Tensor], Tensor],
        config: ContrastiveConfig,
    ) -> None:
        """
        Args:
            encoder_fn: callable (B, T) -> (B, T, D) hidden states
            config: ContrastiveConfig instance
        """
        self.encoder_fn = encoder_fn
        self.config = config

    def encode(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode token IDs to unit-norm sentence embeddings.

        Args:
            token_ids:      (B, T) integer token ids
            attention_mask: (B, T) optional mask (1=real, 0=pad)

        Returns:
            (B, D) L2-normalized embeddings
        """
        hidden = self.encoder_fn(token_ids)  # (B, T, D)
        pooled = pool_hidden_states(hidden, attention_mask, self.config.pooling)
        return normalize_embeddings(pooled)

    def compute_loss(
        self,
        anchor_ids: Tensor,
        positive_ids: Tensor,
        negative_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute contrastive loss and diagnostics.

        Args:
            anchor_ids:   (B, T)
            positive_ids: (B, T)
            negative_ids: (B, T) optional hard negatives

        Returns:
            dict with keys:
              "loss"               – scalar loss
              "similarity_matrix"  – (B, B) anchor–positive similarity
              "mean_positive_sim"  – mean diagonal similarity
              "mean_negative_sim"  – mean off-diagonal similarity
        """
        anchor_emb = self.encode(anchor_ids)  # (B, D)
        positive_emb = self.encode(positive_ids)  # (B, D)

        sim_matrix = cosine_similarity_matrix(anchor_emb, positive_emb)  # (B, B)

        b = anchor_emb.size(0)
        diag_mask = torch.eye(b, dtype=torch.bool, device=sim_matrix.device)
        mean_pos = sim_matrix[diag_mask].mean()
        mean_neg = (
            sim_matrix[~diag_mask].mean()
            if b > 1
            else torch.zeros(1, device=sim_matrix.device).squeeze()
        )

        if negative_ids is not None:
            negative_emb = self.encode(negative_ids)
            loss = hard_negative_loss(
                anchor_emb,
                positive_emb,
                negative_emb,
                self.config.temperature,
            )
        else:
            loss = in_batch_negatives_loss(
                anchor_emb,
                positive_emb,
                self.config.temperature,
            )

        return {
            "loss": loss,
            "similarity_matrix": sim_matrix,
            "mean_positive_sim": mean_pos,
            "mean_negative_sim": mean_neg,
        }


# ---------------------------------------------------------------------------
# Embedding quality metrics
# ---------------------------------------------------------------------------


def compute_alignment(a: Tensor, b: Tensor) -> float:
    """Alignment metric for normalized embeddings.

    alignment = -mean(||a_i - b_i||^2)

    Higher (less negative) is better — perfectly aligned pairs give 0.

    Args:
        a: (N, D) L2-normalized embeddings
        b: (N, D) L2-normalized embeddings

    Returns:
        float scalar (<= 0)
    """
    diff = a - b  # (N, D)
    sq_dist = (diff**2).sum(dim=-1)  # (N,)
    return (-sq_dist.mean()).item()


def compute_uniformity(embeddings: Tensor, t: float = 2.0) -> float:
    """Uniformity metric for a set of normalized embeddings.

    uniformity = log( mean_{i,j} exp(-t * ||e_i - e_j||^2) )

    Lower is better — perfectly uniform embeddings minimise this.

    Args:
        embeddings: (N, D) L2-normalized embeddings
        t: kernel bandwidth

    Returns:
        float scalar
    """
    # pairwise squared distances via broadcasting
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, D)
    sq_dist = (diff**2).sum(dim=-1)  # (N, N)
    return torch.log(torch.exp(-t * sq_dist).mean()).item()
