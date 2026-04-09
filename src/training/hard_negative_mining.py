"""Online hard negative mining for contrastive and metric learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class HardNegativeConfig:
    """Configuration for online hard negative mining."""

    mining_strategy: str = "semi-hard"
    """Mining strategy: "hard" | "semi-hard" | "distance-weighted"."""

    margin: float = 0.5
    """Triplet margin for semi-hard mining and triplet loss."""

    distance_metric: str = "cosine"
    """Distance metric: "cosine" | "euclidean"."""

    n_hard_negatives: int = 1
    """Number of hard negatives per anchor."""

    temperature: float = 0.07
    """Temperature for InfoNCE loss."""

    max_pairs: int = 1000
    """Maximum number of pairwise comparisons to limit memory usage."""


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def pairwise_cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
    """Compute pairwise cosine similarity between two sets of vectors.

    Args:
        x: (M, d) tensor of query vectors.
        y: (N, d) tensor of key vectors.

    Returns:
        (M, N) cosine similarity matrix.
    """
    x_norm = F.normalize(x, p=2, dim=-1)  # (M, d)
    y_norm = F.normalize(y, p=2, dim=-1)  # (N, d)
    return x_norm @ y_norm.T              # (M, N)


def pairwise_euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    """Compute pairwise squared Euclidean distance between two sets of vectors.

    Uses the expand trick: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b

    Args:
        x: (M, d) tensor.
        y: (N, d) tensor.

    Returns:
        (M, N) squared Euclidean distance matrix.
    """
    x_sq = (x * x).sum(dim=-1, keepdim=True)   # (M, 1)
    y_sq = (y * y).sum(dim=-1, keepdim=True)   # (N, 1)
    dot = x @ y.T                               # (M, N)
    sq_dist = x_sq + y_sq.T - 2.0 * dot        # (M, N)
    return sq_dist.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def _get_distance_matrix(embeddings: Tensor, distance_metric: str) -> Tensor:
    """Return (B, B) pairwise distance matrix."""
    if distance_metric == "cosine":
        # Convert similarity to distance: d = 1 - sim
        sim = pairwise_cosine_similarity(embeddings, embeddings)
        return (1.0 - sim).clamp(min=0.0)
    else:  # euclidean
        return pairwise_euclidean_distance(embeddings, embeddings)


def mine_hard_negatives(
    embeddings: Tensor,
    labels: Tensor,
    config: HardNegativeConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mine hard negative triplets from a batch of embeddings.

    For each anchor i:
    - Identify positives: samples with the same class label.
    - Identify negatives: samples with a different class label.
    - Select hard negative according to config.mining_strategy.

    Strategies:
    - "hard": negative closest to the anchor (minimum distance).
    - "semi-hard": negative farther than the positive but within the margin;
      falls back to the hard negative when no semi-hard candidate exists.
    - "distance-weighted": sample negatives weighted by inverse distance
      (closer negatives receive higher sampling weight).

    Args:
        embeddings: (B, d) embedding matrix.
        labels: (B,) integer class labels.
        config: HardNegativeConfig instance controlling mining behaviour.

    Returns:
        Tuple (anchors, positives, negatives), each (N_triplets, d).
        May contain fewer than B triplets when an anchor has no valid triplet.
    """
    B = embeddings.shape[0]
    device = embeddings.device

    dist_matrix = _get_distance_matrix(embeddings, config.distance_metric)

    anchor_list: list[Tensor] = []
    positive_list: list[Tensor] = []
    negative_list: list[Tensor] = []

    for i in range(B):
        label_i = labels[i]

        # Masks for positives and negatives
        pos_mask = (labels == label_i)
        pos_mask[i] = False  # exclude self
        neg_mask = (labels != label_i)

        if not pos_mask.any() or not neg_mask.any():
            continue  # no valid triplet for this anchor

        # Choose a positive: the closest positive
        d_row = dist_matrix[i]
        pos_dists = d_row.clone()
        pos_dists[~pos_mask] = float("inf")
        pos_idx = pos_dists.argmin()

        d_ap = d_row[pos_idx]

        # Mine negative according to strategy
        neg_idx = _select_negative(
            d_row=d_row,
            neg_mask=neg_mask,
            d_ap=d_ap,
            config=config,
            device=device,
        )

        if neg_idx is None:
            continue

        anchor_list.append(embeddings[i])
        positive_list.append(embeddings[pos_idx])
        negative_list.append(embeddings[neg_idx])

    if not anchor_list:
        # Return empty tensors with correct embedding dimension
        d = embeddings.shape[1]
        empty = embeddings.new_zeros(0, d)
        return empty, empty, empty

    anchors = torch.stack(anchor_list, dim=0)
    positives = torch.stack(positive_list, dim=0)
    negatives = torch.stack(negative_list, dim=0)
    return anchors, positives, negatives


def _select_negative(
    d_row: Tensor,
    neg_mask: Tensor,
    d_ap: Tensor,
    config: HardNegativeConfig,
    device: torch.device,
) -> int | None:
    """Select a single negative index for one anchor.

    Returns None when no valid negative can be found.
    """
    strategy = config.mining_strategy

    if strategy == "hard":
        d_neg = d_row.clone()
        d_neg[~neg_mask] = float("inf")
        if d_neg.min() == float("inf"):
            return None
        return int(d_neg.argmin().item())

    elif strategy == "semi-hard":
        # Semi-hard: d(a,n) > d(a,p)  AND  d(a,n) < d(a,p) + margin
        semi_mask = neg_mask & (d_row > d_ap) & (d_row < d_ap + config.margin)
        if semi_mask.any():
            d_semi = d_row.clone()
            d_semi[~semi_mask] = float("inf")
            return int(d_semi.argmin().item())
        else:
            # Fall back to hard negative
            d_neg = d_row.clone()
            d_neg[~neg_mask] = float("inf")
            if d_neg.min() == float("inf"):
                return None
            return int(d_neg.argmin().item())

    else:  # "distance-weighted"
        # Weight ∝ exp(-d): closer negatives are harder, so sampled more
        weights = torch.exp(-d_row)
        weights[~neg_mask] = 0.0
        total = weights.sum()
        if total < 1e-12:
            # Fall back to uniform over negatives
            weights = neg_mask.float()
            total = weights.sum()
            if total < 1e-12:
                return None
        weights = weights / total
        return int(torch.multinomial(weights, num_samples=1).item())


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def triplet_loss(
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    margin: float = 0.5,
    distance_metric: str = "cosine",
) -> Tensor:
    """Standard margin-based triplet loss.

    Loss = mean(max(d(a, p) - d(a, n) + margin, 0))

    For cosine distance: d = 1 - cosine_similarity.

    Args:
        anchors: (N, d) anchor embeddings.
        positives: (N, d) positive embeddings.
        negatives: (N, d) negative embeddings.
        margin: Triplet margin.
        distance_metric: "cosine" | "euclidean".

    Returns:
        Scalar loss tensor.
    """
    if distance_metric == "cosine":
        # d = 1 - cosine_sim; use row-wise dot after normalisation
        a_norm = F.normalize(anchors, p=2, dim=-1)
        p_norm = F.normalize(positives, p=2, dim=-1)
        n_norm = F.normalize(negatives, p=2, dim=-1)

        sim_ap = (a_norm * p_norm).sum(dim=-1)  # (N,)
        sim_an = (a_norm * n_norm).sum(dim=-1)  # (N,)

        d_ap = 1.0 - sim_ap
        d_an = 1.0 - sim_an
    else:  # euclidean — use squared distance for efficiency
        sq_ap = ((anchors - positives) ** 2).sum(dim=-1)  # (N,)
        sq_an = ((anchors - negatives) ** 2).sum(dim=-1)  # (N,)
        d_ap = sq_ap.clamp(min=0.0).sqrt()
        d_an = sq_an.clamp(min=0.0).sqrt()

    losses = F.relu(d_ap - d_an + margin)
    return losses.mean()


def infonce_loss(
    query: Tensor,
    positives: Tensor,
    negatives: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE loss (contrastive loss with in-batch negatives).

    Loss = -log(
        exp(sim(q, p) / T) /
        (exp(sim(q, p) / T) + sum_k exp(sim(q, neg_k) / T))
    )

    Args:
        query: (B, d) query embeddings.
        positives: (B, d) positive embeddings.
        negatives: (B, K, d) negative embeddings (K negatives per query).
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss tensor.
    """
    B, d = query.shape
    K = negatives.shape[1]

    q_norm = F.normalize(query, p=2, dim=-1)       # (B, d)
    p_norm = F.normalize(positives, p=2, dim=-1)   # (B, d)
    n_norm = F.normalize(negatives, p=2, dim=-1)   # (B, K, d)

    # Similarity between query and positive: (B,)
    sim_pos = (q_norm * p_norm).sum(dim=-1) / temperature

    # Similarity between query and each negative: (B, K)
    # q_norm: (B, 1, d) -> broadcast over K
    sim_neg = (q_norm.unsqueeze(1) * n_norm).sum(dim=-1) / temperature  # (B, K)

    # log-sum-exp denominator: log(exp(sim_pos) + sum_k exp(sim_neg_k))
    # Stack: (B, 1+K)
    all_logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # (B, 1+K)

    # Log probability of positive (index 0 in all_logits)
    log_denom = torch.logsumexp(all_logits, dim=1)  # (B,)
    log_numerator = sim_pos                          # (B,)

    loss = -(log_numerator - log_denom).mean()
    return loss


# ---------------------------------------------------------------------------
# HardNegativeMiner wrapper
# ---------------------------------------------------------------------------

class HardNegativeMiner:
    """Wraps hard negative mining and triplet loss computation.

    Args:
        config: HardNegativeConfig controlling all mining and loss parameters.
    """

    def __init__(self, config: HardNegativeConfig) -> None:
        self.config = config

    def mine_and_compute_loss(
        self, embeddings: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict]:
        """Mine hard negative triplets and compute triplet loss.

        Args:
            embeddings: (B, d) embedding matrix.
            labels: (B,) integer class labels.

        Returns:
            Tuple of:
            - loss: scalar triplet loss tensor.
            - info: dict with keys "n_triplets" (int), "mean_pos_dist" (float),
              "mean_neg_dist" (float).
        """
        anchors, positives, negatives = mine_hard_negatives(
            embeddings, labels, self.config
        )

        n_triplets = anchors.shape[0]

        if n_triplets == 0:
            loss = embeddings.new_zeros(()).requires_grad_(True)
            return loss, {
                "n_triplets": 0,
                "mean_pos_dist": 0.0,
                "mean_neg_dist": 0.0,
            }

        loss = triplet_loss(
            anchors,
            positives,
            negatives,
            margin=self.config.margin,
            distance_metric=self.config.distance_metric,
        )

        # Compute diagnostic distances (no-grad)
        with torch.no_grad():
            if self.config.distance_metric == "cosine":
                a_norm = F.normalize(anchors, p=2, dim=-1)
                p_norm = F.normalize(positives, p=2, dim=-1)
                n_norm = F.normalize(negatives, p=2, dim=-1)
                mean_pos_dist = float((1.0 - (a_norm * p_norm).sum(dim=-1)).mean().item())
                mean_neg_dist = float((1.0 - (a_norm * n_norm).sum(dim=-1)).mean().item())
            else:
                mean_pos_dist = float(
                    ((anchors - positives) ** 2).sum(dim=-1).clamp(min=0).sqrt().mean().item()
                )
                mean_neg_dist = float(
                    ((anchors - negatives) ** 2).sum(dim=-1).clamp(min=0).sqrt().mean().item()
                )

        info = {
            "n_triplets": n_triplets,
            "mean_pos_dist": mean_pos_dist,
            "mean_neg_dist": mean_neg_dist,
        }
        return loss, info
