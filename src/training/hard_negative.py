"""
Hard Negative Mining for contrastive and preference training in the Aurelius LLM project.

Random negatives are easy to distinguish from anchors, leading to weak gradient signal.
Hard negatives are similar to positives and thus more challenging, producing stronger
gradients and more efficient training.

Supported strategies:
- hardest: the closest negative in embedding space (smallest distance)
- semi_hard: negatives farther than the positive but within a margin
- distance_weighted: sample negatives with probability proportional to difficulty
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hard Negative Miner
# ---------------------------------------------------------------------------

class HardNegativeMiner:
    """
    Find hard negative samples within a batch for contrastive learning.

    Args:
        margin: margin for semi-hard mining (default 0.2)
        mining_strategy: 'hardest', 'semi_hard', or 'distance_weighted'
        distance_metric: 'cosine', 'euclidean', or 'dot_product'
    """

    def __init__(
        self,
        margin: float = 0.2,
        mining_strategy: str = 'semi_hard',
        distance_metric: str = 'cosine',
    ) -> None:
        if mining_strategy not in ('hardest', 'semi_hard', 'distance_weighted'):
            raise ValueError(
                f"mining_strategy must be one of 'hardest', 'semi_hard', "
                f"'distance_weighted', got '{mining_strategy}'"
            )
        if distance_metric not in ('cosine', 'euclidean', 'dot_product'):
            raise ValueError(
                f"distance_metric must be one of 'cosine', 'euclidean', "
                f"'dot_product', got '{distance_metric}'"
            )
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.distance_metric = distance_metric

    # ------------------------------------------------------------------
    def compute_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all embeddings.

        Args:
            embeddings: (N, d) tensor of embeddings.

        Returns:
            (N, N) distance matrix.

        For cosine: distance = 1 - cosine_similarity  (range [0, 2])
        For euclidean: L2 distance
        For dot_product: -dot_product (higher dot = closer, so negate)
        """
        if self.distance_metric == 'cosine':
            normed = F.normalize(embeddings, p=2, dim=-1)  # (N, d)
            # cosine similarity in [-1, 1]; distance in [0, 2]
            sim = torch.matmul(normed, normed.T)            # (N, N)
            # Clamp to handle tiny floating-point violations
            return (1.0 - sim).clamp(min=0.0, max=2.0)

        elif self.distance_metric == 'euclidean':
            # Expand to compute pairwise L2: ||a - b||_2
            # Use the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            sq_norms = (embeddings ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
            dot = torch.matmul(embeddings, embeddings.T)            # (N, N)
            sq_dist = sq_norms + sq_norms.T - 2.0 * dot
            # Clamp to avoid numerical negatives before sqrt
            sq_dist = sq_dist.clamp(min=0.0)
            return sq_dist.sqrt()

        else:  # dot_product
            dot = torch.matmul(embeddings, embeddings.T)  # (N, N)
            return -dot

    # ------------------------------------------------------------------
    def mine_hardest(
        self,
        anchor_idx: torch.Tensor,    # (B,)
        positive_idx: torch.Tensor,  # (B,)
        distances: torch.Tensor,     # (N, N)
        labels: torch.Tensor,        # (N,)
    ) -> torch.Tensor:
        """
        For each anchor, find the hardest negative: the negative with the
        SMALLEST distance to the anchor (most similar, hardest to distinguish).

        Returns:
            (B,) indices of hard negatives.
        """
        B = anchor_idx.shape[0]
        N = distances.shape[0]
        device = distances.device

        neg_indices = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            a = anchor_idx[i].item()
            anchor_label = labels[a]

            # Mask: valid negatives are different class AND different index from anchor
            neg_mask = (labels != anchor_label)  # (N,)
            neg_mask[a] = False                  # exclude self

            # Among valid negatives, find the one with the smallest distance
            d_row = distances[a].clone()
            # Set non-negative entries to large value so argmin ignores them
            d_row[~neg_mask] = float('inf')

            neg_indices[i] = d_row.argmin()

        return neg_indices

    # ------------------------------------------------------------------
    def mine_semi_hard(
        self,
        anchor_idx: torch.Tensor,    # (B,)
        positive_idx: torch.Tensor,  # (B,)
        distances: torch.Tensor,     # (N, N)
        labels: torch.Tensor,        # (N,)
    ) -> torch.Tensor:
        """
        Semi-hard negatives: negatives that are farther than the positive
        (d(a,n) > d(a,p)) but within the margin (d(a,n) < d(a,p) + margin).

        Falls back to hardest negative when no semi-hard negative exists.

        Returns:
            (B,) indices of semi-hard (or fallback hardest) negatives.
        """
        B = anchor_idx.shape[0]
        device = distances.device

        neg_indices = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            a = anchor_idx[i].item()
            p = positive_idx[i].item()
            anchor_label = labels[a]

            d_ap = distances[a, p]

            # Base negative mask: different class, not self
            neg_mask = (labels != anchor_label)
            neg_mask[a] = False

            d_row = distances[a]

            # Semi-hard: d(a,n) > d(a,p) AND d(a,n) < d(a,p) + margin
            semi_mask = neg_mask & (d_row > d_ap) & (d_row < d_ap + self.margin)

            if semi_mask.any():
                # Among semi-hard negatives, pick the closest (hardest within range)
                d_semi = d_row.clone()
                d_semi[~semi_mask] = float('inf')
                neg_indices[i] = d_semi.argmin()
            else:
                # Fall back: hardest negative (smallest distance)
                d_hard = d_row.clone()
                d_hard[~neg_mask] = float('inf')
                neg_indices[i] = d_hard.argmin()

        return neg_indices

    # ------------------------------------------------------------------
    def mine_distance_weighted(
        self,
        anchor_idx: torch.Tensor,   # (B,)
        distances: torch.Tensor,    # (N, N)
        labels: torch.Tensor,       # (N,)
    ) -> torch.Tensor:
        """
        Sample negatives with probability proportional to their difficulty.
        P(sample) ∝ exp(-distance)  (closer = more likely to be sampled).

        Returns:
            (B,) indices of sampled negatives.
        """
        B = anchor_idx.shape[0]
        device = distances.device

        neg_indices = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            a = anchor_idx[i].item()
            anchor_label = labels[a]

            neg_mask = (labels != anchor_label)
            neg_mask[a] = False

            d_row = distances[a].clone()

            # P ∝ exp(-d): assign 0 probability to non-negatives
            weights = torch.exp(-d_row)
            weights[~neg_mask] = 0.0

            # Normalise; if all zeros fall back to uniform over negatives
            total = weights.sum()
            if total < 1e-12:
                weights = neg_mask.float()
                total = weights.sum()

            weights = weights / total

            # Sample one index
            neg_indices[i] = torch.multinomial(weights, num_samples=1).squeeze(0)

        return neg_indices

    # ------------------------------------------------------------------
    def mine(
        self,
        embeddings: torch.Tensor,    # (N, d)
        labels: torch.Tensor,        # (N,)
        anchor_idx: torch.Tensor,    # (B,)
        positive_idx: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Route to correct mining strategy.

        Returns:
            (B,) negative indices.
        """
        distances = self.compute_distances(embeddings)

        if self.mining_strategy == 'hardest':
            return self.mine_hardest(anchor_idx, positive_idx, distances, labels)
        elif self.mining_strategy == 'semi_hard':
            return self.mine_semi_hard(anchor_idx, positive_idx, distances, labels)
        else:  # distance_weighted
            return self.mine_distance_weighted(anchor_idx, distances, labels)


# ---------------------------------------------------------------------------
# Triplet Loss with Online Hard Negative Mining
# ---------------------------------------------------------------------------

class TripletLossWithMining(nn.Module):
    """Triplet loss with online hard negative mining.

    L = max(0, d(a,p) - d(a,n) + margin)

    Args:
        margin: triplet margin
        miner: HardNegativeMiner instance (created with defaults if None)
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        margin: float = 0.2,
        miner: HardNegativeMiner | None = None,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.margin = margin
        self.miner = miner or HardNegativeMiner(margin=margin)
        if reduction not in ('mean', 'sum'):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction

    # ------------------------------------------------------------------
    def forward(
        self,
        embeddings: torch.Tensor,  # (N, d)
        labels: torch.Tensor,      # (N,)
    ) -> torch.Tensor:
        """
        For each sample as anchor:
        1. Find a positive (same class, different index).
        2. Mine a hard negative (different class).
        3. Compute triplet loss.

        Returns:
            Scalar loss tensor.
        """
        N = embeddings.shape[0]
        device = embeddings.device

        # Build anchor / positive pairs: for each sample find one positive
        anchor_list: list[int] = []
        positive_list: list[int] = []

        for i in range(N):
            # Positive: same label, different index
            same_class = (labels == labels[i]).nonzero(as_tuple=False).squeeze(1)
            candidates = same_class[same_class != i]
            if candidates.numel() == 0:
                continue  # skip samples with no positive in batch
            # Pick first positive (deterministic)
            anchor_list.append(i)
            positive_list.append(candidates[0].item())

        if not anchor_list:
            # Degenerate: all samples have unique labels — return zero loss
            return embeddings.new_zeros(())

        anchor_idx = torch.tensor(anchor_list, dtype=torch.long, device=device)
        positive_idx = torch.tensor(positive_list, dtype=torch.long, device=device)

        # Mine hard negatives
        negative_idx = self.miner.mine(embeddings, labels, anchor_idx, positive_idx)

        # Compute pairwise distances using the miner's metric
        distances = self.miner.compute_distances(embeddings)

        d_ap = distances[anchor_idx, positive_idx]   # (B,)
        d_an = distances[anchor_idx, negative_idx]   # (B,)

        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        losses = F.relu(d_ap - d_an + self.margin)   # (B,)

        if self.reduction == 'mean':
            return losses.mean()
        else:
            return losses.sum()


# ---------------------------------------------------------------------------
# Preference Hard Negative Mining (for DPO / RLHF)
# ---------------------------------------------------------------------------

class PreferenceHardNegative:
    """
    Hard negative mining for DPO / preference training.

    Given a set of (prompt, chosen, rejected) triples, find harder rejected
    responses by looking at other items' chosen responses.

    "Hard" rejected = a chosen response from another example that scores
    similarly to the current chosen (confusing for the reward model).

    Args:
        similarity_threshold: cosine similarity above which another example's
            chosen embedding is considered a hard negative (default 0.7).
    """

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    def hard_negative_indices(
        self,
        chosen_embeddings: torch.Tensor,  # (B, d)
    ) -> torch.Tensor:
        """Return (B,) indices: for each i, which j to use as hard negative.

        Finds j ≠ i whose chosen_embedding is most similar to chosen_embedding[i].
        If that similarity >= similarity_threshold, returns j; otherwise returns i
        (so the caller can fall back to the original rejected using the same index).
        """
        B = chosen_embeddings.shape[0]
        device = chosen_embeddings.device

        normed = F.normalize(chosen_embeddings, p=2, dim=-1)  # (B, d)
        sim = torch.matmul(normed, normed.T)                  # (B, B)

        # Mask out self-similarity
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(diag_mask, -float('inf'))

        # For each i, find the most similar j ≠ i
        best_sim, best_j = sim.max(dim=1)  # (B,) each

        # If similarity is below threshold, fall back to self (keep original rejected)
        indices = torch.where(
            best_sim >= self.similarity_threshold,
            best_j,
            torch.arange(B, dtype=torch.long, device=device),
        )

        return indices

    # ------------------------------------------------------------------
    def find_hard_rejected(
        self,
        chosen_embeddings: torch.Tensor,    # (B, d)
        rejected_embeddings: torch.Tensor,  # (B, d)
    ) -> torch.Tensor:
        """
        For each example i:
        - Find example j (j ≠ i) whose chosen_embedding is closest to chosen_embedding[i].
        - If similarity > threshold, use chosen_embedding[j] as hard negative for i.
        - Else keep original rejected_embedding[i].

        Returns:
            (B, d) hard rejected embeddings.
        """
        indices = self.hard_negative_indices(chosen_embeddings)  # (B,)
        B = chosen_embeddings.shape[0]
        device = chosen_embeddings.device

        hard_rejected = torch.zeros_like(rejected_embeddings)

        for i in range(B):
            j = indices[i].item()
            if j == i:
                # Below threshold — keep original rejected
                hard_rejected[i] = rejected_embeddings[i]
            else:
                # Use chosen[j] as a hard negative for example i
                hard_rejected[i] = chosen_embeddings[j]

        return hard_rejected
