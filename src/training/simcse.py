"""
SimCSE: Simple Contrastive Learning of Sentence Embeddings
Gao et al. 2021

Unsupervised SimCSE: dropout as minimal data augmentation
Supervised SimCSE: NLI entailment/contradiction pairs as hard negatives
Alignment & Uniformity: Wang & Isola 2020 embedding quality metrics
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# SimCSEEncoder
# ---------------------------------------------------------------------------


class SimCSEEncoder(nn.Module):
    """Wraps an arbitrary backbone to produce pooled sentence embeddings.

    Args:
        backbone: callable(input_ids: Tensor[B, T]) -> hidden_states Tensor[B, T, D]
        d_model: hidden dimension of the backbone
        pooling: one of "cls" | "mean" | "last"
    """

    def __init__(self, backbone: Callable, d_model: int, pooling: str = "cls") -> None:
        super().__init__()
        if pooling not in ("cls", "mean", "last"):
            raise ValueError(f"pooling must be 'cls', 'mean', or 'last', got {pooling!r}")
        self.backbone = backbone
        self.d_model = d_model
        self.pooling = pooling

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, T) integer token ids
        Returns:
            embeddings: (B, d_model)
        """
        hidden = self.backbone(input_ids)  # (B, T, D)
        if self.pooling == "cls":
            emb = hidden[:, 0, :]
        elif self.pooling == "mean":
            emb = hidden.mean(dim=1)
        else:  # "last"
            emb = hidden[:, -1, :]
        return emb  # (B, D)

    def normalize(self, embeddings: Tensor) -> Tensor:
        """L2-normalize embeddings along last dimension."""
        return F.normalize(embeddings, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Unsupervised SimCSE Loss (NT-Xent with dropout augmentation)
# ---------------------------------------------------------------------------


class UnsupervisedSimCSELoss(nn.Module):
    """NT-Xent contrastive loss using two dropout passes as positives.

    Args:
        temperature: softmax temperature tau (default 0.05)
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        encoder: SimCSEEncoder,
        input_ids: Tensor,
    ) -> tuple:
        """
        Args:
            encoder: SimCSEEncoder (must be in train mode for dropout to vary)
            input_ids: (B, T)
        Returns:
            (loss, avg_cosine_pos, avg_cosine_neg)
        """
        z1 = encoder(input_ids)
        z2 = encoder(input_ids)

        z1 = encoder.normalize(z1)
        z2 = encoder.normalize(z2)

        B = z1.size(0)
        sim = torch.mm(z1, z2.t()) / self.temperature  # (B, B)
        labels = torch.arange(B, device=z1.device)
        loss = F.cross_entropy(sim, labels)

        with torch.no_grad():
            cos_matrix = torch.mm(z1, z2.t())
            pos_sims = cos_matrix.diagonal()
            avg_cosine_pos = pos_sims.mean()
            mask = ~torch.eye(B, dtype=torch.bool, device=z1.device)
            avg_cosine_neg = cos_matrix[mask].mean()

        return loss, avg_cosine_pos, avg_cosine_neg


# ---------------------------------------------------------------------------
# Supervised SimCSE Loss
# ---------------------------------------------------------------------------


class SupervisedSimCSELoss(nn.Module):
    """Supervised NT-Xent with hard negatives from NLI contradiction pairs.

    Args:
        temperature: softmax temperature tau (default 0.05)
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        encoder: SimCSEEncoder,
        anchor_ids: Tensor,
        positive_ids: Tensor,
        negative_ids: Tensor,
    ) -> tuple:
        """
        Args:
            encoder: SimCSEEncoder
            anchor_ids: (B, T)
            positive_ids: (B, T) entailment pairs
            negative_ids: (B, T) contradiction pairs
        Returns:
            (loss, pos_sim, neg_sim)
        """
        za = encoder.normalize(encoder(anchor_ids))
        zp = encoder.normalize(encoder(positive_ids))
        zn = encoder.normalize(encoder(negative_ids))

        B = za.size(0)

        keys = torch.cat([zp, zn], dim=0)  # (2B, D)
        sim = torch.mm(za, keys.t()) / self.temperature  # (B, 2B)
        labels = torch.arange(B, device=za.device)
        loss = F.cross_entropy(sim, labels)

        with torch.no_grad():
            cos_full = torch.mm(za, keys.t())
            pos_sim = cos_full[:, :B].diagonal().mean()
            neg_sim = cos_full[:, B:].diagonal().mean()

        return loss, pos_sim, neg_sim


# ---------------------------------------------------------------------------
# Alignment & Uniformity Loss (Wang & Isola 2020)
# ---------------------------------------------------------------------------


class AlignmentUniformityLoss(nn.Module):
    """Measures embedding space quality via alignment and uniformity.

    Args:
        t: Gaussian kernel width for uniformity (default 2.0)
    """

    def __init__(self, t: float = 2.0) -> None:
        super().__init__()
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")
        self.t = t

    def alignment(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Mean squared L2 distance between paired embeddings.

        Args:
            z1, z2: (B, D)
        Returns:
            scalar >= 0
        """
        return (z1 - z2).pow(2).sum(dim=-1).mean()

    def uniformity(self, z: Tensor) -> Tensor:
        """Log of mean Gaussian kernel over all pairs.

        Args:
            z: (B, D)
        Returns:
            scalar (typically negative)
        """
        sq_dist = torch.cdist(z, z, p=2).pow(2)
        return torch.log(torch.exp(-self.t * sq_dist).mean())

    def loss(
        self,
        z1: Tensor,
        z2: Tensor,
        align_weight: float = 1.0,
        uniform_weight: float = 1.0,
    ) -> Tensor:
        """Weighted sum of alignment and average uniformity.

        Args:
            z1, z2: (B, D) normalized positive pairs
            align_weight: weight for alignment term
            uniform_weight: weight for uniformity term
        Returns:
            scalar total loss
        """
        align = self.alignment(z1, z2)
        unif = (self.uniformity(z1) + self.uniformity(z2)) / 2.0
        return align_weight * align + uniform_weight * unif


# ---------------------------------------------------------------------------
# SimCSETrainer
# ---------------------------------------------------------------------------


class SimCSETrainer:
    """Training wrapper for SimCSE.

    Args:
        encoder: SimCSEEncoder
        optimizer: any torch optimizer wrapping encoder parameters
        mode: "unsupervised" or "supervised"
    """

    def __init__(
        self,
        encoder: SimCSEEncoder,
        optimizer: torch.optim.Optimizer,
        mode: str = "unsupervised",
    ) -> None:
        if mode not in ("unsupervised", "supervised"):
            raise ValueError(f"mode must be 'unsupervised' or 'supervised', got {mode!r}")
        self.encoder = encoder
        self.optimizer = optimizer
        self.mode = mode
        self._unsup_loss = UnsupervisedSimCSELoss()
        self._sup_loss = SupervisedSimCSELoss()

    def train_step(
        self,
        input_ids: Tensor,
        positive_ids: Tensor | None = None,
        negative_ids: Tensor | None = None,
    ) -> dict:
        """Run one gradient update step.

        Returns:
            dict with keys: "loss", "pos_sim", "neg_sim"
        """
        self.encoder.train()
        self.optimizer.zero_grad()

        if self.mode == "unsupervised":
            loss, pos_sim, neg_sim = self._unsup_loss(self.encoder, input_ids)
        else:
            if positive_ids is None or negative_ids is None:
                raise ValueError("supervised mode requires positive_ids and negative_ids")
            loss, pos_sim, neg_sim = self._sup_loss(
                self.encoder, input_ids, positive_ids, negative_ids
            )

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "pos_sim": pos_sim.item(),
            "neg_sim": neg_sim.item(),
        }

    def evaluate_sts(
        self,
        pairs: list,
        scores: list,
    ) -> float:
        """Evaluate on STS-style data via Spearman rank correlation.

        Args:
            pairs: list of (ids_a, ids_b) tensors
            scores: list of human similarity scores (floats)
        Returns:
            Spearman rho in [-1, 1]
        """
        if len(pairs) != len(scores):
            raise ValueError("pairs and scores must have the same length")
        if len(pairs) < 2:
            raise ValueError("Need at least 2 pairs for correlation")

        self.encoder.eval()
        cos_sims = []

        with torch.no_grad():
            for ids_a, ids_b in pairs:
                if ids_a.dim() == 1:
                    ids_a = ids_a.unsqueeze(0)
                if ids_b.dim() == 1:
                    ids_b = ids_b.unsqueeze(0)

                za = self.encoder.normalize(self.encoder(ids_a))
                zb = self.encoder.normalize(self.encoder(ids_b))
                cos_sim = (za * zb).sum(dim=-1).item()
                cos_sims.append(cos_sim)

        return _spearman(cos_sims, scores)


# ---------------------------------------------------------------------------
# Spearman rank correlation (stdlib only, no scipy)
# ---------------------------------------------------------------------------


def _rank(values: list) -> list:
    """Convert a list of values to average ranks (1-based)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: list, y: list) -> float:
    """Spearman rank correlation coefficient."""
    n = len(x)
    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_rx = math.sqrt(sum((r - mean_rx) ** 2 for r in rx))
    std_ry = math.sqrt(sum((r - mean_ry) ** 2 for r in ry))

    if std_rx == 0.0 or std_ry == 0.0:
        return 0.0
    return cov / (std_rx * std_ry)
