"""Contrastive learning for text representations (SimCSE-style)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ContrastiveConfig:
    temperature: float = 0.05  # SimCSE temperature
    pooling: str = "mean"  # "mean" | "cls" | "last"
    projection_dim: int | None = None  # optional projection head dim
    hard_negative_weight: float = 0.0  # weight for hard negatives


def pool_hidden_states(hidden: Tensor, pooling: str) -> Tensor:
    """Pool hidden states to get sequence representation.

    hidden: (B, T, d_model)
    pooling: "mean" -> mean over T, "cls" -> hidden[:, 0, :], "last" -> hidden[:, -1, :]
    Returns (B, d_model).
    """
    if pooling == "mean":
        return hidden.mean(dim=1)
    elif pooling == "cls":
        return hidden[:, 0, :]
    elif pooling == "last":
        return hidden[:, -1, :]
    else:
        raise ValueError(
            f"Unknown pooling strategy: {pooling!r}. Choose from 'mean', 'cls', 'last'."
        )


def simcse_loss(
    z1: Tensor,
    z2: Tensor,
    temperature: float,
    z_hard_neg: Tensor | None = None,
    hard_neg_weight: float = 0.0,
) -> tuple[Tensor, dict]:
    """In-batch contrastive loss (InfoNCE).

    For each anchor z1[i], positive is z2[i], negatives are z2[j!=i].
    sim(i,j) = dot(z1[i], z2[j]) / (||z1[i]|| * ||z2[j]|| * temperature)
    loss = CE(sim[i], label=i) averaged over batch.

    Returns (loss, metrics) where metrics has:
        'alignment': float — mean cosine sim of positive pairs
        'uniformity': float — log mean pairwise exp(-2*||z1[i]-z1[j]||^2) (lower is better)
        'accuracy': float — fraction where positive pair has highest similarity
    """
    B = z1.size(0)

    # L2-normalise
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)

    # Cosine similarity matrix: (B, B)
    sim_matrix = torch.mm(z1_norm, z2_norm.t()) / temperature  # (B, B)

    if z_hard_neg is not None and hard_neg_weight > 0.0:
        z_hn_norm = F.normalize(z_hard_neg, dim=-1)
        # sim to hard negatives: (B, 1)
        sim_hard = (z1_norm * z_hn_norm).sum(dim=-1, keepdim=True) / temperature
        # Concat: (B, B+1) — positives at diagonal, hard neg as extra column
        sim_matrix_aug = torch.cat([sim_matrix, sim_hard * hard_neg_weight], dim=1)
        labels = torch.arange(B, device=z1.device)
        loss = F.cross_entropy(sim_matrix_aug, labels)
    else:
        labels = torch.arange(B, device=z1.device)
        loss = F.cross_entropy(sim_matrix, labels)

    # --- Metrics ---
    # Alignment: mean cosine similarity of positive pairs (un-scaled)
    alignment = (z1_norm * z2_norm).sum(dim=-1).mean().item()

    # Uniformity: log mean pairwise Gaussian kernel on z1
    sq_dists = torch.pdist(z1_norm, p=2).pow(2)
    uniformity = torch.log(torch.exp(-2.0 * sq_dists).mean() + 1e-10).item()

    # Accuracy: fraction of samples where diagonal is the argmax
    with torch.no_grad():
        preds = sim_matrix.argmax(dim=1)  # use original sim_matrix for eval
        accuracy = (preds == labels).float().mean().item()

    metrics = {
        "alignment": alignment,
        "uniformity": uniformity,
        "accuracy": accuracy,
    }
    return loss, metrics


class TextEncoder(nn.Module):
    """Encodes token sequences to dense representations via backbone + pooling."""

    def __init__(
        self,
        backbone: nn.Module,
        config: ContrastiveConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config

        # Detect d_model from backbone config if available
        d_model = getattr(getattr(backbone, "config", None), "d_model", None)
        if d_model is None:
            # Fallback: inspect the embed layer
            d_model = backbone.embed.embedding_dim

        self.d_model = d_model

        # Optional linear projection head
        if config.projection_dim is not None:
            self.projection: nn.Module = nn.Sequential(
                nn.Linear(d_model, config.projection_dim, bias=False),
                nn.ReLU(),
                nn.Linear(config.projection_dim, config.projection_dim, bias=False),
            )
        else:
            self.projection = None  # type: ignore[assignment]

    def _get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Walk the backbone manually to extract hidden states before lm_head.

        Returns (B, T, d_model).
        """
        backbone = self.backbone
        B, S = input_ids.shape

        x = backbone.embed(input_ids)

        # Slice RoPE frequencies to sequence length
        freqs_cis = backbone.freqs_cis[:S]

        for layer in backbone.layers:
            x, _ = layer(x, freqs_cis, mask=None, past_kv=None)

        x = backbone.norm(x)
        return x  # (B, T, d_model)

    def encode(self, input_ids: Tensor) -> Tensor:
        """input_ids (B, T) -> embeddings (B, d_model or projection_dim)."""
        hidden = self._get_hidden_states(input_ids)  # (B, T, d_model)
        pooled = pool_hidden_states(hidden, self.config.pooling)  # (B, d_model)

        if self.projection is not None:
            pooled = self.projection(pooled)

        return pooled

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.encode(input_ids)


class SimCSETrainer:
    """SimCSE training: same sentence twice through backbone (different dropout masks)."""

    def __init__(
        self,
        encoder: TextEncoder,
        config: ContrastiveConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.encoder = encoder
        self.config = config
        self.optimizer = optimizer

    def train_step(self, input_ids: Tensor) -> dict:
        """Pass input_ids through encoder twice (train mode -> different dropout).

        Compute simcse_loss on the two sets of embeddings.
        Returns dict with: 'loss', 'alignment', 'uniformity', 'accuracy'
        """
        self.encoder.train()
        self.optimizer.zero_grad()

        z1 = self.encoder.encode(input_ids)
        z2 = self.encoder.encode(input_ids)

        loss, metrics = simcse_loss(
            z1,
            z2,
            temperature=self.config.temperature,
        )

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            **metrics,
        }


class HardNegativeSimCSETrainer(SimCSETrainer):
    """SimCSE with explicit hard negatives."""

    def train_step_with_negatives(
        self,
        input_ids: Tensor,  # (B, T) anchors
        neg_ids: Tensor,  # (B, T) hard negatives
    ) -> dict:
        """Include hard negatives in the contrastive loss.

        Returns same metrics as SimCSETrainer.train_step.
        """
        self.encoder.train()
        self.optimizer.zero_grad()

        z1 = self.encoder.encode(input_ids)
        z2 = self.encoder.encode(input_ids)
        z_hard_neg = self.encoder.encode(neg_ids)

        loss, metrics = simcse_loss(
            z1,
            z2,
            temperature=self.config.temperature,
            z_hard_neg=z_hard_neg,
            hard_neg_weight=self.config.hard_negative_weight,
        )

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            **metrics,
        }
