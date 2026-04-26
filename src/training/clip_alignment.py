"""CLIP-style InfoNCE contrastive alignment for Aurelius text embeddings.

Aligns Aurelius text embeddings with external modality embeddings using
symmetric InfoNCE loss, as in CLIP (Radford et al., 2021).

Symmetric InfoNCE:
    similarity = text_embeds @ other_embeds.T / temperature   (B, B)
    labels     = torch.arange(B)  (diagonal = positive pairs)
    loss       = (CE(similarity, labels) + CE(similarity.T, labels)) / 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CLIPAlignmentConfig:
    """Configuration for CLIP-style contrastive alignment."""

    temperature_init: float = 0.07
    temperature_learnable: bool = True
    temperature_min: float = 0.01
    temperature_max: float = 100.0
    embedding_dim: int = 256
    dropout: float = 0.0
    normalize_embeddings: bool = True


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def clip_loss(
    text_embeds: Tensor,
    other_embeds: Tensor,
    log_temperature: Tensor,
) -> Tensor:
    """Symmetric InfoNCE contrastive loss.

    Args:
        text_embeds:     (B, D) L2-normalized text embeddings.
        other_embeds:    (B, D) L2-normalized other-modality embeddings.
        log_temperature: scalar log-scale temperature; exp() applied internally.

    Returns:
        Scalar loss tensor.
    """
    B = text_embeds.shape[0]
    temp = log_temperature.exp()

    sim = text_embeds @ other_embeds.T / temp  # (B, B)
    labels = torch.arange(B, device=text_embeds.device)

    loss_t2o = F.cross_entropy(sim, labels)
    loss_o2t = F.cross_entropy(sim.T, labels)
    return (loss_t2o + loss_o2t) / 2.0


def contrastive_accuracy(
    text_embeds: Tensor,
    other_embeds: Tensor,
    temperature: float = 0.07,
) -> float:
    """Fraction of samples where the diagonal pair has maximum cosine similarity.

    Args:
        text_embeds:  (B, D) normalized.
        other_embeds: (B, D) normalized.
        temperature:  scalar divisor for similarity scaling.

    Returns:
        Float in [0, 1].
    """
    with torch.no_grad():
        sim = text_embeds @ other_embeds.T / temperature  # (B, B)
        preds = sim.argmax(dim=1)
        labels = torch.arange(sim.shape[0], device=sim.device)
        correct = (preds == labels).float().mean()
    return correct.item()


# ---------------------------------------------------------------------------
# CLIPAlignmentLayer
# ---------------------------------------------------------------------------


def _build_projection_head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Two-layer MLP projection head: Linear -> GELU -> Linear."""
    hidden = (in_dim + out_dim) // 2
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden, bias=False),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim, bias=False))
    return nn.Sequential(*layers)


class CLIPAlignmentLayer(nn.Module):
    """Projects text and modality embeddings into a shared space for contrastive training.

    Wraps two projection heads (text_proj and modality_proj) with a learnable temperature.
    """

    def __init__(
        self,
        text_dim: int,
        modality_dim: int,
        cfg: CLIPAlignmentConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.text_proj = _build_projection_head(text_dim, cfg.embedding_dim, cfg.dropout)
        self.modality_proj = _build_projection_head(modality_dim, cfg.embedding_dim, cfg.dropout)

        log_temp_val = torch.tensor(math.log(cfg.temperature_init))
        if cfg.temperature_learnable:
            self.log_temperature = nn.Parameter(log_temp_val)
        else:
            self.register_buffer("log_temperature", log_temp_val)

    @property
    def temperature(self) -> Tensor:
        """exp(log_temperature) — always positive."""
        return self.log_temperature.exp()

    def project_text(self, text_hidden: Tensor) -> Tensor:
        """Project text hidden states into shared embedding space.

        Args:
            text_hidden: (B, text_dim)

        Returns:
            (B, embedding_dim) L2-normalized.
        """
        out = self.text_proj(text_hidden)
        if self.cfg.normalize_embeddings:
            out = F.normalize(out, dim=-1)
        return out

    def project_modality(self, modality_features: Tensor) -> Tensor:
        """Project external modality features into shared embedding space.

        Args:
            modality_features: (B, modality_dim)

        Returns:
            (B, embedding_dim) L2-normalized.
        """
        out = self.modality_proj(modality_features)
        if self.cfg.normalize_embeddings:
            out = F.normalize(out, dim=-1)
        return out

    def forward(
        self,
        text_hidden: Tensor,
        modality_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project both modalities and compute symmetric InfoNCE loss.

        Args:
            text_hidden:       (B, text_dim) e.g. mean-pooled LM hidden states.
            modality_features: (B, modality_dim) external features.

        Returns:
            (text_proj, modality_proj, loss)
        """
        text_proj = self.project_text(text_hidden)
        mod_proj = self.project_modality(modality_features)
        loss = clip_loss(text_proj, mod_proj, self.log_temperature)
        return text_proj, mod_proj, loss


# ---------------------------------------------------------------------------
# CLIPAlignmentTrainer
# ---------------------------------------------------------------------------


class CLIPAlignmentTrainer:
    """Trainer that aligns Aurelius text embeddings with external modality features."""

    def __init__(
        self,
        model: nn.Module,
        alignment_layer: CLIPAlignmentLayer,
        optimizer: torch.optim.Optimizer,
        cfg: CLIPAlignmentConfig,
    ) -> None:
        self.model = model
        self.alignment_layer = alignment_layer
        self.optimizer = optimizer
        self.cfg = cfg

    def get_text_embeddings(self, input_ids: Tensor) -> Tensor:
        """Forward model and mean-pool last hidden states.

        Args:
            input_ids: (B, S) tokenized text.

        Returns:
            (B, d_model) mean-pooled hidden states.
        """
        B, S = input_ids.shape
        x = self.model.embed(input_ids)
        freqs_cis = self.model.freqs_cis[:S]
        for layer in self.model.layers:
            x, _ = layer(x, freqs_cis, mask=None, past_kv=None)
        x = self.model.norm(x)  # (B, S, d_model)
        return x.mean(dim=1)  # (B, d_model)

    def train_step(
        self,
        input_ids: Tensor,
        modality_features: Tensor,
    ) -> dict[str, float]:
        """One contrastive alignment training step.

        Args:
            input_ids:         (B, S) tokenized text.
            modality_features: (B, modality_dim) external features.

        Returns:
            Dict with keys 'loss', 'accuracy', 'temperature'.
        """
        self.alignment_layer.train()
        self.optimizer.zero_grad()

        text_hidden = self.get_text_embeddings(input_ids)
        text_proj, mod_proj, loss = self.alignment_layer(text_hidden, modality_features)

        loss.backward()
        self.optimizer.step()

        acc = contrastive_accuracy(
            text_proj.detach(),
            mod_proj.detach(),
            temperature=self.alignment_layer.temperature.item(),
        )

        return {
            "loss": loss.item(),
            "accuracy": acc,
            "temperature": self.alignment_layer.temperature.item(),
        }

    def freeze_backbone(self) -> None:
        """Freeze model parameters, only train alignment_layer."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()


# ---------------------------------------------------------------------------
# Hard Negative Mining
# ---------------------------------------------------------------------------


def hard_negative_mining(
    text_embeds: Tensor,
    other_embeds: Tensor,
    n_hard_negatives: int = 1,
) -> tuple[Tensor, Tensor]:
    """Mine hard negatives from the batch for augmented contrastive training.

    For each anchor, finds the closest (highest cosine similarity) non-matching
    embeddings as hard negatives.

    Args:
        text_embeds:      (B, D) L2-normalized.
        other_embeds:     (B, D) L2-normalized.
        n_hard_negatives: number of hard negatives per sample.

    Returns:
        (hard_neg_text, hard_neg_other) each of shape (B * n_hard_negatives, D).
    """
    B, D = text_embeds.shape

    sim = text_embeds @ other_embeds.T  # (B, B)

    # Mask diagonal (positive pairs)
    mask = torch.eye(B, dtype=torch.bool, device=sim.device)
    sim_masked = sim.masked_fill(mask, float("-inf"))

    # Hard other embeddings for each text anchor
    _, top_other_idx = sim_masked.topk(n_hard_negatives, dim=1)  # (B, n_hard)
    hard_neg_other = other_embeds[top_other_idx.reshape(-1)]  # (B*n_hard, D)

    # Hard text embeddings for each other anchor
    _, top_text_idx = sim_masked.T.topk(n_hard_negatives, dim=1)  # (B, n_hard)
    hard_neg_text = text_embeds[top_text_idx.reshape(-1)]  # (B*n_hard, D)

    return hard_neg_text, hard_neg_other
