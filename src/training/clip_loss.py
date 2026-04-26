"""CLIP-style contrastive image-text alignment loss and training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CLIPConfig:
    """Configuration for CLIP-style image-text contrastive learning."""

    temperature: float = 0.07  # learnable temperature (log scale internally)
    embed_dim: int = 128  # joint embedding space dimension
    d_vision: int = 256  # vision encoder output dim
    d_text: int = 64  # text encoder output dim
    normalize: bool = True  # L2 normalize embeddings


# ---------------------------------------------------------------------------
# LearnableTemperature
# ---------------------------------------------------------------------------


class LearnableTemperature(nn.Module):
    """Learnable temperature parameter stored in log-space for numerical stability."""

    def __init__(self, init_temp: float = 0.07) -> None:
        super().__init__()
        self.log_temp: nn.Parameter = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self) -> Tensor:
        """Return temperature as exp(log_temp) — always positive."""
        return self.log_temp.exp()

    def clamp_(self, min_temp: float = 0.01, max_temp: float = 100.0) -> None:
        """Clamp log_temp in log-space to bound the temperature."""
        with torch.no_grad():
            self.log_temp.clamp_(
                min=torch.log(torch.tensor(min_temp)).item(),
                max=torch.log(torch.tensor(max_temp)).item(),
            )


# ---------------------------------------------------------------------------
# CLIPProjection
# ---------------------------------------------------------------------------


class CLIPProjection(nn.Module):
    """Project encoder outputs into a shared joint embedding space."""

    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Project and normalize input features.

        Args:
            x: (B, D_in) encoder output.

        Returns:
            (B, embed_dim) projected features.
        """
        x = self.proj(x)  # (B, embed_dim)
        x = self.norm(x)  # (B, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Functional: clip_contrastive_loss
# ---------------------------------------------------------------------------


def clip_contrastive_loss(
    image_embeds: Tensor,
    text_embeds: Tensor,
    temperature: float | Tensor,
) -> Tensor:
    """Symmetric CLIP contrastive loss.

    Args:
        image_embeds: (B, D) L2-normalized image embeddings.
        text_embeds:  (B, D) L2-normalized text embeddings.
        temperature:  scalar float or Tensor divisor.

    Returns:
        Scalar loss.
    """
    B = image_embeds.shape[0]
    labels = torch.arange(B, device=image_embeds.device)

    sim = image_embeds @ text_embeds.T / temperature  # (B, B)

    loss_i2t = F.cross_entropy(sim, labels)
    loss_t2i = F.cross_entropy(sim.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


# ---------------------------------------------------------------------------
# Functional: compute_retrieval_metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    image_embeds: Tensor,
    text_embeds: Tensor,
) -> dict[str, float]:
    """Compute image-text and text-image retrieval recall metrics.

    Args:
        image_embeds: (B, D) L2-normalized image embeddings.
        text_embeds:  (B, D) L2-normalized text embeddings.

    Returns:
        Dict with keys i2t_r1, i2t_r5, t2i_r1, t2i_r5, mean_r1.
    """
    with torch.no_grad():
        B = image_embeds.shape[0]
        sim = image_embeds @ text_embeds.T  # (B, B)
        labels = torch.arange(B, device=image_embeds.device)

        def recall_at_k(similarity: Tensor, k: int) -> float:
            # For each row, check if the correct index appears in top-k
            topk_indices = similarity.topk(min(k, B), dim=1).indices  # (B, k)
            correct = (topk_indices == labels.unsqueeze(1)).any(dim=1)
            return correct.float().mean().item()

        i2t_r1 = recall_at_k(sim, 1)
        i2t_r5 = recall_at_k(sim, 5)
        t2i_r1 = recall_at_k(sim.T, 1)
        t2i_r5 = recall_at_k(sim.T, 5)
        mean_r1 = (i2t_r1 + t2i_r1) / 2.0

    return {
        "i2t_r1": i2t_r1,
        "i2t_r5": i2t_r5,
        "t2i_r1": t2i_r1,
        "t2i_r5": t2i_r5,
        "mean_r1": mean_r1,
    }


# ---------------------------------------------------------------------------
# CLIPModel
# ---------------------------------------------------------------------------


class CLIPModel(nn.Module):
    """CLIP-style model with vision and text encoders and learnable temperature."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        config: CLIPConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.config = config

        self.vision_proj = CLIPProjection(config.d_vision, config.embed_dim)
        self.text_proj = CLIPProjection(config.d_text, config.embed_dim)
        self.temperature = LearnableTemperature(init_temp=config.temperature)

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode images to joint embedding space.

        Args:
            images: (B, ...) input images or features for the vision encoder.

        Returns:
            (B, embed_dim) image embeddings, L2-normalized if config.normalize.
        """
        feats = self.vision_encoder(images)  # (B, d_vision)
        embeds = self.vision_proj(feats)  # (B, embed_dim)
        if self.config.normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds

    def encode_text(self, token_embeddings: Tensor) -> Tensor:
        """Encode token embeddings to joint embedding space.

        Args:
            token_embeddings: (B, T, D) token-level embeddings.

        Returns:
            (B, embed_dim) text embeddings, L2-normalized if config.normalize.
        """
        # Mean pool over sequence dimension
        pooled = token_embeddings.mean(dim=1)  # (B, D)
        embeds = self.text_proj(pooled)  # (B, embed_dim)
        if self.config.normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds

    def forward(self, images: Tensor, token_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Encode both modalities.

        Args:
            images:            (B, ...) image inputs.
            token_embeddings:  (B, T, D) text token embeddings.

        Returns:
            (image_embeds, text_embeds) each of shape (B, embed_dim).
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(token_embeddings)
        return image_embeds, text_embeds

    def compute_loss(
        self,
        images: Tensor,
        token_embeddings: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Forward pass and contrastive loss computation.

        Args:
            images:           (B, ...) image inputs.
            token_embeddings: (B, T, D) text token embeddings.

        Returns:
            (loss, metrics) where metrics contains contrastive_loss and temperature.
        """
        image_embeds, text_embeds = self.forward(images, token_embeddings)
        temp = self.temperature()
        loss = clip_contrastive_loss(image_embeds, text_embeds, temp)
        metrics = {
            "contrastive_loss": loss.item(),
            "temperature": temp.item(),
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# CLIPTrainer
# ---------------------------------------------------------------------------


class CLIPTrainer:
    """Training wrapper for CLIPModel with one-step train interface."""

    def __init__(self, model: CLIPModel, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def train_step(self, images: Tensor, token_embeddings: Tensor) -> dict[str, float]:
        """One training step: forward, loss, backward, optimizer step.

        Args:
            images:           (B, ...) image inputs.
            token_embeddings: (B, T, D) text token embeddings.

        Returns:
            Dict with keys 'loss' and 'temperature'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = self.model.compute_loss(images, token_embeddings)
        loss.backward()
        self.optimizer.step()

        # Clamp temperature after update
        self.model.temperature.clamp_()

        return {
            "loss": metrics["contrastive_loss"],
            "temperature": self.model.temperature().item(),
        }
