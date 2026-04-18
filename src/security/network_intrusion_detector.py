"""Transformer-based network intrusion and anomaly detector for the Aurelius LLM research platform.

Classifies sequential float feature vectors (e.g., network flow statistics) as
normal or anomalous using a compact transformer encoder with mean-pool aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IntrusionConfig:
    """Configuration for the intrusion detection transformer."""

    n_features: int = 16
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_classes: int = 2
    max_seq_len: int = 64
    dropout: float = 0.1


class FeatureProjector(nn.Module):
    """Projects raw feature vectors into the model's embedding space."""

    def __init__(self, config: IntrusionConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.n_features, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class IntrusionTransformer(nn.Module):
    """Transformer encoder that maps (B, T, n_features) to (B, n_classes) logits."""

    def __init__(self, config: IntrusionConfig) -> None:
        super().__init__()
        self.config = config
        self.projector = FeatureProjector(config)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.classifier = nn.Linear(config.d_model, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.projector(x) + self.pos_embedding(positions)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


class IntrusionDetector:
    """Thin wrapper around IntrusionTransformer providing inference utilities."""

    def __init__(self, config: IntrusionConfig) -> None:
        self.config = config
        self.model = IntrusionTransformer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, n_classes)."""
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices as a LongTensor of shape (B,)."""
        with torch.no_grad():
            logits = self.model(x)
        return logits.argmax(dim=-1)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probability of the anomalous class (class index 1), shape (B,)."""
        with torch.no_grad():
            logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]

    def train_step(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one forward/backward pass and return the scalar loss value."""
        self.model.train()
        optimizer.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        self.model.eval()
        return loss.item()
