"""Dense retriever training (DPR-style) for retrieval-augmented generation.

Implements DenseEncoder, in-batch negatives contrastive loss, and DPRTrainer
for training bi-encoder dense retrieval models on top of AureliusTransformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RetrieverConfig:
    """Configuration for dense retriever training."""

    embedding_dim: int = 64
    """Output embedding dimension after projection."""

    temperature: float = 0.07
    """Temperature scaling for contrastive loss."""

    n_negatives: int = 4
    """Number of in-batch negatives per query (informational; actual negatives = batch_size - 1)."""

    pooling: str = "cls"
    """Pooling strategy: 'cls' (first token) or 'mean' (average over sequence)."""


class DenseEncoder(nn.Module):
    """Encodes text to dense L2-normalized vectors using a transformer backbone.

    Uses a forward hook on the last transformer layer to extract hidden states,
    then pools and projects to the target embedding dimension.
    """

    def __init__(self, backbone: nn.Module, config: RetrieverConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config

        # Determine backbone hidden dim
        if hasattr(backbone, "config"):
            d_model = backbone.config.d_model
        else:
            d_model = backbone.embed.weight.shape[1]

        self.proj = nn.Linear(d_model, config.embedding_dim, bias=False)

        # Storage for hook output
        self._hidden: Tensor | None = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self) -> None:
        """Register a forward hook on the last transformer layer."""
        last_layer = self.backbone.layers[-1]

        def _hook(module, input, output):
            # output may be (hidden, kv) tuple; always take first element
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self._hidden = hidden

        self._hook_handle = last_layer.register_forward_hook(_hook)

    def _pool(self, hidden: Tensor) -> Tensor:
        """Pool hidden states to a single vector per sequence.

        Args:
            hidden: (B, S, d_model)

        Returns:
            (B, d_model)
        """
        if self.config.pooling == "cls":
            return hidden[:, 0, :]
        elif self.config.pooling == "mean":
            return hidden.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling!r}")

    def encode(self, input_ids: Tensor) -> Tensor:
        """Encode token ids to L2-normalized dense vectors.

        Args:
            input_ids: (B, S) integer token ids.

        Returns:
            (B, embedding_dim) L2-normalized embedding vectors.
        """
        self._hidden = None
        # Run backbone -- returns (loss, logits, pkv) plain tuple
        _ = self.backbone(input_ids)

        if self._hidden is None:
            raise RuntimeError("Hook did not capture hidden states.")

        pooled = self._pool(self._hidden)  # (B, d_model)
        projected = self.proj(pooled)  # (B, embedding_dim)
        return F.normalize(projected, dim=-1)  # L2-normalize

    def forward(self, input_ids: Tensor) -> Tensor:
        """Same as encode."""
        return self.encode(input_ids)


def in_batch_negatives_loss(
    query_embs: Tensor,
    pos_embs: Tensor,
    temperature: float,
) -> Tensor:
    """DPR-style in-batch negatives contrastive loss.

    For each query, the positive is its paired passage. All other positives
    in the batch serve as negatives (in-batch negatives). Uses cross-entropy
    over dot-product similarities scaled by temperature.

    Args:
        query_embs: (B, D) L2-normalized query embeddings.
        pos_embs:   (B, D) L2-normalized positive passage embeddings.
        temperature: Temperature scaling scalar.

    Returns:
        Scalar cross-entropy loss.
    """
    # Similarity matrix: (B, B)
    logits = torch.matmul(query_embs, pos_embs.T) / temperature  # (B, B)

    # Labels: diagonal -- query i matches passage i
    labels = torch.arange(logits.size(0), device=logits.device)

    return F.cross_entropy(logits, labels)


class DPRTrainer:
    """Trains query and passage encoders using DPR-style in-batch negatives."""

    def __init__(
        self,
        query_encoder: DenseEncoder,
        passage_encoder: DenseEncoder,
        config: RetrieverConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.config = config
        self.optimizer = optimizer

    def train_step(self, query_ids: Tensor, passage_ids: Tensor) -> dict:
        """Encode queries and passages, compute loss, and update parameters.

        Args:
            query_ids:   (B, S) query token ids.
            passage_ids: (B, S) positive passage token ids.

        Returns:
            dict with keys "loss" (float) and "sim_pos_mean" (float).
        """
        self.query_encoder.train()
        self.passage_encoder.train()
        self.optimizer.zero_grad()

        q_embs = self.query_encoder(query_ids)  # (B, D)
        p_embs = self.passage_encoder(passage_ids)  # (B, D)

        loss = in_batch_negatives_loss(q_embs, p_embs, self.config.temperature)

        loss.backward()
        self.optimizer.step()

        # Mean similarity of positive pairs (diagonal)
        with torch.no_grad():
            sim_pos = (q_embs * p_embs).sum(dim=-1).mean().item()

        return {"loss": loss.item(), "sim_pos_mean": sim_pos}

    @torch.no_grad()
    def retrieve(
        self,
        query_ids: Tensor,
        passage_ids_list: list[Tensor],
        top_k: int,
    ) -> list[int]:
        """Score a query against all passages and return top-k indices.

        Args:
            query_ids:        (1, S) or (S,) query token ids.
            passage_ids_list: List of tensors, each (1, S) or (S,).
            top_k:            Number of top passages to return.

        Returns:
            List of int indices (length top_k) into passage_ids_list.
        """
        self.query_encoder.eval()
        self.passage_encoder.eval()

        if query_ids.dim() == 1:
            query_ids = query_ids.unsqueeze(0)

        q_emb = self.query_encoder(query_ids)  # (1, D)

        scores: list[float] = []
        for p_ids in passage_ids_list:
            if p_ids.dim() == 1:
                p_ids = p_ids.unsqueeze(0)
            p_emb = self.passage_encoder(p_ids)  # (1, D)
            score = (q_emb * p_emb).sum(dim=-1).item()
            scores.append(score)

        top_k = min(top_k, len(scores))
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:top_k]
