"""Contrastive dense-embedding trainer for Aurelius retrieval.

Implements the SimCSE / DPR-style training recipe for a small bi-encoder:
a transformer encoder produces mean-pooled, L2-normalized sentence
embeddings, and an InfoNCE loss with in-batch negatives pulls matched
anchor/positive pairs together while pushing unmatched pairs apart.

References:
    - Gao, Yao, Chen (2021). "SimCSE: Simple Contrastive Learning of
      Sentence Embeddings." arXiv:2104.08821.
    - Karpukhin et al. (2020). "Dense Passage Retrieval for Open-Domain
      Question Answering." arXiv:2004.04906.

The architecture is intentionally self-contained pure PyTorch, with zero
coupling to ``src.model``: the retrieval surface must be importable in a
hermetic subprocess without pulling in the generative transformer stack.

Public surface:
    - EmbedderConfig
    - DenseEmbedder
    - InfoNCELoss
    - EmbeddingTrainer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmbedderConfig:
    """Configuration for :class:`DenseEmbedder`.

    Defaults target a tiny, test-friendly encoder. Production configs will
    override these via YAML. Invalid configurations raise ``ValueError``:
    there are no silent fallbacks.
    """

    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_seq_len: int = 64
    dropout: float = 0.0
    pad_token_id: int = 0
    embed_dim: int = 64

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.pad_token_id < 0 or self.pad_token_id >= self.vocab_size:
            raise ValueError(
                f"pad_token_id ({self.pad_token_id}) out of range [0, {self.vocab_size})"
            )
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")


class DenseEmbedder(nn.Module):
    """Small transformer-encoder bi-encoder producing L2-normalized vectors.

    Pipeline:
        token_ids -> token+pos embeddings -> N x TransformerEncoderLayer
        -> mean-pool over non-pad positions -> linear projection
        -> L2 normalize.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__()
        if not isinstance(config, EmbedderConfig):
            raise TypeError(
                f"config must be EmbedderConfig, got {type(config).__name__}"
            )
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_norm = nn.LayerNorm(config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.projection = nn.Linear(config.d_model, config.embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode token ids into L2-normalized embeddings.

        Args:
            input_ids: [B, T] long tensor of token ids.
            attention_mask: Optional [B, T] bool/int tensor where 1 marks
                real tokens and 0 marks padding. If ``None``, positions
                equal to ``pad_token_id`` are treated as padding.

        Returns:
            [B, embed_dim] float tensor with unit L2 norm per row.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D [B, T], got shape {tuple(input_ids.shape)}"
            )
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).to(
                dtype=torch.long
            )
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask shape "
                    f"{tuple(attention_mask.shape)} must match input_ids "
                    f"{tuple(input_ids.shape)}"
                )
            attention_mask = attention_mask.to(dtype=torch.long)

        positions = torch.arange(
            seq_len, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        # ``src_key_padding_mask`` is True at positions to IGNORE.
        key_padding_mask = attention_mask == 0
        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean-pool over non-pad tokens. If a row has zero real tokens, we
        # fall back to a zero vector (which will then be re-projected and
        # re-normalized); guard the denominator against division by zero.
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)  # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)  # [B, d_model]
        counts = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = summed / counts

        projected = self.projection(pooled)
        normalized = F.normalize(projected, p=2, dim=-1, eps=1e-12)
        return normalized


class InfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE contrastive loss.

    For anchor/positive pairs with unit-norm embeddings of shape [B, D],
    the similarity matrix S = A @ P^T / tau has targets on the diagonal.
    The symmetric loss averages anchor->positive and positive->anchor
    cross-entropies, matching the DPR / SimCSE formulation.

    With B=1 there are no in-batch negatives so the loss degenerates to
    0 (the single positive is the only class).
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor
    ) -> torch.Tensor:
        if anchor.dim() != 2 or positive.dim() != 2:
            raise ValueError(
                "anchor and positive must be 2-D [B, D], got "
                f"{tuple(anchor.shape)} and {tuple(positive.shape)}"
            )
        if anchor.shape != positive.shape:
            raise ValueError(
                f"anchor shape {tuple(anchor.shape)} must match "
                f"positive shape {tuple(positive.shape)}"
            )
        batch_size = anchor.shape[0]
        logits = anchor @ positive.transpose(0, 1) / self.temperature
        targets = torch.arange(batch_size, device=anchor.device)
        loss_a = F.cross_entropy(logits, targets)
        loss_b = F.cross_entropy(logits.transpose(0, 1), targets)
        return 0.5 * (loss_a + loss_b)

    # Allow InfoNCELoss(...) instances to be called directly without
    # worrying about nn.Module forward hooks in simple trainer loops.
    __call__ = forward  # type: ignore[assignment]


class EmbeddingTrainer:
    """Thin training harness around an embedder + optimizer + loss.

    Keeps the training primitive (``train_step``) deterministic and free
    of hidden global state; schedulers, AMP, and DDP wrappers are the
    caller's responsibility.
    """

    def __init__(
        self,
        embedder: DenseEmbedder,
        optimizer: torch.optim.Optimizer,
        loss_fn: InfoNCELoss,
    ) -> None:
        if not isinstance(embedder, DenseEmbedder):
            raise TypeError(
                f"embedder must be DenseEmbedder, got {type(embedder).__name__}"
            )
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer must be torch.optim.Optimizer, got "
                f"{type(optimizer).__name__}"
            )
        if not isinstance(loss_fn, InfoNCELoss):
            raise TypeError(
                f"loss_fn must be InfoNCELoss, got {type(loss_fn).__name__}"
            )
        self.embedder = embedder
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(
        self,
        anchor_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run a single gradient-descent step on one paired batch."""
        if anchor_ids.shape[0] != positive_ids.shape[0]:
            raise ValueError(
                f"anchor batch {anchor_ids.shape[0]} must match positive "
                f"batch {positive_ids.shape[0]}"
            )
        self.embedder.train()
        self.optimizer.zero_grad(set_to_none=True)
        a = self.embedder(anchor_ids, anchor_mask)
        p = self.embedder(positive_ids, positive_mask)
        loss = self.loss_fn(a, p)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().item())}


__all__ = [
    "EmbedderConfig",
    "DenseEmbedder",
    "InfoNCELoss",
    "EmbeddingTrainer",
]
