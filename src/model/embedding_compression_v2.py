"""Embedding compression v2: product quantization, mixed-precision, token-level dropout.

Implements alternative embedding compression strategies with the new EmbedConfig API.
V2 because embedding_compression.py already exists with a different API.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EmbedConfigV2:
    vocab_size: int = 50257
    d_model: int = 512
    d_embed: int = 128
    n_codebooks: int = 4
    n_codes: int = 256
    tie_weights: bool = True


class FactorizedEmbeddingV2(nn.Module):
    """Two-matrix factorization: E∈(vocab, d_embed) @ P∈(d_embed, d_model)."""

    def __init__(self, vocab_size: int, d_embed: int, d_model: int) -> None:
        super().__init__()
        self.E = nn.Embedding(vocab_size, d_embed)
        self.P = nn.Linear(d_embed, d_model, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.P(self.E(token_ids))

    def parameter_count(self) -> int:
        vocab_size, d_embed = self.E.weight.shape
        d_model = self.P.weight.shape[0]
        return vocab_size * d_embed + d_embed * d_model


class ProductQuantizedEmbedding(nn.Module):
    """PQ embedding: n_codebooks × n_codes × (d_model // n_codebooks)."""

    def __init__(self, vocab_size: int, d_model: int, n_codebooks: int, n_codes: int) -> None:
        super().__init__()
        assert d_model % n_codebooks == 0, "d_model must be divisible by n_codebooks"
        self.n_codebooks = n_codebooks
        self.n_codes = n_codes
        self.sub_dim = d_model // n_codebooks
        # Learnable codebooks
        self.codebooks = nn.Parameter(torch.randn(n_codebooks, n_codes, self.sub_dim))
        # Fixed integer codes per token (registered as buffer, not trained)
        codes = torch.randint(0, n_codes, (vocab_size, n_codebooks))
        self.register_buffer("codes", codes)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (B, T)
        token_codes = self.codes[token_ids]  # (B, T, n_codebooks)
        out = []
        for cb in range(self.n_codebooks):
            idx = token_codes[..., cb]  # (B, T)
            sub = self.codebooks[cb][idx]  # (B, T, sub_dim)
            out.append(sub)
        return torch.cat(out, dim=-1)  # (B, T, d_model)

    def parameter_count(self) -> int:
        return self.n_codebooks * self.n_codes * self.sub_dim


def embed_dropout(embeddings: Tensor, p: float, training: bool) -> Tensor:
    """Token-level dropout: zero entire embedding vectors with probability p."""
    if not training or p == 0.0:
        return embeddings
    # (B, T, d) → mask shape (B, T, 1) broadcast over d
    mask = torch.bernoulli(torch.full(embeddings.shape[:-1], 1 - p, device=embeddings.device))
    mask = mask.unsqueeze(-1)
    return embeddings * mask / (1 - p + 1e-8)


def compute_embedding_norm(embedding_weight: Tensor) -> Tensor:
    """(vocab_size, d_model) → (vocab_size,) L2 norm per row."""
    return embedding_weight.norm(dim=-1)


class MixedPrecisionEmbedding(nn.Module):
    """Stores embedding weight in reduced precision, upcasts on forward."""

    def __init__(self, vocab_size: int, d_model: int, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.storage_dtype = dtype
        # Store in reduced precision
        self.embed.weight.data = self.embed.weight.data.to(dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        # Upcast to float32 for computation
        weight = self.embed.weight.float()
        return F.embedding(token_ids, weight)
