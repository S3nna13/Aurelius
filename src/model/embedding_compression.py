"""Embedding compression: factorize large embedding tables, share weights, and reduce memory footprint."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EmbedConfig:
    """Configuration for embedding compression techniques."""
    vocab_size: int = 32000
    d_model: int = 512
    embedding_dim: int = 128       # compressed embedding dim (< d_model)
    n_clusters: int = 256          # for cluster-based compression
    tie_weights: bool = True       # tie input/output embeddings
    use_factorization: bool = True  # ALBERT-style factorization


class FactorizedEmbedding(nn.Module):
    """ALBERT-style embedding factorization: vocab → small_dim → d_model.

    Saves parameters when embedding_dim << d_model.
    Standard: vocab_size * d_model
    Factorized: vocab_size * embedding_dim + embedding_dim * d_model
    """

    def __init__(self, vocab_size: int, embedding_dim: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, d_model, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed then project: returns (B, T, d_model)."""
        return self.proj(self.embed(input_ids))

    def num_parameters(self) -> int:
        """Returns vocab_size * embedding_dim + embedding_dim * d_model."""
        return self.vocab_size * self.embedding_dim + self.embedding_dim * self.d_model

    def compression_ratio(self, d_model: int) -> float:
        """Ratio of standard params to factorized params.

        Standard: vocab_size * d_model
        Factorized: vocab_size * embedding_dim + embedding_dim * d_model
        """
        standard = self.vocab_size * d_model
        factorized = self.vocab_size * self.embedding_dim + self.embedding_dim * d_model
        return standard / factorized


class ClusteredEmbedding(nn.Module):
    """Cluster-based embedding: each token maps to a cluster centroid + offset.

    Parameters:
        centroids: n_clusters * d_model
        offsets: vocab_size * d_model
    """

    def __init__(self, vocab_size: int, d_model: int, n_clusters: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_clusters = n_clusters
        # cluster assignment: which cluster each token belongs to
        cluster_assign = torch.zeros(vocab_size, dtype=torch.long)
        self.register_buffer("cluster_assign", cluster_assign)
        self.centroids = nn.Embedding(n_clusters, d_model)
        self.offsets = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Lookup cluster centroid + per-token offset: returns (B, T, d_model)."""
        # input_ids: (B, T)
        cluster_ids = self.cluster_assign[input_ids]  # (B, T)
        centroid = self.centroids(cluster_ids)         # (B, T, d_model)
        offset = self.offsets(input_ids)               # (B, T, d_model)
        return centroid + offset

    def num_parameters(self) -> int:
        """Returns n_clusters * d_model + vocab_size * d_model."""
        return self.n_clusters * self.d_model + self.vocab_size * self.d_model


class TiedEmbedding(nn.Module):
    """Shared input/output embedding matrix (reduces parameters).

    A single weight matrix serves both token lookup and unembedding (logit projection).
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def embed(self, input_ids: Tensor) -> Tensor:
        """Token lookup: F.embedding(input_ids, self.weight) → (B, T, d_model)."""
        return F.embedding(input_ids, self.weight)

    def unembed(self, hidden: Tensor) -> Tensor:
        """Project to vocab: hidden @ self.weight.T → (B, T, vocab_size)."""
        return hidden @ self.weight.T

    def forward(self, input_ids: Tensor) -> Tensor:
        """Alias for embed."""
        return self.embed(input_ids)


def estimate_embedding_memory(
    vocab_size: int,
    d_model: int,
    embedding_dim: int,
    dtype_bytes: int = 4,
) -> dict[str, float]:
    """Compare memory usage: standard vs factorized vs clustered.

    Args:
        vocab_size: Number of tokens in vocabulary.
        d_model: Model hidden dimension.
        embedding_dim: Compressed embedding dimension (factorized).
        dtype_bytes: Bytes per parameter (4 for float32).

    Returns:
        dict with keys:
            standard_mb: Memory for standard embedding (MB).
            factorized_mb: Memory for factorized embedding (MB).
            reduction_factor: standard_mb / factorized_mb.
    """
    standard_params = vocab_size * d_model
    factorized_params = vocab_size * embedding_dim + embedding_dim * d_model

    mb = 1024 * 1024
    standard_mb = standard_params * dtype_bytes / mb
    factorized_mb = factorized_params * dtype_bytes / mb
    reduction_factor = standard_mb / factorized_mb

    return {
        "standard_mb": standard_mb,
        "factorized_mb": factorized_mb,
        "reduction_factor": reduction_factor,
    }


class EmbeddingCompressor:
    """Compress an existing embedding layer via SVD truncation.

    Decomposes a (V, D) embedding weight into (V, r) and (r, D) factors
    such that U_r @ V_r ≈ original, keeping the top-r singular values.
    """

    def __init__(self, target_rank: int) -> None:
        self.target_rank = target_rank

    def compress(self, embedding_weight: Tensor) -> tuple[Tensor, Tensor]:
        """SVD of embedding weight → truncated (U_r, V_r) factors.

        Args:
            embedding_weight: (V, D) embedding weight matrix.

        Returns:
            u_r: (V, r) left singular vectors scaled by singular values.
            v_r: (r, D) right singular vectors.
        """
        # Full SVD: U (V,V), S (min(V,D),), Vh (D,D)
        U, S, Vh = torch.linalg.svd(embedding_weight, full_matrices=False)
        r = self.target_rank
        U_r = U[:, :r] * S[:r]  # (V, r) — absorb singular values into U
        V_r = Vh[:r, :]          # (r, D)
        return U_r, V_r

    def reconstruction_error(self, original: Tensor, u_r: Tensor, v_r: Tensor) -> float:
        """Relative Frobenius reconstruction error.

        Returns ||original - u_r @ v_r||_F / ||original||_F.
        """
        reconstructed = u_r @ v_r
        diff_norm = torch.linalg.norm(original - reconstructed, ord="fro")
        orig_norm = torch.linalg.norm(original, ord="fro")
        return (diff_norm / orig_norm).item()
