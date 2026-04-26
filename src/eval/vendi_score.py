"""
vendi_score.py -- Vendi Score Diversity Metric

Implements the Vendi Score from:
  "The Vendi Score: A Diversity Evaluation Metric for Machine Learning"
  Friedman & Dieng, arXiv:2210.02410

VS(S) = exp(H(K/n))
where K is the n×n kernel matrix, K/n is normalized, and
H is the von Neumann entropy: H(K/n) = -sum_i λ_i log(λ_i).

VS = 1  => all samples identical
VS = n  => all samples orthogonal (maximally diverse)

Pure PyTorch only -- uses torch.linalg.eigvalsh for eigendecomposition.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Core: Vendi Score from a kernel matrix
# ---------------------------------------------------------------------------


def vendi_score(K: Tensor, eps: float = 1e-10) -> float:
    """Vendi Score from kernel matrix K (n×n, PSD).

    VS = exp(-sum_i λ_i log(λ_i)) where λ_i are the eigenvalues of K/n.
    Eigenvalues that are numerically ≤ 0 are clamped to eps before the
    log to guarantee numerical stability.

    Args:
        K:   (n, n) symmetric positive semi-definite kernel matrix.
        eps: Floor for eigenvalues before computing entropy (default 1e-10).

    Returns:
        Vendi Score as a Python float.
    """
    n = K.shape[0]
    if n == 1:
        return 1.0

    # Normalise: K/n so that tr(K/n) == 1 (matches paper §2)
    K_norm = K / n

    # Eigenvalues of a symmetric matrix (ascending order).
    # eigvalsh is numerically stable for symmetric/Hermitian inputs.
    eigenvalues = torch.linalg.eigvalsh(K_norm)  # shape (n,)

    # Clamp negatives that arise from floating-point noise
    eigenvalues = eigenvalues.clamp(min=0.0)

    # Normalise so they sum to 1 (they should already, but floating-point)
    eigenvalues = eigenvalues / (eigenvalues.sum() + eps)

    # von Neumann entropy: H = -sum_i λ_i log(λ_i), skip λ_i == 0
    mask = eigenvalues > eps
    lam = eigenvalues[mask]
    entropy = -(lam * torch.log(lam)).sum().item()

    return math.exp(entropy)


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------


def _cosine_kernel(embeddings: Tensor) -> Tensor:
    """Cosine similarity kernel: K_{ij} = (e_i · e_j) / (||e_i|| ||e_j||).

    Args:
        embeddings: (n, d) embedding matrix.

    Returns:
        (n, n) kernel matrix with values in [-1, 1].
    """
    normed = F.normalize(embeddings, p=2, dim=-1)  # (n, d)
    return normed @ normed.T  # (n, n)


def _linear_kernel(embeddings: Tensor) -> Tensor:
    """Linear (dot-product) kernel: K_{ij} = e_i · e_j.

    Args:
        embeddings: (n, d) embedding matrix.

    Returns:
        (n, n) kernel matrix.
    """
    return embeddings @ embeddings.T  # (n, n)


def _rbf_kernel(embeddings: Tensor, sigma: float = 1.0) -> Tensor:
    """RBF (Gaussian) kernel: K_{ij} = exp(-||e_i - e_j||^2 / (2 σ^2)).

    Args:
        embeddings: (n, d) embedding matrix.
        sigma:      Bandwidth parameter.

    Returns:
        (n, n) kernel matrix with values in (0, 1].
    """
    # Squared Euclidean distances via broadcasting
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (n, n, d)
    sq_dist = (diff**2).sum(dim=-1)  # (n, n)
    return torch.exp(-sq_dist / (2.0 * sigma**2))


def _exact_match_kernel(sequences: list[list[int]]) -> Tensor:
    """Exact-match kernel: K_{ij} = 1 if s_i == s_j else 0.

    Args:
        sequences: List of n token-id sequences.

    Returns:
        (n, n) float kernel matrix.
    """
    n = len(sequences)
    K = torch.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            if sequences[i] == sequences[j]:
                K[i, j] = 1.0
                K[j, i] = 1.0
    return K


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embedding_vendi_score(
    embeddings: Tensor,
    kernel: str = "cosine",
    rbf_sigma: float = 1.0,
) -> float:
    """Compute Vendi Score from an embedding matrix.

    Args:
        embeddings: (n, d) tensor — one embedding per sample.
        kernel:     Kernel type: "cosine" | "linear" | "rbf".
        rbf_sigma:  Bandwidth for RBF kernel (ignored otherwise).

    Returns:
        Vendi Score as a Python float.
    """
    if embeddings.shape[0] == 1:
        return 1.0

    embeddings = embeddings.float()

    if kernel == "cosine":
        K = _cosine_kernel(embeddings)
    elif kernel == "linear":
        K = _linear_kernel(embeddings)
    elif kernel == "rbf":
        K = _rbf_kernel(embeddings, sigma=rbf_sigma)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'cosine', 'linear', or 'rbf'.")

    return vendi_score(K)


def token_vendi_score(sequences: list[list[int]]) -> float:
    """Vendi Score using the exact-match kernel on token-id sequences.

    K_{ij} = 1 if sequences i and j are identical, else 0.

    Args:
        sequences: List of token-id sequences (each a list of ints).

    Returns:
        Vendi Score as a Python float.
    """
    if len(sequences) == 1:
        return 1.0

    K = _exact_match_kernel(sequences)
    return vendi_score(K)


# ---------------------------------------------------------------------------
# VendiScorer class
# ---------------------------------------------------------------------------


class VendiScorer:
    """Stateful Vendi Score evaluator.

    Args:
        kernel:    Kernel type: "cosine" | "linear" | "rbf".
        rbf_sigma: Bandwidth for RBF kernel (ignored otherwise).
    """

    def __init__(self, kernel: str = "cosine", rbf_sigma: float = 1.0) -> None:
        if kernel not in ("cosine", "linear", "rbf"):
            raise ValueError(f"Unknown kernel '{kernel}'. Choose 'cosine', 'linear', or 'rbf'.")
        self.kernel = kernel
        self.rbf_sigma = rbf_sigma

    def score(self, embeddings: Tensor) -> float:
        """Compute Vendi Score for a single group of embeddings.

        Args:
            embeddings: (n, d) tensor.

        Returns:
            Vendi Score as a Python float.
        """
        return embedding_vendi_score(
            embeddings,
            kernel=self.kernel,
            rbf_sigma=self.rbf_sigma,
        )

    def score_batch(self, embeddings_list: list[Tensor]) -> list[float]:
        """Score multiple groups of embeddings independently.

        Args:
            embeddings_list: List of (n_i, d) tensors.

        Returns:
            List of Vendi Scores, one per group.
        """
        return [self.score(emb) for emb in embeddings_list]
