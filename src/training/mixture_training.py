"""Mixture model training utilities using soft cluster assignments (EM-style).

Provides functions and classes for routing training examples to mixture
components via soft assignments, running EM-style centroid updates, and
computing mixture-weighted losses.

Typical usage during training:

    trainer = MixtureTrainer(model, config)
    centroids = trainer.init_centroids(embeddings)
    assignments, centroids = trainer.run_em(embeddings, config.em_iterations)
    loss = trainer.compute_loss(per_sample_losses, embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MixtureConfig:
    """Configuration for mixture model training.

    Attributes:
        n_components:          Number of mixture components (clusters).
        d_model:               Model embedding dimensionality.
        temperature:           Softmax temperature for soft assignment sharpness.
                               Lower values produce harder assignments.
        min_component_weight:  Minimum weight assigned to each component to
                               avoid degenerate solutions.
        em_iterations:         Number of EM steps to run during training.
    """

    n_components: int = 4
    d_model: int = 512
    temperature: float = 1.0
    min_component_weight: float = 0.01
    em_iterations: int = 10


# ---------------------------------------------------------------------------
# Core EM functions
# ---------------------------------------------------------------------------

def compute_soft_assignments(
    embeddings: Tensor,
    centroids: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Compute soft cluster assignments via softmax over negative distances.

    Args:
        embeddings: (N, d) tensor of sample embeddings.
        centroids:  (K, d) tensor of cluster centroids.
        temperature: Softmax temperature; lower is sharper.

    Returns:
        (N, K) tensor of soft assignments, each row sums to 1.
    """
    # (N, K) squared Euclidean distances
    # ||e - c||^2 = ||e||^2 + ||c||^2 - 2 e·c
    emb_sq = (embeddings ** 2).sum(dim=-1, keepdim=True)          # (N, 1)
    cen_sq = (centroids ** 2).sum(dim=-1, keepdim=True).t()        # (1, K)
    dot = embeddings @ centroids.t()                                # (N, K)
    distances = emb_sq + cen_sq - 2.0 * dot                        # (N, K)

    logits = -distances / temperature
    return torch.softmax(logits, dim=-1)


def update_centroids(embeddings: Tensor, assignments: Tensor) -> Tensor:
    """Weighted mean update for cluster centroids (M-step).

    new_centroid_k = Σ_n assignment_{n,k} * embed_n / Σ_n assignment_{n,k}

    Args:
        embeddings:  (N, d) tensor of sample embeddings.
        assignments: (N, K) tensor of soft assignments.

    Returns:
        (K, d) updated centroid tensor.
    """
    # assignments: (N, K) -> (K, N) for matmul
    weights = assignments.t()                           # (K, N)
    weight_sums = weights.sum(dim=-1, keepdim=True)    # (K, 1)
    # Avoid division by zero
    weight_sums = weight_sums.clamp(min=1e-8)
    new_centroids = (weights @ embeddings) / weight_sums  # (K, d)
    return new_centroids


def em_step(
    embeddings: Tensor,
    centroids: Tensor,
    temperature: float,
) -> Tuple[Tensor, Tensor]:
    """Run one EM step: E-step (soft assignments) + M-step (centroid update).

    Args:
        embeddings:  (N, d) tensor of sample embeddings.
        centroids:   (K, d) current centroid estimates.
        temperature: Softmax temperature for the E-step.

    Returns:
        Tuple of (new_assignments (N, K), new_centroids (K, d)).
    """
    new_assignments = compute_soft_assignments(embeddings, centroids, temperature)
    new_centroids = update_centroids(embeddings, new_assignments)
    return new_assignments, new_centroids


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_component_loss(
    per_sample_loss: Tensor,
    assignments: Tensor,
) -> Tensor:
    """Compute weighted loss per mixture component.

    loss_k = Σ_n assignment_{n,k} * loss_n / Σ_n assignment_{n,k}

    Args:
        per_sample_loss: (N,) per-sample scalar losses.
        assignments:     (N, K) soft assignments.

    Returns:
        (K,) tensor of per-component weighted losses.
    """
    weights = assignments.t()                           # (K, N)
    weight_sums = weights.sum(dim=-1)                   # (K,)
    weight_sums = weight_sums.clamp(min=1e-8)
    # (K, N) @ (N,) -> (K,)
    weighted_losses = weights @ per_sample_loss         # (K,)
    return weighted_losses / weight_sums


def mixture_weighted_loss(
    per_sample_loss: Tensor,
    assignments: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Compute total mixture-weighted loss as Σ_k weight_k * loss_k.

    Args:
        per_sample_loss: (N,) per-sample scalar losses.
        assignments:     (N, K) soft assignments.
        weights:         (K,) component weights. If None, uses uniform weights.

    Returns:
        Scalar total loss.
    """
    component_losses = compute_component_loss(per_sample_loss, assignments)  # (K,)
    K = component_losses.shape[0]
    if weights is None:
        weights = torch.ones(K, dtype=component_losses.dtype, device=component_losses.device) / K
    return (weights * component_losses).sum()


# ---------------------------------------------------------------------------
# Gating network
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Learned gating network that produces soft component assignments.

    Maps token embeddings to soft assignments over K mixture components
    via a single linear projection followed by temperature-scaled softmax.

    Args:
        d_model:      Input embedding dimension.
        n_components: Number of mixture components.
        temperature:  Softmax temperature.
    """

    def __init__(self, d_model: int, n_components: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.linear = nn.Linear(d_model, n_components, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Compute soft assignments for each token position.

        Args:
            x: (B, T, d_model) input tensor.

        Returns:
            (B, T, n_components) soft assignments summing to 1 per position.
        """
        logits = self.linear(x)                        # (B, T, n_components)
        return torch.softmax(logits / self.temperature, dim=-1)


# ---------------------------------------------------------------------------
# Mixture trainer
# ---------------------------------------------------------------------------

class MixtureTrainer:
    """Manages mixture model training with EM-based centroid tracking.

    Args:
        model:  The model being trained (nn.Module).
        config: MixtureConfig with mixture hyperparameters.
    """

    def __init__(self, model: nn.Module, config: MixtureConfig) -> None:
        self.model = model
        self.config = config
        self._centroids: Optional[Tensor] = None

    def init_centroids(self, embeddings: Tensor) -> Tensor:
        """Initialise centroids as a random subset of embeddings.

        Args:
            embeddings: (N, d) tensor of sample embeddings where N >= K.

        Returns:
            (K, d) initial centroid tensor.
        """
        N = embeddings.shape[0]
        K = self.config.n_components
        indices = torch.randperm(N, device=embeddings.device)[:K]
        self._centroids = embeddings[indices].detach().clone()
        return self._centroids

    def run_em(self, embeddings: Tensor, n_iter: int) -> Tuple[Tensor, Tensor]:
        """Run n_iter EM steps starting from current centroids.

        Centroids are initialised automatically if not yet set.

        Args:
            embeddings: (N, d) tensor of sample embeddings.
            n_iter:     Number of EM iterations to run.

        Returns:
            Tuple of (assignments (N, K), centroids (K, d)) after n_iter steps.
        """
        if self._centroids is None:
            self.init_centroids(embeddings)

        centroids = self._centroids
        assignments = compute_soft_assignments(
            embeddings, centroids, self.config.temperature
        )
        for _ in range(n_iter):
            assignments, centroids = em_step(
                embeddings, centroids, self.config.temperature
            )
        self._centroids = centroids
        return assignments, centroids

    def compute_loss(
        self, per_sample_losses: Tensor, embeddings: Tensor
    ) -> Tensor:
        """Run one EM step and compute the mixture-weighted loss.

        Args:
            per_sample_losses: (N,) per-sample scalar losses.
            embeddings:        (N, d) sample embeddings used for assignment.

        Returns:
            Scalar mixture-weighted loss.
        """
        if self._centroids is None:
            self.init_centroids(embeddings)

        assignments, self._centroids = em_step(
            embeddings, self._centroids, self.config.temperature
        )
        return mixture_weighted_loss(per_sample_losses, assignments)
