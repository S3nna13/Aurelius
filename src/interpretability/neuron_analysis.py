"""
src/interpretability/neuron_analysis.py

Neuron analysis utilities for characterising individual neurons/features in
transformer hidden states.

Pure PyTorch — no HuggingFace, no scipy, no sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# NeuronConfig
# ---------------------------------------------------------------------------


@dataclass
class NeuronConfig:
    """Configuration for neuron analysis experiments."""

    n_top_examples: int = 10
    activation_threshold: float = 0.0
    dead_neuron_threshold: float = 1e-3


# ---------------------------------------------------------------------------
# compute_activation_statistics
# ---------------------------------------------------------------------------


def compute_activation_statistics(
    activations: Tensor,
    threshold: float = 0.0,
) -> dict[str, Tensor]:
    """
    Compute per-neuron summary statistics over a batch of activations.

    Parameters
    ----------
    activations : Tensor, shape (B, T, d_model)
    threshold   : float — value above which an activation counts as "active"
                  (used for the sparsity metric).  Defaults to 0.0.

    Returns
    -------
    dict with keys:
        "mean"     : (d_model,)
        "std"      : (d_model,)
        "max"      : (d_model,)
        "min"      : (d_model,)
        "sparsity" : (d_model,) — fraction of activations above *threshold*
    """
    # Flatten batch and sequence dims → (N, d_model)
    flat = activations.reshape(-1, activations.shape[-1])  # (N, d_model)

    mean = flat.mean(dim=0)  # (d_model,)
    std = flat.std(dim=0, unbiased=False)  # (d_model,)
    max_ = flat.max(dim=0).values  # (d_model,)
    min_ = flat.min(dim=0).values  # (d_model,)
    sparsity = (flat > threshold).float().mean(dim=0)  # (d_model,)

    return {
        "mean": mean,
        "std": std,
        "max": max_,
        "min": min_,
        "sparsity": sparsity,
    }


# ---------------------------------------------------------------------------
# find_dead_neurons
# ---------------------------------------------------------------------------


def find_dead_neurons(activations: Tensor, threshold: float = 1e-3) -> Tensor:
    """
    Identify neurons that never meaningfully activate.

    Parameters
    ----------
    activations : Tensor, shape (B, T, d_model)
    threshold   : float — a neuron is dead if its max |activation| < threshold.

    Returns
    -------
    Tensor of shape (d_model,), dtype bool.
    True means the neuron is dead.
    """
    flat = activations.reshape(-1, activations.shape[-1])  # (N, d_model)
    max_abs = flat.abs().max(dim=0).values  # (d_model,)
    return max_abs < threshold


# ---------------------------------------------------------------------------
# compute_neuron_correlation
# ---------------------------------------------------------------------------


def compute_neuron_correlation(
    activations_a: Tensor,
    activations_b: Tensor,
) -> Tensor:
    """
    Per-neuron Pearson correlation between two activation matrices.

    Parameters
    ----------
    activations_a : Tensor, shape (N, d)  where N = B*T (already flattened)
    activations_b : Tensor, shape (N, d)

    Returns
    -------
    Tensor of shape (d,) containing per-neuron Pearson r values.
    Neurons with zero variance in either matrix receive correlation = 0.
    """
    activations_a.shape[0]

    mean_a = activations_a.mean(dim=0)  # (d,)
    mean_b = activations_b.mean(dim=0)  # (d,)

    da = activations_a - mean_a  # (N, d)
    db = activations_b - mean_b  # (N, d)

    cov = (da * db).sum(dim=0)  # (d,)
    std_a = da.pow(2).sum(dim=0).sqrt()  # (d,)
    std_b = db.pow(2).sum(dim=0).sqrt()  # (d,)

    denom = std_a * std_b  # (d,)

    # Zero-variance neurons → correlation = 0
    corr = torch.where(denom > 0, cov / denom, torch.zeros_like(cov))
    return corr


# ---------------------------------------------------------------------------
# top_activating_examples
# ---------------------------------------------------------------------------


def top_activating_examples(activations: Tensor, k: int) -> Tensor:
    """
    Find the top-k highest-activating examples per neuron.

    Parameters
    ----------
    activations : Tensor, shape (N, d)  — N is already the flattened B*T
    k           : int — number of top examples to return

    Returns
    -------
    Tensor of shape (d, k) — indices into the N dimension (LongTensor).
    """
    # topk along the N dimension for each neuron
    # activations is (N, d); we want topk over N for each column → transpose first
    top_indices = activations.topk(k, dim=0).indices  # (k, d)
    return top_indices.T  # (d, k)


# ---------------------------------------------------------------------------
# polysemanticity_score
# ---------------------------------------------------------------------------


def polysemanticity_score(activations: Tensor, k: int = 5) -> Tensor:
    """
    Measure how many distinct "roles" each neuron plays.

    Implementation: compute the std of the top-k activating position indices,
    normalised by N.  A high score means the neuron activates for many
    spread-out examples (polysemantic); a low score means it focuses on a
    cluster.

    Parameters
    ----------
    activations : Tensor, shape (N, d)
    k           : int — number of top examples to consider

    Returns
    -------
    Tensor of shape (d,) with values in [0, 1].
    """
    N, d = activations.shape
    k = min(k, N)

    top_idx = top_activating_examples(activations, k)  # (d, k)
    # std of top-k indices per neuron, normalised by N
    # When k==1 std is 0; that is correct (no spread).
    score = top_idx.float().std(dim=1) / N  # (d,)
    # Clamp to [0, 1] for safety
    score = score.clamp(0.0, 1.0)
    return score


# ---------------------------------------------------------------------------
# NeuronAnalyzer
# ---------------------------------------------------------------------------


class NeuronAnalyzer:
    """High-level API for neuron characterisation."""

    def __init__(self, config: NeuronConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def analyze(self, activations: Tensor) -> dict[str, Any]:
        """
        Run a full neuron analysis pass.

        Parameters
        ----------
        activations : Tensor, shape (B, T, d_model)

        Returns
        -------
        dict with keys:
            "stats"       : output of compute_activation_statistics
            "dead_mask"   : (d_model,) bool tensor
            "top_examples": (d_model, n_top_examples) LongTensor
        """
        stats = compute_activation_statistics(
            activations,
            threshold=self.config.activation_threshold,
        )
        dead_mask = find_dead_neurons(
            activations,
            threshold=self.config.dead_neuron_threshold,
        )

        flat = activations.reshape(-1, activations.shape[-1])  # (N, d_model)
        k = min(self.config.n_top_examples, flat.shape[0])
        top_examples = top_activating_examples(flat, k)  # (d_model, k)

        return {
            "stats": stats,
            "dead_mask": dead_mask,
            "top_examples": top_examples,
        }

    # ------------------------------------------------------------------
    # compare_layers
    # ------------------------------------------------------------------

    def compare_layers(
        self,
        layer_a: Tensor,
        layer_b: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compare two (B, T, d) activation tensors at the neuron level.

        Returns
        -------
        dict with keys:
            "correlation" : (d,) per-neuron Pearson r
            "mean_diff"   : (d,) absolute difference of per-neuron means
            "std_diff"    : (d,) absolute difference of per-neuron stds
        """
        flat_a = layer_a.reshape(-1, layer_a.shape[-1])  # (N, d)
        flat_b = layer_b.reshape(-1, layer_b.shape[-1])  # (N, d)

        corr = compute_neuron_correlation(flat_a, flat_b)

        stats_a = compute_activation_statistics(layer_a)
        stats_b = compute_activation_statistics(layer_b)

        mean_diff = (stats_a["mean"] - stats_b["mean"]).abs()
        std_diff = (stats_a["std"] - stats_b["std"]).abs()

        return {
            "correlation": corr,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
        }
