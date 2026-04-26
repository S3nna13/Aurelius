"""Neuron Activation Analyzer — per-neuron statistics for Aurelius transformer.

Provides tools to identify dead neurons and monosemantic (selective)
neurons from activation tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NeuronActivationStats:
    layer_idx: int
    neuron_idx: int
    mean: float
    std: float
    max_val: float
    top_tokens: list[int]  # top-5 flat position indices by activation magnitude


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class NeuronActivationAnalyzer:
    """Compute per-neuron activation statistics from forward-pass activations."""

    def compute_stats(
        self,
        activations: torch.Tensor,
        layer_idx: int,
    ) -> list[NeuronActivationStats]:
        """Compute per-neuron statistics from an activation tensor.

        Args:
            activations: Float tensor of shape (batch, seq, d_model).
            layer_idx: Index of the layer these activations come from.

        Returns:
            List of NeuronActivationStats, one per neuron (d_model dimension).
        """
        # activations: (B, S, D)
        if activations.dim() != 3:
            raise ValueError(
                f"Expected 3-D activations (batch, seq, d_model), got {activations.dim()}-D"
            )

        B, S, D = activations.shape
        # Flatten to (B*S, D) for per-neuron statistics
        flat = activations.reshape(B * S, D)  # (N, D)

        means = flat.mean(dim=0)  # (D,)
        stds = flat.std(dim=0)  # (D,)
        maxvals = flat.abs().max(dim=0).values  # (D,) — magnitude max

        stats: list[NeuronActivationStats] = []
        # Top-5 flat positions by magnitude for each neuron
        k = min(5, B * S)
        for neuron_idx in range(D):
            neuron_acts = flat[:, neuron_idx].abs()  # (N,)
            top_k = neuron_acts.topk(k).indices.tolist()
            stats.append(
                NeuronActivationStats(
                    layer_idx=layer_idx,
                    neuron_idx=neuron_idx,
                    mean=float(means[neuron_idx].item()),
                    std=float(stds[neuron_idx].item()),
                    max_val=float(maxvals[neuron_idx].item()),
                    top_tokens=top_k,
                )
            )
        return stats

    def find_dead_neurons(
        self,
        stats: list[NeuronActivationStats],
        threshold: float = 0.01,
    ) -> list[int]:
        """Return neuron indices whose max activation is below threshold.

        A neuron is considered dead if it never activates significantly,
        i.e. max_val < threshold.

        Args:
            stats: List of NeuronActivationStats from compute_stats().
            threshold: Max activation below which a neuron is dead.

        Returns:
            Sorted list of dead neuron indices.
        """
        return sorted(s.neuron_idx for s in stats if s.max_val < threshold)

    def find_monosemantic_neurons(
        self,
        stats: list[NeuronActivationStats],
        mono_threshold: float = 0.9,
    ) -> list[int]:
        """Return neuron indices that appear monosemantic (selective activation).

        Proxy: std / |mean| < mono_threshold when |mean| > 0.
        Low relative variance suggests the neuron responds strongly to a
        specific feature and weakly to most others.

        Args:
            stats: List of NeuronActivationStats from compute_stats().
            mono_threshold: Relative std threshold (std/|mean| < this => mono).

        Returns:
            Sorted list of monosemantic neuron indices.
        """
        result: list[int] = []
        for s in stats:
            abs_mean = abs(s.mean)
            if abs_mean > 1e-8:
                relative_std = s.std / abs_mean
                if relative_std < mono_threshold:
                    result.append(s.neuron_idx)
        return sorted(result)
