"""
Probing classifiers for the Aurelius LLM project.

Fits simple linear models on hidden states to test what information is
encoded at each layer of the transformer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProbingConfig:
    """Configuration for probing classifiers."""

    n_classes: int = 2
    d_hidden: int = 512
    n_epochs: int = 100
    lr: float = 1e-3
    l2_reg: float = 1e-4
    batch_size: int = 64


# ---------------------------------------------------------------------------
# Linear Probe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Single linear layer probe: maps hidden states to class logits."""

    def __init__(self, d_hidden: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_hidden, n_classes, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: (N, d_hidden) hidden state tensor

        Returns:
            logits: (N, n_classes)
        """
        return self.linear(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_probe(
    probe: LinearProbe,
    hiddens: Tensor,
    labels: Tensor,
    config: ProbingConfig,
) -> list[float]:
    """
    Train the probe with AdamW + L2 regularisation.

    Args:
        probe:   LinearProbe to train (mutated in-place).
        hiddens: (N, d_hidden) hidden states.
        labels:  (N,) integer class labels.
        config:  ProbingConfig controlling training hyper-parameters.

    Returns:
        List of per-epoch average cross-entropy losses.
    """
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=config.l2_reg)

    n = hiddens.size(0)
    epoch_losses: list[float] = []

    probe.train()
    for _ in range(config.n_epochs):
        # Shuffle indices each epoch
        perm = torch.randperm(n)
        hiddens_shuffled = hiddens[perm]
        labels_shuffled = labels[perm]

        batch_losses: list[float] = []
        for start in range(0, n, config.batch_size):
            end = start + config.batch_size
            h_batch = hiddens_shuffled[start:end]
            y_batch = labels_shuffled[start:end]

            optimizer.zero_grad()
            logits = probe(h_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_losses.append(float(sum(batch_losses) / len(batch_losses)))

    probe.eval()
    return epoch_losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_probe(
    probe: LinearProbe,
    hiddens: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    """
    Evaluate probe accuracy and loss on a test set.

    Args:
        probe:   Trained LinearProbe.
        hiddens: (N, d_hidden) hidden states.
        labels:  (N,) integer class labels.

    Returns:
        Dict with keys "accuracy" and "loss".
    """
    probe.eval()
    with torch.no_grad():
        logits = probe(hiddens)
        loss = F.cross_entropy(logits, labels).item()
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()

    return {"accuracy": accuracy, "loss": loss}


# ---------------------------------------------------------------------------
# Mutual Information Proxy
# ---------------------------------------------------------------------------


def compute_mutual_information_proxy(
    hiddens: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> float:
    """
    Approximate mutual information between hidden states and labels.

    Uses hiddens[:,0] as a scalar proxy for the hidden representation,
    then bins it to form a discrete joint distribution with labels.

    MI = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )

    Args:
        hiddens: (N, d_hidden) hidden state tensor.
        labels:  (N,) integer class labels.
        n_bins:  Number of bins for discretising the hidden dimension.

    Returns:
        Non-negative float approximation of MI.
    """
    x = hiddens[:, 0].float()
    y = labels.long()

    n = x.size(0)
    n_classes = int(y.max().item()) + 1

    # Bin the continuous dimension
    x_min = x.min()
    x_max = x.max()
    # Avoid degenerate case where all values are identical
    if (x_max - x_min).abs() < 1e-8:
        return 0.0

    # Map x to bin indices in [0, n_bins-1]
    bin_indices = ((x - x_min) / (x_max - x_min) * n_bins).long().clamp(0, n_bins - 1)

    # Build joint histogram p(x_bin, y)
    joint = torch.zeros(n_bins, n_classes)
    for i in range(n):
        joint[bin_indices[i], y[i]] += 1.0
    joint = joint / n  # normalise to probabilities

    # Marginals
    p_x = joint.sum(dim=1)  # (n_bins,)
    p_y = joint.sum(dim=0)  # (n_classes,)

    # MI = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )
    mi = 0.0
    for xi in range(n_bins):
        for yi in range(n_classes):
            pxy = joint[xi, yi].item()
            if pxy <= 0.0:
                continue
            px = p_x[xi].item()
            py = p_y[yi].item()
            denom = px * py
            if denom <= 0.0:
                continue
            mi += pxy * math.log(pxy / denom)

    return max(0.0, mi)


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------


@dataclass
class LayerProbingResults:
    """Results from probing a single transformer layer."""

    layer_idx: int
    accuracy: float
    loss: float
    train_losses: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Multi-layer prober
# ---------------------------------------------------------------------------


class MultiLayerProber:
    """Probe multiple transformer layers and identify the most informative one."""

    def __init__(self, config: ProbingConfig) -> None:
        self.config = config

    def probe_layer(
        self,
        hiddens: Tensor,
        labels: Tensor,
        layer_idx: int,
    ) -> LayerProbingResults:
        """
        Train and evaluate a probe for a single layer.

        Args:
            hiddens:   (N, d_hidden) hidden states for this layer.
            labels:    (N,) integer class labels.
            layer_idx: Index of the layer being probed.

        Returns:
            LayerProbingResults with accuracy, loss, and per-epoch losses.
        """
        d_hidden = hiddens.size(1)
        probe = LinearProbe(d_hidden, self.config.n_classes)
        train_losses = train_probe(probe, hiddens, labels, self.config)
        metrics = evaluate_probe(probe, hiddens, labels)
        return LayerProbingResults(
            layer_idx=layer_idx,
            accuracy=metrics["accuracy"],
            loss=metrics["loss"],
            train_losses=train_losses,
        )

    def probe_all_layers(
        self,
        all_hiddens: list[Tensor],
        labels: Tensor,
    ) -> list[LayerProbingResults]:
        """
        Probe every layer and return results sorted by layer index.

        Args:
            all_hiddens: List of (N, d_hidden) tensors, one per layer.
            labels:      (N,) integer class labels.

        Returns:
            List of LayerProbingResults sorted by layer_idx.
        """
        results = [
            self.probe_layer(hiddens, labels, layer_idx)
            for layer_idx, hiddens in enumerate(all_hiddens)
        ]
        results.sort(key=lambda r: r.layer_idx)
        return results

    def get_best_layer(self, results: list[LayerProbingResults]) -> int:
        """
        Return the layer index with the highest probing accuracy.

        Args:
            results: List of LayerProbingResults from probe_all_layers.

        Returns:
            layer_idx of the best-performing layer.
        """
        best = max(results, key=lambda r: r.accuracy)
        return best.layer_idx
