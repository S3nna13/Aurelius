"""Structural Probes and MDL Probing for representation analysis.

Structural probe: tests if syntactic structure is linearly encoded.
MDL probe: measures how efficiently a probe can encode labels,
controlling for probe capacity via online coding (compression).

References:
    Hewitt & Manning 2019 — https://arxiv.org/abs/1905.06316
    Voita & Titov 2020 — https://arxiv.org/abs/2003.12298
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor


class StructuralProbe(nn.Module):
    """Learns a linear map B such that ||B(h_i - h_j)||^2 approx syntactic_dist(i, j).

    The low-rank transformation B in R^{rank x d_model} is the probe parameter.
    If rank < d_model this is a low-rank structural probe.
    """

    def __init__(self, d_model: int, rank: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        # B: (rank, d_model) — transformation matrix
        self.B = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def distance(self, h_i: Tensor, h_j: Tensor) -> Tensor:
        """Squared distance in probe space between pairs of hidden states.

        Args:
            h_i: (B, d_model) hidden states at position i
            h_j: (B, d_model) hidden states at position j

        Returns:
            (B,) squared distances ||B(h_i - h_j)||^2
        """
        diff = h_i - h_j  # (B, d_model)
        projected = diff @ self.B.T  # (B, rank)
        return (projected ** 2).sum(dim=-1)  # (B,)

    def distance_matrix(self, hidden_states: Tensor) -> Tensor:
        """Compute pairwise squared distances for all positions.

        Args:
            hidden_states: (T, d_model)

        Returns:
            (T, T) matrix of pairwise squared distances
        """
        # Project all hidden states: (T, rank)
        projected = hidden_states @ self.B.T  # (T, rank)

        # ||Bh_i - Bh_j||^2 = ||Bh_i||^2 + ||Bh_j||^2 - 2 * <Bh_i, Bh_j>
        sq_norms = (projected ** 2).sum(dim=-1)  # (T,)
        dot = projected @ projected.T  # (T, T)
        dist_mat = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * dot
        # Clamp to avoid small negatives from floating-point error
        return dist_mat.clamp(min=0.0)

    def loss(self, hidden_states: Tensor, target_distances: Tensor) -> Tensor:
        """MSE loss on upper-triangle (excluding diagonal) of distance matrix.

        Args:
            hidden_states: (T, d_model)
            target_distances: (T, T) syntactic distances

        Returns:
            Scalar MSE loss
        """
        pred = self.distance_matrix(hidden_states)
        T = hidden_states.shape[0]
        # Upper-triangle mask (excluding diagonal)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=hidden_states.device), diagonal=1)
        return F.mse_loss(pred[mask], target_distances[mask])


class MDLProbeDataset:
    """Splits representation data for online-coding MDL estimation.

    The MDL (Minimum Description Length) framework measures how efficiently
    a probe compresses labels, using exponentially growing training subsets.
    """

    def __init__(self, representations: Tensor, labels: Tensor, n_splits: int = 8) -> None:
        """
        Args:
            representations: (N, d_model)
            labels: (N,) long tensor of class labels
            n_splits: number of exponentially growing subsets
        """
        self.representations = representations
        self.labels = labels
        self.n_splits = n_splits
        self.N = representations.shape[0]

    def split_fractions(self) -> List[float]:
        """Returns fractions [1/2, 1/4, ..., 1/2^n_splits] for exponential splits."""
        return [1.0 / (2 ** k) for k in range(1, self.n_splits + 1)]

    def get_split(self, fraction: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split data into train (first fraction*N) and test (rest).

        Args:
            fraction: fraction of N to use as training data

        Returns:
            (train_repr, train_labels, test_repr, test_labels)
        """
        n_train = max(1, int(fraction * self.N))
        train_repr = self.representations[:n_train]
        train_labels = self.labels[:n_train]
        test_repr = self.representations[n_train:]
        test_labels = self.labels[n_train:]
        return train_repr, train_labels, test_repr, test_labels


class MDLProbeTrainer:
    """Trains linear probes on progressively larger subsets to measure description length.

    The MDL score quantifies how efficiently a linear probe encodes a property
    compared to a uniform (random guess) baseline.
    """

    def __init__(self, d_model: int, n_classes: int, n_epochs_per_split: int = 10) -> None:
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_epochs_per_split = n_epochs_per_split

    def train_on_split(self, train_repr: Tensor, train_labels: Tensor) -> nn.Module:
        """Train a fresh linear probe on the given training split.

        Args:
            train_repr: (n_train, d_model)
            train_labels: (n_train,) long tensor

        Returns:
            Trained nn.Linear probe
        """
        probe = nn.Linear(self.d_model, self.n_classes)
        optimizer = optim.Adam(probe.parameters(), lr=1e-3)
        probe.train()
        for _ in range(self.n_epochs_per_split):
            optimizer.zero_grad()
            logits = probe(train_repr)
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()
        probe.eval()
        return probe

    def codelength(self, probe: nn.Module, repr_: Tensor, labels: Tensor) -> float:
        """Total description length (in nats) of labels given representations.

        Args:
            probe: trained nn.Linear
            repr_: (N, d_model)
            labels: (N,) long tensor

        Returns:
            Cross-entropy loss * N (sum reduction), in nats
        """
        with torch.no_grad():
            logits = probe(repr_)
            length = F.cross_entropy(logits, labels, reduction="sum").item()
        return length

    def mdl_score(self, dataset: MDLProbeDataset) -> Dict[str, float]:
        """Compute MDL score via online coding over exponentially growing splits.

        Trains a probe on each fraction of data, measures codelength on the test
        portion, and aggregates. Compression relative to a uniform (random) code
        is the key metric.

        Args:
            dataset: MDLProbeDataset with representations and labels

        Returns:
            Dict with keys:
                'total_codelength': sum of codelengths across all splits (nats)
                'uniform_codelength': N * log(n_classes) — random-guess baseline
                'compression': 1 - total_codelength / uniform_codelength
        """
        N = dataset.N
        total_codelength = 0.0

        for fraction in dataset.split_fractions():
            train_repr, train_labels, test_repr, test_labels = dataset.get_split(fraction)
            if test_repr.shape[0] == 0:
                continue
            probe = self.train_on_split(train_repr, train_labels)
            total_codelength += self.codelength(probe, test_repr, test_labels)

        uniform_codelength = N * math.log(self.n_classes)
        compression = 1.0 - total_codelength / uniform_codelength if uniform_codelength > 0 else 0.0

        return {
            "total_codelength": total_codelength,
            "uniform_codelength": uniform_codelength,
            "compression": compression,
        }


class ProbingBenchmark:
    """Orchestrates multiple probing experiments across layers and methods."""

    def __init__(self) -> None:
        pass

    def run_linear_probe(
        self,
        train_repr: Tensor,
        train_labels: Tensor,
        test_repr: Tensor,
        test_labels: Tensor,
        n_epochs: int = 20,
    ) -> Dict[str, float]:
        """Train a linear probe and report train/test accuracy.

        Args:
            train_repr: (N_train, d_model)
            train_labels: (N_train,) long tensor
            test_repr: (N_test, d_model)
            test_labels: (N_test,) long tensor
            n_epochs: number of training epochs

        Returns:
            Dict with keys: 'train_acc', 'test_acc', 'n_classes'
        """
        d_model = train_repr.shape[1]
        n_classes = int(max(train_labels.max().item(), test_labels.max().item())) + 1

        probe = nn.Linear(d_model, n_classes)
        optimizer = optim.Adam(probe.parameters(), lr=1e-3)

        probe.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = probe(train_repr)
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            train_preds = probe(train_repr).argmax(dim=-1)
            train_acc = (train_preds == train_labels).float().mean().item()
            test_preds = probe(test_repr).argmax(dim=-1)
            test_acc = (test_preds == test_labels).float().mean().item()

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_classes": n_classes,
        }

    def compare_layers(
        self,
        layer_reprs: List[Tensor],
        labels: Tensor,
        test_frac: float = 0.2,
    ) -> List[Dict[str, float]]:
        """Run linear probe on each layer's representations.

        Args:
            layer_reprs: list of (N, d_model) tensors, one per layer
            labels: (N,) long tensor of class labels
            test_frac: fraction of data to hold out as test

        Returns:
            List of result dicts (one per layer), each with 'train_acc', 'test_acc', 'n_classes'
        """
        N = labels.shape[0]
        n_test = max(1, int(test_frac * N))
        n_train = N - n_test

        results: List[Dict[str, float]] = []
        for repr_ in layer_reprs:
            train_repr = repr_[:n_train]
            train_labels = labels[:n_train]
            test_repr = repr_[n_train:]
            test_labels = labels[n_train:]
            result = self.run_linear_probe(train_repr, train_labels, test_repr, test_labels)
            results.append(result)

        return results
