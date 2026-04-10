"""Comprehensive probing suite for Aurelius LLM interpretability.

Extends probing.py (LinearProbe, extract_layer_hiddens) and probing_advanced.py
(ProbingConfig with attention probe type, MLPProbe with ReLU, MultiLayerProber)
with a full suite: ProbingDataset, ProbingClassifier, LayerwiseProber, and
mutual information estimation integrated into layer-wise results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProbingConfig:
    """Configuration for probing_suite experiments.

    Distinct from probing_advanced.ProbingConfig: adds dropout field, removes
    batch_size and n_layers_to_probe (not needed by ProbingClassifier API).
    probe_type supports 'linear' or 'mlp' (not 'attention' like advanced version).
    """

    probe_type: str = "linear"   # "linear" | "mlp"
    n_epochs: int = 10
    lr: float = 1e-3
    hidden_dim: int = 128
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# MLPProbe — 2-layer MLP with GELU + Dropout
# Distinct from probing_advanced.MLPProbe which uses ReLU without Dropout
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """Two-layer MLP probe with GELU activation and Dropout.

    Architecture: Linear -> GELU -> Dropout -> Linear.
    Forward: (B, input_dim) -> (B, n_classes) logits.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, input_dim) -> (B, n_classes) logits."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Layer representation extraction
# ---------------------------------------------------------------------------

def extract_layer_representations(
    model: nn.Module,
    input_ids: Tensor,
    layer_indices: list[int],
) -> dict[int, Tensor]:
    """Extract hidden states from specified transformer layers via forward hooks.

    Args:
        model: AureliusTransformer with a model.layers ModuleList.
        input_ids: (B, T) token ids.
        layer_indices: List of layer indices to capture.

    Returns:
        Dict mapping layer_idx -> Tensor of shape (B, T, D).
    """
    captured: dict[int, Tensor] = {}
    hooks = []

    for idx in layer_indices:
        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                # TransformerBlock returns (hidden, kv) tuple; capture hidden state
                out = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer_idx] = out.detach()
            return hook

        hooks.append(model.layers[idx].register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured


# ---------------------------------------------------------------------------
# ProbingDataset
# ---------------------------------------------------------------------------

class ProbingDataset:
    """Dataset of (representation, label) pairs for probing classifiers.

    Does NOT inherit from torch.utils.data.Dataset - intentionally plain Python
    to keep probing lightweight and avoid requiring DataLoader machinery.

    Args:
        representations: Tensor of shape (N, D).
        labels: Integer label tensor of shape (N,).
    """

    def __init__(self, representations: Tensor, labels: Tensor) -> None:
        if representations.shape[0] != labels.shape[0]:
            raise ValueError(
                f"representations and labels must have the same first dimension, "
                f"got {representations.shape[0]} vs {labels.shape[0]}"
            )
        self.representations = representations
        self.labels = labels

    def __len__(self) -> int:
        return self.representations.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.representations[idx], self.labels[idx]

    def split(self, ratio: float = 0.8) -> tuple["ProbingDataset", "ProbingDataset"]:
        """Split dataset into train/val subsets.

        Args:
            ratio: Fraction of data for training (default 0.8).

        Returns:
            (train_dataset, val_dataset) tuple.
        """
        N = len(self)
        n_train = max(1, int(N * ratio))

        train_reps = self.representations[:n_train]
        train_labels = self.labels[:n_train]
        val_reps = self.representations[n_train:]
        val_labels = self.labels[n_train:]

        return (
            ProbingDataset(train_reps, train_labels),
            ProbingDataset(val_reps, val_labels),
        )


# ---------------------------------------------------------------------------
# ProbingClassifier
# ---------------------------------------------------------------------------

class ProbingClassifier:
    """Train and evaluate a linear or MLP probe on frozen representations.

    Args:
        config: ProbingConfig.
        input_dim: Dimensionality of input representations.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        config: ProbingConfig,
        input_dim: int,
        n_classes: int,
    ) -> None:
        self.config = config
        self.input_dim = input_dim
        self.n_classes = n_classes

        if config.probe_type == "mlp":
            self.probe: nn.Module = MLPProbe(
                input_dim=input_dim,
                n_classes=n_classes,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        else:
            self.probe = nn.Linear(input_dim, n_classes)

    def fit(self, train_data: ProbingDataset) -> list[float]:
        """Train the probe for n_epochs.

        Args:
            train_data: ProbingDataset with (N, D) representations and (N,) labels.

        Returns:
            List of per-epoch losses (length == config.n_epochs).
        """
        optimizer = optim.Adam(self.probe.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()
        N = len(train_data)
        loss_history: list[float] = []

        self.probe.train()
        for _epoch in range(self.config.n_epochs):
            perm = torch.randperm(N)
            X = train_data.representations[perm].detach()
            y = train_data.labels[perm]

            optimizer.zero_grad()
            logits = self.probe(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        self.probe.train(False)
        return loss_history

    def evaluate(self, data: ProbingDataset) -> dict:
        """Evaluate the probe on a dataset.

        Args:
            data: ProbingDataset.

        Returns:
            Dict with keys 'accuracy' (float in [0,1]) and 'loss' (float).
        """
        criterion = nn.CrossEntropyLoss()
        self.probe.eval()
        with torch.no_grad():
            X = data.representations.detach()
            y = data.labels
            logits = self.probe(X)
            loss = criterion(logits, y).item()
            preds = logits.argmax(dim=-1)
            accuracy = (preds == y).float().mean().item()
        return {"accuracy": accuracy, "loss": loss}

    def predict(self, x: Tensor) -> Tensor:
        """Return class predictions for input tensor.

        Args:
            x: (B, D) tensor of representations.

        Returns:
            (B,) integer tensor of predicted class indices.
        """
        self.probe.eval()
        with torch.no_grad():
            logits = self.probe(x.detach())
            return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# LayerwiseProbingResults
# ---------------------------------------------------------------------------

@dataclass
class LayerwiseProbingResults:
    """Results from probing all layers of a transformer model.

    Attributes:
        task_name: Human-readable name of the probing task.
        layer_accuracies: Dict mapping layer_idx -> probe accuracy on that layer.
        best_layer: Index of the layer with the highest accuracy.
        best_accuracy: Accuracy achieved at best_layer.
    """

    task_name: str
    layer_accuracies: dict[int, float]
    best_layer: int
    best_accuracy: float


# ---------------------------------------------------------------------------
# LayerwiseProber
# ---------------------------------------------------------------------------

class LayerwiseProber:
    """Probe every layer of a transformer model for a given task.

    Distinct from probing_advanced.MultiLayerProber: uses ProbingClassifier
    (which wraps ProbingDataset), returns a typed LayerwiseProbingResults
    dataclass, and provides mutual_information_estimate and rank_layers helpers.

    Args:
        model: AureliusTransformer with model.layers attribute.
        config: ProbingConfig controlling probe architecture and training.
    """

    def __init__(self, model: nn.Module, config: ProbingConfig) -> None:
        self.model = model
        self.config = config

    def probe_all_layers(
        self,
        input_ids: Tensor,
        labels: Tensor,
        task_name: str,
    ) -> LayerwiseProbingResults:
        """Extract representations from every layer, train a probe on each.

        Args:
            input_ids: (B, T) token ids.
            labels: (B,) integer class labels, one per sequence.
            task_name: Name for this probing task (stored in results).

        Returns:
            LayerwiseProbingResults with accuracies for all layers.
        """
        n_layers = len(list(self.model.layers))
        all_indices = list(range(n_layers))

        representations = extract_layer_representations(
            self.model, input_ids, all_indices
        )

        n_classes = int(labels.max().item()) + 1
        layer_accuracies: dict[int, float] = {}

        for idx in all_indices:
            hidden = representations[idx]   # (B, T, D)
            # Mean-pool over sequence dimension -> (B, D)
            pooled = hidden.mean(dim=1)

            dataset = ProbingDataset(pooled, labels)
            clf = ProbingClassifier(
                config=self.config,
                input_dim=pooled.shape[-1],
                n_classes=n_classes,
            )
            clf.fit(dataset)
            metrics = clf.evaluate(dataset)
            layer_accuracies[idx] = metrics["accuracy"]

        best_layer = max(layer_accuracies, key=lambda k: layer_accuracies[k])
        best_accuracy = layer_accuracies[best_layer]

        return LayerwiseProbingResults(
            task_name=task_name,
            layer_accuracies=layer_accuracies,
            best_layer=best_layer,
            best_accuracy=best_accuracy,
        )

    def mutual_information_estimate(self, reps: Tensor, labels: Tensor) -> float:
        """Estimate mutual information I(X; Y) via discretized entropy.

        Uses the normalized-entropy formulation: H(Y) - H(Y|X), where X is
        quantized into 10 equal-width bins along its mean projection.

        Args:
            reps: (N, D) representation tensor.
            labels: (N,) integer class labels.

        Returns:
            Non-negative MI estimate in nats (natural log).
        """
        N = reps.shape[0]
        n_bins = 10

        # Project to scalar via feature mean
        x_scalar = reps.float().mean(dim=-1)   # (N,)
        x_min = x_scalar.min().item()
        x_max = x_scalar.max().item()

        if x_max == x_min:
            return 0.0

        bin_edges = torch.linspace(x_min, x_max, n_bins + 1)
        bin_ids = torch.bucketize(x_scalar, bin_edges[1:-1])  # (N,) in [0, n_bins-1]

        y_long = labels.long()
        n_classes = int(y_long.max().item()) + 1

        # H(Y)
        py = torch.zeros(n_classes)
        for c in range(n_classes):
            py[c] = (y_long == c).float().sum() / N
        py_safe = py.clamp(min=1e-12)
        hy = -(py_safe * torch.log(py_safe)).sum().item()

        # H(Y | X_binned)
        hy_given_x = 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            n_b = mask.float().sum().item()
            if n_b == 0:
                continue
            p_b = n_b / N
            py_given_b = torch.zeros(n_classes)
            for c in range(n_classes):
                py_given_b[c] = ((y_long == c) & mask).float().sum() / n_b
            py_given_b_safe = py_given_b.clamp(min=1e-12)
            h_b = -(py_given_b_safe * torch.log(py_given_b_safe)).sum().item()
            hy_given_x += p_b * h_b

        mi = hy - hy_given_x
        return float(max(0.0, mi))   # clamp to non-negative

    def rank_layers(
        self, results: LayerwiseProbingResults
    ) -> list[tuple[int, float]]:
        """Return layers sorted by probe accuracy in descending order.

        Args:
            results: LayerwiseProbingResults from probe_all_layers.

        Returns:
            List of (layer_idx, accuracy) tuples, highest accuracy first.
        """
        return sorted(
            results.layer_accuracies.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
