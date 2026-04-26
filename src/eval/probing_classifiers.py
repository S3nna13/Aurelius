"""Probing classifiers: train simple probes on frozen hidden states to test layer representations.

Provides LinearProbe, MLPProbe, ProbingEvaluator, and ProbingResult for systematic
layer-wise representation analysis of transformer models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim


class LinearProbe(nn.Module):
    """Single linear layer probe trained on frozen hidden states.

    Args:
        d_model: Dimensionality of input features.
        n_classes: Number of output classes.
        bias: Whether to include a bias term.
    """

    def __init__(self, d_model: int, n_classes: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional mean-pooling over sequence dimension.

        Args:
            x: Input tensor of shape (B, d_model) or (B, T, d_model).

        Returns:
            Logits of shape (B, n_classes).
        """
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.linear(x)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> LinearProbe:
        """Train the probe using Adam optimizer with cross-entropy loss.

        Args:
            features: Input features of shape (N, d_model) or (N, T, d_model).
            labels: Integer class labels of shape (N,).
            n_epochs: Number of training epochs.
            lr: Learning rate for Adam optimizer.

        Returns:
            self (for chaining).
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = self(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        self.eval()
        return self

    @torch.no_grad()
    def accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy.

        Args:
            features: Input features of shape (N, d_model) or (N, T, d_model).
            labels: Integer class labels of shape (N,).

        Returns:
            Accuracy as a float in [0, 1].
        """
        self.eval()
        logits = self(features)
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()


class MLPProbe(nn.Module):
    """Two-layer MLP probe trained on frozen hidden states.

    Architecture: Linear -> ReLU -> Dropout -> Linear

    Args:
        d_model: Dimensionality of input features.
        hidden_dim: Dimensionality of the hidden layer.
        n_classes: Number of output classes.
        dropout: Dropout probability (0.0 = no dropout).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional mean-pooling over sequence dimension.

        Args:
            x: Input tensor of shape (B, d_model) or (B, T, d_model).

        Returns:
            Logits of shape (B, n_classes).
        """
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> MLPProbe:
        """Train the probe using Adam optimizer with cross-entropy loss.

        Args:
            features: Input features of shape (N, d_model) or (N, T, d_model).
            labels: Integer class labels of shape (N,).
            n_epochs: Number of training epochs.
            lr: Learning rate for Adam optimizer.

        Returns:
            self (for chaining).
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = self(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        self.eval()
        return self

    @torch.no_grad()
    def accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy.

        Args:
            features: Input features of shape (N, d_model) or (N, T, d_model).
            labels: Integer class labels of shape (N,).

        Returns:
            Accuracy as a float in [0, 1].
        """
        self.eval()
        logits = self(features)
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()


class ProbingEvaluator:
    """Evaluates probes across transformer layers using 80/20 train/val splits.

    Args:
        probe_cls: Probe class to instantiate (LinearProbe or MLPProbe).
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        probe_cls: type[nn.Module] = LinearProbe,
        n_classes: int = 2,
    ) -> None:
        self.probe_cls = probe_cls
        self.n_classes = n_classes

    def evaluate_layer(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 100,
    ) -> dict[str, float]:
        """Evaluate a probe on a single layer's features with 80/20 split.

        Args:
            features: Input features of shape (N, d_model) or (N, T, d_model).
            labels: Integer class labels of shape (N,).
            n_epochs: Number of training epochs.

        Returns:
            Dict with keys 'train_acc', 'val_acc', 'n_params'.
        """
        n = len(features)
        n_train = int(0.8 * n)

        train_features = features[:n_train]
        train_labels = labels[:n_train]
        val_features = features[n_train:]
        val_labels = labels[n_train:]

        # Determine d_model from feature shape
        if features.dim() == 3:
            d_model = features.shape[2]
        else:
            d_model = features.shape[1]

        # Instantiate probe; MLPProbe requires hidden_dim argument
        if self.probe_cls is MLPProbe:
            probe = self.probe_cls(d_model, d_model // 2 or 1, self.n_classes)
        else:
            probe = self.probe_cls(d_model, self.n_classes)

        probe.fit(train_features, train_labels, n_epochs=n_epochs)

        train_acc = probe.accuracy(train_features, train_labels)
        val_acc = probe.accuracy(val_features, val_labels)
        n_params = sum(p.numel() for p in probe.parameters())

        return {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "n_params": float(n_params),
        }

    def evaluate_all_layers(
        self,
        layer_features: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> list[dict[str, float]]:
        """Evaluate probes across all layers.

        Args:
            layer_features: List of feature tensors, one per layer.
            labels: Integer class labels of shape (N,).

        Returns:
            List of result dicts (one per layer), each with keys
            'train_acc', 'val_acc', 'n_params'.
        """
        return [self.evaluate_layer(features, labels) for features in layer_features]


@dataclass
class ProbingResult:
    """Result container for a single layer's probing evaluation.

    Attributes:
        layer_idx: Index of the transformer layer.
        train_acc: Training set accuracy.
        val_acc: Validation set accuracy.
        n_params: Number of trainable parameters in the probe.
    """

    layer_idx: int
    train_acc: float
    val_acc: float
    n_params: int

    def is_significant(
        self,
        chance_level: float = 0.5,
        threshold: float = 0.1,
    ) -> bool:
        """Check whether val_acc is significantly above chance.

        Args:
            chance_level: Expected accuracy for a random classifier.
            threshold: Minimum margin above chance to be considered significant.

        Returns:
            True if val_acc > chance_level + threshold, else False.
        """
        return self.val_acc > chance_level + threshold
