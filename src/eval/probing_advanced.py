"""Advanced probing: nonlinear probes, multi-layer analysis, and mutual information estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ProbingConfig:
    """Configuration for advanced probing experiments."""

    probe_type: str = "linear"  # "linear" | "mlp" | "attention"
    hidden_dim: int = 128
    n_layers_to_probe: list[int] | None = None  # None = all layers
    n_epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32


class LinearProbeV2(nn.Module):
    """Single linear probe: (B, input_dim) -> (B, n_classes) logits."""

    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, input_dim) -> (B, n_classes) logits."""
        return self.linear(x)


class MLPProbe(nn.Module):
    """Two-layer MLP probe: Linear -> ReLU -> Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, input_dim) -> (B, n_classes) logits."""
        return self.net(x)


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
        Dict mapping layer_idx -> hidden_state Tensor of shape (B, T, D).
    """
    captured: dict[int, Tensor] = {}
    hooks = []

    for idx in layer_indices:
        # TransformerBlock.forward returns (x, kv) -- take o[0] which is the hidden state
        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                captured[layer_idx] = output[0].detach()

            return hook

        hooks.append(model.layers[idx].register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured


def train_probe(probe: nn.Module, X: Tensor, y: Tensor, config: ProbingConfig) -> dict:
    """Train a probe via SGD with cross-entropy loss.

    Args:
        probe: nn.Module (LinearProbeV2 or MLPProbe).
        X: (N, D) feature tensor.
        y: (N,) integer class labels.
        config: ProbingConfig controlling training hyperparameters.

    Returns:
        Dict with keys train_acc (float) and final_loss (float).
    """
    optimizer = torch.optim.SGD(probe.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    N = X.shape[0]

    probe.train()
    final_loss = 0.0
    for _epoch in range(config.n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for start in range(0, N, config.batch_size):
            idx = perm[start : start + config.batch_size]
            x_batch = X[idx].detach()
            y_batch = y[idx]
            optimizer.zero_grad()
            logits = probe(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss

    probe.train(False)
    with torch.no_grad():
        logits = probe(X.detach())
        preds = logits.argmax(dim=-1)
        train_acc = (preds == y).float().mean().item()

    return {"train_acc": train_acc, "final_loss": final_loss}


def estimate_mutual_information(X: Tensor, y: Tensor, n_bins: int = 10) -> float:
    """Estimate mutual information I(X; Y) in bits using histogram binning.

    Projects X to a scalar via mean across features, bins into n_bins, then
    computes I(X_binned; Y) = H(Y) - H(Y | X_binned) empirically.

    Args:
        X: (N, D) feature tensor.
        y: (N,) integer class labels.
        n_bins: Number of bins for the scalar projection of X.

    Returns:
        MI estimate in bits (log2).
    """
    N = X.shape[0]
    # Project X to scalar by mean across feature dimension
    x_scalar = X.float().mean(dim=-1)  # (N,)

    # Bin the scalar projection
    x_min = x_scalar.min().item()
    x_max = x_scalar.max().item()
    if x_max == x_min:
        # No variation -- MI is 0
        return 0.0

    bin_edges = torch.linspace(x_min, x_max, n_bins + 1)
    bin_ids = torch.bucketize(x_scalar, bin_edges[1:-1])  # (N,) in [0, n_bins-1]

    y_long = y.long()
    n_classes = int(y_long.max().item()) + 1

    # H(Y)
    py = torch.zeros(n_classes)
    for c in range(n_classes):
        py[c] = (y_long == c).float().sum() / N
    py_safe = py.clamp(min=1e-12)
    hy = -(py_safe * torch.log2(py_safe)).sum().item()

    # H(Y | X_binned) = sum_b p(b) * H(Y | X=b)
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
        h_b = -(py_given_b_safe * torch.log2(py_given_b_safe)).sum().item()
        hy_given_x += p_b * h_b

    mi = hy - hy_given_x
    return float(mi)


class MultiLayerProber:
    """Probe a transformer model across multiple layers."""

    def __init__(self, model: nn.Module, config: ProbingConfig) -> None:
        self.model = model
        self.config = config

    def probe_all_layers(
        self,
        input_ids: Tensor,
        labels: Tensor,
        layer_indices: list[int],
    ) -> dict[int, dict]:
        """Extract representations and train a probe on each specified layer.

        Args:
            input_ids: (B, T) token ids.
            labels: (B,) integer class labels, one per sequence.
            layer_indices: Which layer indices to probe.

        Returns:
            Dict mapping layer_idx -> {"train_acc": float, "probe_type": str}.
        """
        representations = extract_layer_representations(self.model, input_ids, layer_indices)

        results: dict[int, dict] = {}
        for idx in layer_indices:
            hidden = representations[idx]  # (B, T, D)
            # Pool over sequence dimension -> (B, D)
            pooled = hidden.mean(dim=1)
            B, D = pooled.shape
            n_classes = int(labels.max().item()) + 1

            if self.config.probe_type == "mlp":
                probe = MLPProbe(D, self.config.hidden_dim, n_classes)
            else:
                probe = LinearProbeV2(D, n_classes)

            stats = train_probe(probe, pooled, labels, self.config)
            results[idx] = {
                "train_acc": stats["train_acc"],
                "probe_type": self.config.probe_type,
            }

        return results
