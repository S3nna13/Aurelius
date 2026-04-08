"""Linear probing: train simple classifiers on frozen hidden states to test layer-wise representations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class ProbeConfig:
    """Configuration for a linear probe classifier."""

    n_classes: int = 2
    d_model: int = 1024
    lr: float = 1e-3
    n_epochs: int = 10
    batch_size: int = 32


class LinearProbe(nn.Module):
    """Single linear layer trained on frozen hidden states.

    Usage:
        probe = LinearProbe(ProbeConfig(n_classes=5, d_model=64))
        probe.fit(hidden_states, labels)   # train
        acc = probe.evaluate(hidden_states, labels)
        preds = probe.predict(hidden_states)
    """

    def __init__(self, cfg: ProbeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d_model) -> (N, n_classes) logits."""
        return self.linear(x)

    def fit(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        verbose: bool = False,
    ) -> list[float]:
        """Train on (N, d_model) hidden states with (N,) integer labels.

        Returns list of per-epoch losses.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        criterion = nn.CrossEntropyLoss()
        N = hidden_states.shape[0]
        losses: list[float] = []
        self.train()
        for epoch in range(self.cfg.n_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            for start in range(0, N, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                x = hidden_states[idx].detach()
                y = labels[idx]
                optimizer.zero_grad()
                loss = criterion(self(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
            if verbose:
                print(f"  epoch {epoch}: loss={epoch_loss:.4f}")
        self.eval()
        return losses

    def evaluate(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> float:
        """Return accuracy on hidden_states/labels."""
        with torch.no_grad():
            logits = self(hidden_states.detach())
            preds = logits.argmax(dim=-1)
            return (preds == labels).float().mean().item()

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return (N,) predicted class indices."""
        with torch.no_grad():
            return self(hidden_states.detach()).argmax(dim=-1)


def extract_layer_hiddens(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Extract hidden states from layer `layer_idx` of AureliusTransformer.

    Returns (B, S, d_model) hidden states from the output of layers[layer_idx].
    Uses a forward hook.
    """
    hidden: list[torch.Tensor] = []

    # Hook into model.layers[layer_idx]
    hook = model.layers[layer_idx].register_forward_hook(
        lambda m, i, o: hidden.append(o[0].detach())  # o is (x, present_kv), take x
    )
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    return hidden[0]  # (B, S, d_model)


def probe_all_layers(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    n_layers: int,
    probe_cfg: ProbeConfig,
) -> dict[int, float]:
    """Train and evaluate a probe on each transformer layer.

    Args:
        model: AureliusTransformer
        input_ids: (N, S) token IDs
        labels: (N*S,) or (N,) labels -- if (N*S,), probe each token position
        n_layers: number of layers to probe (0..n_layers-1)
        probe_cfg: ProbeConfig

    Returns:
        dict mapping layer_idx -> accuracy
    """
    results: dict[int, float] = {}
    for layer_idx in range(n_layers):
        hiddens = extract_layer_hiddens(model, input_ids, layer_idx)
        B, S, D = hiddens.shape

        # Reshape: if labels is (N*S,), use all token positions; if (N,), use mean pool
        if labels.shape[0] == B * S:
            h_flat = hiddens.reshape(B * S, D)
            y = labels
        else:
            h_flat = hiddens.mean(dim=1)  # (B, D)
            y = labels

        probe = LinearProbe(
            ProbeConfig(
                n_classes=probe_cfg.n_classes,
                d_model=D,
                lr=probe_cfg.lr,
                n_epochs=probe_cfg.n_epochs,
                batch_size=probe_cfg.batch_size,
            )
        )
        probe.fit(h_flat, y)
        acc = probe.evaluate(h_flat, y)
        results[layer_idx] = acc

    return results
