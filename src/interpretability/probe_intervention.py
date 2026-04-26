"""
Probe Intervention for the Aurelius LLM project.

Implements the methodology from Hernandez et al. 2023, "Linearity of Relation
Decoding in Transformer LMs":
  1. Train a linear probe to predict a concept from internal representations.
  2. Intervene by editing activations along the probe direction.
  3. If patching along the direction changes model behavior, the representation
     is causal (not merely correlational).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    """Configuration for probe intervention experiments."""

    d_model: int = 64
    n_classes: int = 2
    lr: float = 0.01
    n_epochs: int = 100


# ---------------------------------------------------------------------------
# Linear Probe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Linear probe: maps hidden states to class logits.

    Supports both 2-D inputs (batch, d_model) and 3-D inputs
    (batch, seq, d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.linear = nn.Linear(d_model, n_classes, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch, d_model) or (batch, seq, d_model)

        Returns:
            logits: (batch, n_classes) or (batch, seq, n_classes)
        """
        return self.linear(hidden_states)

    def get_direction(self) -> Tensor:
        """Return the probe direction for binary probes.

        For a binary probe (n_classes=2), the decision boundary is determined
        by the difference of the two class weight vectors.  This difference
        (W[1] - W[0]) points from class-0 towards class-1 in representation
        space and is normalised to unit length.

        Returns:
            direction: (d_model,) unit-norm tensor.

        Raises:
            NotImplementedError: for probes with n_classes != 2.
        """
        if self.n_classes != 2:
            raise NotImplementedError(
                "get_direction is only defined for binary probes (n_classes=2). "
                f"This probe has n_classes={self.n_classes}."
            )
        # Weight matrix shape: (n_classes, d_model)
        w = self.linear.weight  # (2, d_model)
        direction = w[1] - w[0]  # (d_model,)
        norm = direction.norm()
        if norm < 1e-12:
            return direction
        return direction / norm


# ---------------------------------------------------------------------------
# Probe Trainer
# ---------------------------------------------------------------------------


class ProbeTrainer:
    """Trains a LinearProbe with Adam and evaluates it."""

    def __init__(
        self,
        probe: LinearProbe,
        lr: float = 0.01,
        n_epochs: int = 100,
    ) -> None:
        self.probe = probe
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X: Tensor, y: Tensor) -> dict:
        """Train the probe on (X, y).

        Args:
            X: (n_samples, d_model) hidden-state matrix.
            y: (n_samples,) integer class labels.

        Returns:
            dict with keys 'train_accuracy' and 'final_loss'.
        """
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=self.lr)

        self.probe.train()
        final_loss = float("inf")
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.probe(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.probe.eval()
        train_accuracy = self.evaluate(X, y)
        return {
            "train_accuracy": train_accuracy,
            "final_loss": final_loss,
        }

    def evaluate(self, X: Tensor, y: Tensor) -> float:
        """Compute classification accuracy.

        Args:
            X: (n_samples, d_model) hidden-state matrix.
            y: (n_samples,) integer class labels.

        Returns:
            Accuracy in [0, 1].
        """
        self.probe.eval()
        with torch.no_grad():
            logits = self.probe(X)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == y).float().mean().item()
        return accuracy


# ---------------------------------------------------------------------------
# Probe Intervention Experiment
# ---------------------------------------------------------------------------


class ProbeInterventionExperiment:
    """Combines representation extraction, causal intervention, and scrubbing.

    Works with AureliusTransformer (or any nn.Module whose ``layers``
    attribute is an iterable of transformer blocks accepting
    ``(x, freqs_cis, mask, past_kv)`` and returning ``(x, kv)``).
    """

    def __init__(
        self,
        model: nn.Module,
        probe: LinearProbe,
        layer_idx: int,
    ) -> None:
        self.model = model
        self.probe = probe
        self.layer_idx = layer_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_to_layer(
        self,
        input_ids: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """Run the model up to and including ``layer_idx``.

        Returns residual stream tensor of shape (batch, seq, d_model).
        """
        with torch.no_grad():
            B, S = input_ids.shape
            x = self.model.embed(input_ids)
            freqs_cis = self.model.freqs_cis[:S]
            for i, layer in enumerate(self.model.layers):
                x, _ = layer(x, freqs_cis, None, None)
                if i == layer_idx:
                    break
        return x  # (batch, seq, d_model)

    def _run_from_layer(
        self,
        x: Tensor,
        start_layer_idx: int,
    ) -> Tensor:
        """Continue running the model from ``start_layer_idx + 1`` to end.

        Args:
            x: (batch, seq, d_model) residual stream.
            start_layer_idx: the last layer already applied.

        Returns:
            logits: (batch, seq, vocab_size)
        """
        with torch.no_grad():
            S = x.shape[1]
            freqs_cis = self.model.freqs_cis[:S]
            for i, layer in enumerate(self.model.layers):
                if i <= start_layer_idx:
                    continue
                x, _ = layer(x, freqs_cis, None, None)
            x = self.model.norm(x)
            logits = self.model.lm_head(x)
        return logits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_representations(
        self,
        input_ids: Tensor,
        layer_idx: int | None = None,
    ) -> Tensor:
        """Extract activations at the specified layer, mean-pooled over seq.

        Args:
            input_ids: (batch, seq) token ids.
            layer_idx: which layer to extract from; defaults to self.layer_idx.

        Returns:
            activations: (batch, d_model)
        """
        if layer_idx is None:
            layer_idx = self.layer_idx
        x = self._run_to_layer(input_ids, layer_idx)  # (B, S, d_model)
        return x.mean(dim=1)  # (B, d_model)

    def intervention_effect(
        self,
        input_ids: Tensor,
        intervention_strength: float = 1.0,
        metric_fn: Callable | None = None,
    ) -> float:
        """Measure the causal effect of patching along the probe direction.

        Adds ``intervention_strength * probe_direction`` to all token
        positions at ``self.layer_idx``, then runs the remaining layers and
        measures the change in ``metric_fn(logits)``.

        Args:
            input_ids: (batch, seq) token ids.
            intervention_strength: scalar multiplier for the direction patch.
            metric_fn: callable that maps logits (batch, seq, vocab) -> scalar.
                       Defaults to mean top-token probability at the last position.

        Returns:
            delta: metric(patched_logits) - metric(clean_logits), a float.
        """
        if metric_fn is None:

            def metric_fn(logits: Tensor) -> float:
                probs = logits[:, -1, :].softmax(dim=-1)
                top_prob = probs.max(dim=-1).values
                return top_prob.mean().item()

        # Clean forward
        x_clean = self._run_to_layer(input_ids, self.layer_idx)
        clean_logits = self._run_from_layer(x_clean.clone(), self.layer_idx)
        clean_metric = metric_fn(clean_logits)

        # Patched forward: add direction to every token position
        direction = self.probe.get_direction()  # (d_model,)
        x_patched = x_clean + intervention_strength * direction.unsqueeze(0).unsqueeze(0)
        patched_logits = self._run_from_layer(x_patched, self.layer_idx)
        patched_metric = metric_fn(patched_logits)

        return float(patched_metric - clean_metric)

    def causal_scrubbing(
        self,
        clean_ids: Tensor,
        corrupted_ids: Tensor,
    ) -> float:
        """Estimate how much of the clean-corrupted difference the probe explains.

        Replaces only the component of the clean activation that lies along
        the probe direction with the corresponding component from the corrupted
        activation, then measures how much of the original clean-vs-corrupted
        output difference is recovered.

        Returns:
            fraction in (-inf, 1] where 1.0 means the probe fully explains the
            difference and 0.0 means it explains nothing.
        """
        x_clean = self._run_to_layer(clean_ids, self.layer_idx)  # (B, S, d_model)
        x_corrupted = self._run_to_layer(corrupted_ids, self.layer_idx)

        direction = self.probe.get_direction()  # (d_model,)
        d = direction  # unit vector

        # Project both representations onto the probe direction
        proj_clean = (x_clean * d).sum(dim=-1, keepdim=True)  # (B, S, 1)
        proj_corr = (x_corrupted * d).sum(dim=-1, keepdim=True)  # (B, S, 1)

        # Replace the clean probe-direction component with the corrupted one
        x_scrubbed = x_clean + (proj_corr - proj_clean) * d  # (B, S, d_model)

        # Compute logits for all three conditions
        clean_logits = self._run_from_layer(x_clean.clone(), self.layer_idx)
        corrupted_logits = self._run_from_layer(x_corrupted.clone(), self.layer_idx)
        scrubbed_logits = self._run_from_layer(x_scrubbed, self.layer_idx)

        def _scalar(logits: Tensor) -> float:
            return logits[:, -1, :].softmax(dim=-1).max(dim=-1).values.mean().item()

        clean_val = _scalar(clean_logits)
        corr_val = _scalar(corrupted_logits)
        scrubbed_val = _scalar(scrubbed_logits)

        denom = clean_val - corr_val
        if abs(denom) < 1e-12:
            return 0.0

        fraction = (scrubbed_val - corr_val) / denom
        return float(fraction)
