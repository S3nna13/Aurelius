"""Structured pruning for LLMs.

Prunes entire attention heads and FFN neurons (not individual weights),
which actually speeds up inference unlike unstructured pruning.

Components:
  - StructuredPruningConfig: hyperparameters
  - HeadImportanceScorer: score and select attention heads to prune
  - FFNImportanceScorer: score and select FFN neurons to prune
  - apply_head_mask: zero out pruned head weights
  - apply_ffn_mask: zero out pruned FFN neuron weights
  - count_active_parameters: measure sparsity
  - StructuredPruner: high-level orchestrator
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StructuredPruningConfig:
    """Configuration for structured pruning."""

    head_pruning_ratio: float = 0.25
    """Fraction of attention heads to prune globally."""

    ffn_pruning_ratio: float = 0.25
    """Fraction of FFN intermediate neurons to prune per layer."""

    importance_metric: str = "magnitude"
    """How to score importance: 'magnitude' | 'gradient' | 'random'."""

    n_calibration_steps: int = 10
    """Number of calibration forward passes for gradient/magnitude scoring."""


# ---------------------------------------------------------------------------
# Head importance
# ---------------------------------------------------------------------------

class HeadImportanceScorer:
    """Compute and rank attention head importance."""

    def __init__(self, model: nn.Module, config: StructuredPruningConfig) -> None:
        self.model = model
        self.config = config

    def compute_head_importance(self, calibration_data: list[Tensor]) -> dict[int, Tensor]:
        """Compute per-head importance score for every layer.

        Args:
            calibration_data: List of (B, T) input token id tensors.

        Returns:
            Dict mapping layer_idx -> Tensor of shape (n_heads,).
        """
        metric = self.config.importance_metric
        layers = self.model.layers

        if metric == "magnitude":
            return self._magnitude_head_importance(layers)
        elif metric == "gradient":
            return self._gradient_head_importance(layers, calibration_data)
        elif metric == "random":
            return self._random_head_importance(layers)
        else:
            raise ValueError(f"Unknown importance_metric: {metric!r}")

    def _get_head_dim(self, attn: nn.Module) -> int:
        """Infer head_dim from q_proj weight."""
        if hasattr(attn, "head_dim"):
            return attn.head_dim
        if hasattr(attn, "q_proj"):
            n_heads = attn.n_heads if hasattr(attn, "n_heads") else 1
            return attn.q_proj.out_features // n_heads
        return 1

    def _get_n_heads(self, attn: nn.Module) -> int:
        if hasattr(attn, "n_heads"):
            return attn.n_heads
        if hasattr(attn, "q_proj"):
            head_dim = self._get_head_dim(attn)
            return attn.q_proj.out_features // head_dim
        return 1

    def _magnitude_head_importance(self, layers) -> dict[int, Tensor]:
        """Score each head by mean abs of its slice in o_proj weight rows."""
        result: dict[int, Tensor] = {}
        for i, layer in enumerate(layers):
            attn = layer.attn if hasattr(layer, "attn") else None
            if attn is None:
                continue
            n_heads = self._get_n_heads(attn)
            head_dim = self._get_head_dim(attn)
            scores = torch.zeros(n_heads)

            # Use o_proj columns: shape (d_model, n_heads * head_dim)
            # Each head occupies columns [h*head_dim:(h+1)*head_dim]
            if hasattr(attn, "o_proj"):
                w = attn.o_proj.weight.data  # (d_model, n_heads * head_dim)
                for h in range(n_heads):
                    col_slice = w[:, h * head_dim:(h + 1) * head_dim]
                    scores[h] = col_slice.abs().mean()
            elif hasattr(attn, "out_proj"):
                w = attn.out_proj.weight.data
                for h in range(n_heads):
                    col_slice = w[:, h * head_dim:(h + 1) * head_dim]
                    scores[h] = col_slice.abs().mean()
            else:
                # Fallback: use q_proj rows
                if hasattr(attn, "q_proj"):
                    w = attn.q_proj.weight.data  # (n_heads * head_dim, d_model)
                    for h in range(n_heads):
                        row_slice = w[h * head_dim:(h + 1) * head_dim, :]
                        scores[h] = row_slice.abs().mean()

            result[i] = scores
        return result

    def _gradient_head_importance(self, layers, calibration_data: list[Tensor]) -> dict[int, Tensor]:
        """Score heads by accumulated gradient magnitude of o_proj columns."""
        n_layers = len(layers)
        head_counts = []
        for layer in layers:
            attn = layer.attn if hasattr(layer, "attn") else None
            if attn is not None:
                head_counts.append(self._get_n_heads(attn))
            else:
                head_counts.append(0)

        grad_accum: dict[int, Tensor] = {
            i: torch.zeros(head_counts[i]) for i in range(n_layers) if head_counts[i] > 0
        }

        was_training = self.model.training
        self.model.train()

        n_steps = min(self.config.n_calibration_steps, len(calibration_data))
        for step, batch in enumerate(calibration_data[:n_steps]):
            self.model.zero_grad()
            try:
                _, logits, _ = self.model(batch)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    batch[:, 1:].reshape(-1),
                )
                loss.backward()
            except Exception:
                continue

            for i, layer in enumerate(layers):
                if i not in grad_accum:
                    continue
                attn = layer.attn if hasattr(layer, "attn") else None
                if attn is None:
                    continue
                n_heads = head_counts[i]
                head_dim = self._get_head_dim(attn)
                proj = None
                if hasattr(attn, "o_proj") and attn.o_proj.weight.grad is not None:
                    proj = attn.o_proj
                elif hasattr(attn, "out_proj") and attn.out_proj.weight.grad is not None:
                    proj = attn.out_proj

                if proj is not None:
                    g = proj.weight.grad  # (d_model, n_heads * head_dim)
                    for h in range(n_heads):
                        grad_accum[i][h] += g[:, h * head_dim:(h + 1) * head_dim].abs().mean().item()

        self.model.zero_grad()
        if not was_training:
            self.model.eval()

        return grad_accum

    def _random_head_importance(self, layers) -> dict[int, Tensor]:
        result: dict[int, Tensor] = {}
        for i, layer in enumerate(layers):
            attn = layer.attn if hasattr(layer, "attn") else None
            if attn is None:
                continue
            n_heads = self._get_n_heads(attn)
            result[i] = torch.rand(n_heads)
        return result

    def get_heads_to_prune(self, head_importance: dict[int, Tensor]) -> dict[int, list[int]]:
        """Select heads to prune using a global threshold on importance.

        Args:
            head_importance: Dict from compute_head_importance().

        Returns:
            Dict mapping layer_idx -> list of head indices to prune.
        """
        ratio = self.config.head_pruning_ratio
        if not head_importance:
            return {}

        # Flatten all scores with (layer, head) labels
        all_scores: list[tuple[float, int, int]] = []
        for layer_idx, scores in head_importance.items():
            for head_idx, score in enumerate(scores.tolist()):
                all_scores.append((score, layer_idx, head_idx))

        n_total = len(all_scores)
        n_prune = int(n_total * ratio)
        if n_prune == 0:
            return {layer_idx: [] for layer_idx in head_importance}

        # Sort ascending — lowest importance first
        all_scores.sort(key=lambda x: x[0])
        to_prune: dict[int, list[int]] = {layer_idx: [] for layer_idx in head_importance}
        for _, layer_idx, head_idx in all_scores[:n_prune]:
            to_prune[layer_idx].append(head_idx)

        return to_prune


# ---------------------------------------------------------------------------
# FFN importance
# ---------------------------------------------------------------------------

class FFNImportanceScorer:
    """Compute and rank FFN neuron importance."""

    def __init__(self, model: nn.Module, config: StructuredPruningConfig) -> None:
        self.model = model
        self.config = config

    def compute_neuron_importance(self, calibration_data: list[Tensor]) -> dict[int, Tensor]:
        """Compute per-neuron importance for each FFN layer.

        Args:
            calibration_data: List of (B, T) input token id tensors.

        Returns:
            Dict mapping layer_idx -> Tensor of shape (d_ff,).
        """
        metric = self.config.importance_metric
        layers = self.model.layers

        if metric == "magnitude":
            return self._magnitude_ffn_importance(layers)
        elif metric == "gradient":
            return self._gradient_ffn_importance(layers, calibration_data)
        elif metric == "random":
            return self._random_ffn_importance(layers)
        else:
            raise ValueError(f"Unknown importance_metric: {metric!r}")

    def _magnitude_ffn_importance(self, layers) -> dict[int, Tensor]:
        result: dict[int, Tensor] = {}
        for i, layer in enumerate(layers):
            ffn = layer.ffn if hasattr(layer, "ffn") else None
            if ffn is None or not hasattr(ffn, "gate_proj"):
                continue
            w = ffn.gate_proj.weight.data  # (d_ff, d_model)
            # Mean absolute value of each row (neuron)
            result[i] = w.abs().mean(dim=1)
        return result

    def _gradient_ffn_importance(self, layers, calibration_data: list[Tensor]) -> dict[int, Tensor]:
        n_layers = len(layers)
        d_ff_map: dict[int, int] = {}
        for i, layer in enumerate(layers):
            ffn = layer.ffn if hasattr(layer, "ffn") else None
            if ffn is not None and hasattr(ffn, "gate_proj"):
                d_ff_map[i] = ffn.gate_proj.out_features

        grad_accum: dict[int, Tensor] = {i: torch.zeros(d) for i, d in d_ff_map.items()}

        was_training = self.model.training
        self.model.train()

        n_steps = min(self.config.n_calibration_steps, len(calibration_data))
        for step, batch in enumerate(calibration_data[:n_steps]):
            self.model.zero_grad()
            try:
                _, logits, _ = self.model(batch)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    batch[:, 1:].reshape(-1),
                )
                loss.backward()
            except Exception:
                continue

            for i in grad_accum:
                layer = layers[i]
                ffn = layer.ffn if hasattr(layer, "ffn") else None
                if ffn is None or not hasattr(ffn, "gate_proj"):
                    continue
                g = ffn.gate_proj.weight.grad
                if g is not None:
                    grad_accum[i] += g.abs().mean(dim=1).detach()

        self.model.zero_grad()
        if not was_training:
            self.model.eval()

        return grad_accum

    def _random_ffn_importance(self, layers) -> dict[int, Tensor]:
        result: dict[int, Tensor] = {}
        for i, layer in enumerate(layers):
            ffn = layer.ffn if hasattr(layer, "ffn") else None
            if ffn is None or not hasattr(ffn, "gate_proj"):
                continue
            d_ff = ffn.gate_proj.out_features
            result[i] = torch.rand(d_ff)
        return result

    def get_neurons_to_prune(self, importance: dict[int, Tensor]) -> dict[int, list[int]]:
        """Select neuron indices to prune per layer.

        Args:
            importance: Dict from compute_neuron_importance().

        Returns:
            Dict mapping layer_idx -> list of neuron indices to prune.
        """
        ratio = self.config.ffn_pruning_ratio
        result: dict[int, list[int]] = {}
        for layer_idx, scores in importance.items():
            d_ff = scores.shape[0]
            n_prune = int(d_ff * ratio)
            if n_prune == 0:
                result[layer_idx] = []
                continue
            # Sort ascending — lowest importance first
            sorted_indices = scores.argsort()
            result[layer_idx] = sorted_indices[:n_prune].tolist()
        return result


# ---------------------------------------------------------------------------
# Apply masks
# ---------------------------------------------------------------------------

def apply_head_mask(model: nn.Module, heads_to_prune: dict[int, list[int]]) -> None:
    """Zero out weights of pruned attention heads in o_proj columns.

    For layer i, head j: zeros the head_dim-wide column slice of o_proj.weight
    at columns [j*head_dim : (j+1)*head_dim].

    Args:
        model: AureliusTransformer.
        heads_to_prune: Dict mapping layer_idx -> list of head indices.
    """
    for layer_idx, head_ids in heads_to_prune.items():
        if not head_ids:
            continue
        try:
            layer = model.layers[layer_idx]
        except (IndexError, AttributeError):
            continue
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue

        # Determine head_dim
        head_dim = 1
        if hasattr(attn, "head_dim"):
            head_dim = attn.head_dim
        elif hasattr(attn, "q_proj") and hasattr(attn, "n_heads"):
            head_dim = attn.q_proj.out_features // attn.n_heads

        # Zero out o_proj columns for each pruned head
        proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
        if proj is not None:
            with torch.no_grad():
                for h in head_ids:
                    start = h * head_dim
                    end = start + head_dim
                    if end <= proj.weight.shape[1]:
                        proj.weight[:, start:end] = 0.0

        # Also zero corresponding q_proj rows
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is not None:
            with torch.no_grad():
                for h in head_ids:
                    start = h * head_dim
                    end = start + head_dim
                    if end <= q_proj.weight.shape[0]:
                        q_proj.weight[start:end, :] = 0.0


def apply_ffn_mask(model: nn.Module, neurons_to_prune: dict[int, list[int]]) -> None:
    """Zero out FFN neuron weights for pruned neurons.

    Zeroes rows of gate_proj and up_proj, and corresponding columns of
    down_proj.

    Args:
        model: AureliusTransformer.
        neurons_to_prune: Dict mapping layer_idx -> list of neuron indices.
    """
    for layer_idx, neuron_ids in neurons_to_prune.items():
        if not neuron_ids:
            continue
        try:
            layer = model.layers[layer_idx]
        except (IndexError, AttributeError):
            continue
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue

        neuron_ids_t = torch.tensor(neuron_ids, dtype=torch.long)

        gate_proj = getattr(ffn, "gate_proj", None)
        up_proj = getattr(ffn, "up_proj", None)
        down_proj = getattr(ffn, "down_proj", None)

        with torch.no_grad():
            if gate_proj is not None:
                valid = neuron_ids_t[neuron_ids_t < gate_proj.weight.shape[0]]
                gate_proj.weight[valid] = 0.0

            if up_proj is not None:
                valid = neuron_ids_t[neuron_ids_t < up_proj.weight.shape[0]]
                up_proj.weight[valid] = 0.0

            if down_proj is not None:
                valid = neuron_ids_t[neuron_ids_t < down_proj.weight.shape[1]]
                down_proj.weight[:, valid] = 0.0


# ---------------------------------------------------------------------------
# Count active parameters
# ---------------------------------------------------------------------------

def count_active_parameters(model: nn.Module) -> dict:
    """Count total and non-zero parameters.

    Args:
        model: Any nn.Module.

    Returns:
        Dict with keys 'total' (int), 'nonzero' (int), 'sparsity' (float).
    """
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += p.count_nonzero().item()
    sparsity = 1.0 - (nonzero / total) if total > 0 else 0.0
    return {"total": total, "nonzero": nonzero, "sparsity": sparsity}


# ---------------------------------------------------------------------------
# High-level StructuredPruner
# ---------------------------------------------------------------------------

class StructuredPruner:
    """High-level orchestrator for structured pruning."""

    def __init__(self, model: nn.Module, config: StructuredPruningConfig) -> None:
        self.model = model
        self.config = config
        self._head_scorer = HeadImportanceScorer(model, config)
        self._ffn_scorer = FFNImportanceScorer(model, config)

    def calibrate(self, data: list[Tensor]) -> dict:
        """Compute importance scores for heads and FFN neurons.

        Args:
            data: List of (B, T) input token id tensors.

        Returns:
            Dict with keys 'head_importance' and 'ffn_importance'.
        """
        head_importance = self._head_scorer.compute_head_importance(data)
        ffn_importance = self._ffn_scorer.compute_neuron_importance(data)
        return {"head_importance": head_importance, "ffn_importance": ffn_importance}

    def prune(self, calibration_data: list[Tensor]) -> dict:
        """Run calibration, determine what to prune, and apply masks.

        Args:
            calibration_data: List of (B, T) input token id tensors.

        Returns:
            Dict with keys 'heads_pruned' (int), 'neurons_pruned' (int),
            'sparsity' (float).
        """
        importance = self.calibrate(calibration_data)

        heads_to_prune = self._head_scorer.get_heads_to_prune(importance["head_importance"])
        neurons_to_prune = self._ffn_scorer.get_neurons_to_prune(importance["ffn_importance"])

        apply_head_mask(self.model, heads_to_prune)
        apply_ffn_mask(self.model, neurons_to_prune)

        heads_pruned = sum(len(v) for v in heads_to_prune.values())
        neurons_pruned = sum(len(v) for v in neurons_to_prune.values())

        stats = count_active_parameters(self.model)
        return {
            "heads_pruned": heads_pruned,
            "neurons_pruned": neurons_pruned,
            "sparsity": stats["sparsity"],
        }

    def recover_accuracy(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        finetune_data: list[Tensor],
        n_steps: int = 5,
    ) -> float:
        """Finetune model for n_steps to recover accuracy after pruning.

        Args:
            model: The (already pruned) model.
            optimizer: Optimizer to use.
            finetune_data: List of (B, T) input token id tensors.
            n_steps: Number of gradient steps.

        Returns:
            Final training loss (float).
        """
        model.train()
        final_loss = float("inf")

        data_cycle = finetune_data * (math.ceil(n_steps / max(1, len(finetune_data))))

        for step in range(n_steps):
            batch = data_cycle[step % len(data_cycle)]
            optimizer.zero_grad()
            try:
                _, logits, _ = model(batch)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    batch[:, 1:].reshape(-1),
                )
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
            except Exception as e:
                logger.warning("recover_accuracy step %d failed: %s", step, e)
                continue

        return final_loss
