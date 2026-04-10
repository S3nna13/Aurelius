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


# ===========================================================================
# Hard Structured Pruning — removes neurons/dimensions entirely
# ===========================================================================
# The functions below complement the soft-masking approach above by
# physically shrinking weight tensors so inference actually speeds up.

@dataclass
class PruningConfig:
    """Configuration for hard structured pruning."""

    pruning_type: str = "neuron"    # "neuron" | "head" | "layer"
    prune_ratio: float = 0.3        # fraction to prune
    criterion: str = "magnitude"    # "magnitude" | "random"
    min_remaining: int = 1          # minimum units to keep


def score_neurons(weight: Tensor) -> Tensor:
    """Score neurons (rows of weight matrix) by L1 norm of their weights.

    Args:
        weight: (out_features, in_features) — a Linear layer's weight.

    Returns:
        scores: (out_features,) — non-negative L1 norm per row.
    """
    return weight.abs().sum(dim=1)


def score_heads(attn_weight: Tensor, n_heads: int) -> Tensor:
    """Score attention heads by mean L1 norm of their weight rows.

    Args:
        attn_weight: (d_model, d_model) — e.g. q_proj weight.
        n_heads: number of attention heads.

    Returns:
        scores: (n_heads,) — mean L1 norm per head slice.
    """
    out_features = attn_weight.shape[0]
    head_size = out_features // n_heads
    scores = torch.zeros(n_heads, device=attn_weight.device, dtype=attn_weight.dtype)
    for h in range(n_heads):
        rows = attn_weight[h * head_size: (h + 1) * head_size, :]
        scores[h] = rows.abs().mean()
    return scores


def get_prune_mask(scores: Tensor, prune_ratio: float, min_remaining: int = 1) -> Tensor:
    """Return boolean mask: True = keep, False = prune.

    Prunes the lowest-scoring fraction while keeping at least min_remaining.

    Args:
        scores: 1-D tensor of importance scores.
        prune_ratio: Fraction of units to prune.
        min_remaining: Minimum number of units that must remain.

    Returns:
        mask: bool tensor of same shape as scores; True = keep.
    """
    n = scores.shape[0]
    n_prune = int(n * prune_ratio)
    max_prune = n - min_remaining
    n_prune = max(0, min(n_prune, max_prune))

    if n_prune == 0:
        return torch.ones(n, dtype=torch.bool, device=scores.device)

    # Indices of the lowest-scoring units
    sorted_indices = scores.argsort()
    prune_indices = sorted_indices[:n_prune]

    mask = torch.ones(n, dtype=torch.bool, device=scores.device)
    mask[prune_indices] = False
    return mask


def prune_linear_neurons(
    linear: nn.Linear,
    prune_ratio: float,
    criterion: str = "magnitude",
) -> tuple[nn.Linear, Tensor]:
    """Create a new smaller Linear by removing lowest-scoring output neurons.

    This performs hard structural pruning — the returned linear has fewer
    output features than the original.

    Args:
        linear: Source nn.Linear module.
        prune_ratio: Fraction of output neurons to remove.
        criterion: "magnitude" or "random".

    Returns:
        (pruned_linear, kept_indices): a new nn.Linear with fewer rows and
        the 1-D tensor of original row indices that were kept.
    """
    weight = linear.weight.data  # (out, in)

    if criterion == "random":
        scores = torch.rand(weight.shape[0], device=weight.device)
    else:
        scores = score_neurons(weight)

    mask = get_prune_mask(scores, prune_ratio, min_remaining=1)
    kept_indices = mask.nonzero(as_tuple=True)[0]

    new_out = kept_indices.shape[0]
    new_linear = nn.Linear(linear.in_features, new_out, bias=linear.bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(weight[kept_indices])
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.data[kept_indices])

    return new_linear, kept_indices


def prune_ffn_layer(
    model: nn.Module,
    layer_idx: int,
    prune_ratio: float,
) -> dict:
    """Prune FFN neurons in a specific transformer layer (hard pruning).

    Scores output neurons of gate_proj by L1 magnitude, then removes the
    lowest-scoring neurons from gate_proj and up_proj (output dimension) and
    the corresponding input neurons from down_proj, so all dimensions remain
    consistent.  Replaces the FFN sub-modules in-place.

    Args:
        model: AureliusTransformer instance.
        layer_idx: Index of the transformer layer to prune.
        prune_ratio: Fraction of FFN intermediate neurons to remove.

    Returns:
        Dict with keys:
            'original_dim': int — d_ff before pruning.
            'pruned_dim': int — d_ff after pruning.
            'n_pruned': int — number of neurons removed.
    """
    ffn = model.layers[layer_idx].ffn
    gate_proj: nn.Linear = ffn.gate_proj
    up_proj: nn.Linear = ffn.up_proj
    down_proj: nn.Linear = ffn.down_proj

    original_dim = gate_proj.out_features

    # Score neurons by gate_proj output rows
    scores = score_neurons(gate_proj.weight.data)
    mask = get_prune_mask(scores, prune_ratio, min_remaining=1)
    kept = mask.nonzero(as_tuple=True)[0]
    pruned_dim = kept.shape[0]
    n_pruned = original_dim - pruned_dim

    with torch.no_grad():
        # gate_proj: (d_ff, d_model) -> (pruned_dim, d_model)
        new_gate = nn.Linear(gate_proj.in_features, pruned_dim, bias=gate_proj.bias is not None)
        new_gate.weight.copy_(gate_proj.weight.data[kept])
        if gate_proj.bias is not None:
            new_gate.bias.copy_(gate_proj.bias.data[kept])

        # up_proj: same shape as gate_proj
        new_up = nn.Linear(up_proj.in_features, pruned_dim, bias=up_proj.bias is not None)
        new_up.weight.copy_(up_proj.weight.data[kept])
        if up_proj.bias is not None:
            new_up.bias.copy_(up_proj.bias.data[kept])

        # down_proj: (d_model, d_ff) -> (d_model, pruned_dim)
        new_down = nn.Linear(pruned_dim, down_proj.out_features, bias=down_proj.bias is not None)
        new_down.weight.copy_(down_proj.weight.data[:, kept])
        if down_proj.bias is not None:
            new_down.bias.copy_(down_proj.bias.data)

    ffn.gate_proj = new_gate
    ffn.up_proj = new_up
    ffn.down_proj = new_down

    return {
        "original_dim": original_dim,
        "pruned_dim": pruned_dim,
        "n_pruned": n_pruned,
    }


class PruningScheduler:
    """Gradually increases pruning ratio over training steps."""

    def __init__(
        self,
        config: PruningConfig,
        start_step: int = 0,
        end_step: int = 1000,
        start_ratio: float = 0.0,
    ) -> None:
        self.config = config
        self.start_step = start_step
        self.end_step = end_step
        self.start_ratio = start_ratio

    def get_ratio(self, step: int) -> float:
        """Return pruning ratio at given step (linearly interpolated).

        Returns start_ratio before start_step and config.prune_ratio after
        end_step, with a linear ramp between.
        """
        if step <= self.start_step:
            return self.start_ratio
        if step >= self.end_step:
            return self.config.prune_ratio
        t = (step - self.start_step) / max(1, self.end_step - self.start_step)
        return self.start_ratio + t * (self.config.prune_ratio - self.start_ratio)

    def should_prune(self, step: int, prune_every: int = 100) -> bool:
        """Return True if this step should trigger pruning."""
        return step > 0 and step % prune_every == 0


class StructuredPruner:  # type: ignore[no-redef]
    """Unified structured pruner supporting both soft-masking and hard-pruning.

    When constructed with a ``PruningConfig`` and an ``optimizer`` (three-arg
    form) it acts as a hard-pruning training-loop helper.

    When constructed with a ``StructuredPruningConfig`` and no optimizer (the
    legacy two-arg form) it delegates to the soft-masking helpers
    (HeadImportanceScorer, FFNImportanceScorer, apply_head_mask, apply_ffn_mask)
    and exposes ``calibrate``, ``prune``, and ``recover_accuracy`` so that
    existing tests continue to work unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        config,  # PruningConfig or StructuredPruningConfig
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

        if isinstance(config, PruningConfig):
            # Hard-pruning mode
            self._mode = "hard"
            self._scheduler = PruningScheduler(config)
        else:
            # Soft-masking mode — legacy StructuredPruningConfig
            self._mode = "soft"
            self._head_scorer = HeadImportanceScorer(model, config)
            self._ffn_scorer = FFNImportanceScorer(model, config)

    # ------------------------------------------------------------------
    # Soft-masking interface (legacy, StructuredPruningConfig)
    # ------------------------------------------------------------------

    def calibrate(self, data: list[Tensor]) -> dict:
        """Compute importance scores for heads and FFN neurons (soft mode)."""
        if self._mode != "soft":
            raise RuntimeError("calibrate() is only available in soft-masking mode")
        head_importance = self._head_scorer.compute_head_importance(data)
        ffn_importance = self._ffn_scorer.compute_neuron_importance(data)
        return {"head_importance": head_importance, "ffn_importance": ffn_importance}

    def prune(self, calibration_data: list[Tensor]) -> dict:
        """Run calibration, determine what to prune, and apply masks (soft mode)."""
        if self._mode != "soft":
            raise RuntimeError("prune() is only available in soft-masking mode")
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
        """Finetune model for n_steps to recover accuracy after pruning (soft mode)."""
        import math as _math
        import torch.nn.functional as _F
        model.train()
        final_loss = float("inf")
        data_cycle = finetune_data * (_math.ceil(n_steps / max(1, len(finetune_data))))
        for step in range(n_steps):
            batch = data_cycle[step % len(data_cycle)]
            optimizer.zero_grad()
            try:
                _, logits, _ = model(batch)
                B, T, V = logits.shape
                loss = _F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    batch[:, 1:].reshape(-1),
                )
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
            except Exception as e:
                logger.warning("recover_accuracy step %d failed: %s", step, e)
        return final_loss

    def prune_step(self, step: int) -> dict:
        """Prune model if scheduled.  Returns pruning stats dict."""
        ratio = self._scheduler.get_ratio(step)
        if ratio <= 0.0:
            return {"pruned": False, "step": step, "ratio": ratio}

        stats: dict = {"pruned": True, "step": step, "ratio": ratio, "layers": []}

        if self.config.pruning_type == "neuron":
            if hasattr(self.model, "layers"):
                for idx in range(len(self.model.layers)):
                    try:
                        layer_stats = prune_ffn_layer(self.model, idx, ratio)
                        stats["layers"].append(layer_stats)
                    except Exception as exc:
                        logger.warning("prune_ffn_layer layer %d failed: %s", idx, exc)

        return stats

    def train_step(self, input_ids: Tensor) -> dict:
        """Forward + backward + optimizer step.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            Dict with keys: 'loss' (float), 'total_params' (int), 'sparsity' (float).
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, _logits, _pkv = self.model(input_ids)

        if loss is None:
            import torch.nn.functional as _F
            B, T, V = _logits.shape
            loss = _F.cross_entropy(
                _logits[:, :-1].reshape(-1, V),
                input_ids[:, 1:].reshape(-1),
            )

        loss.backward()
        self.optimizer.step()

        total_params = self.parameter_count()
        return {
            "loss": loss.item(),
            "total_params": total_params,
            "sparsity": self.sparsity(),
        }

    def parameter_count(self) -> int:
        """Count current trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def sparsity(self) -> float:
        """Return 0.0 — hard pruning removes params entirely, no sparsity ratio."""
        return 0.0
