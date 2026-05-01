"""Selective gradient checkpointing for the Aurelius transformer.

Provides utilities to choose which layers to apply gradient checkpointing
to, based on memory budget, uniform spacing, or adaptive heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint


@dataclass
class CheckpointSelectorConfig:
    """Configuration for selective checkpointing.

    Attributes:
        strategy: One of "uniform", "memory_optimal", or "adaptive".
        checkpoint_ratio: Fraction of layers to checkpoint (used by uniform/adaptive).
        memory_budget_gb: Target peak activation memory in gigabytes.
        bytes_per_element: Bytes per tensor element (4 = fp32, 2 = fp16/bf16).
    """

    strategy: str = "uniform"  # "uniform" | "memory_optimal" | "adaptive"
    checkpoint_ratio: float = 0.5  # fraction of layers to checkpoint
    memory_budget_gb: float = 8.0
    bytes_per_element: int = 4  # fp32


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def estimate_activation_bytes(
    n_layers: int,
    seq_len: int,
    d_model: int,
    batch_size: int,
    bytes_per_element: int = 4,
) -> int:
    """Estimate total activation memory (bytes) without any checkpointing.

    Per layer we conservatively account for:
      - Attention map:          B * T * T         (quadratic in seq_len)
      - FFN intermediate:       B * T * (4 * d_model)  (gate + up projections)
      - Residual/norm tensors:  2 * B * T * d_model

    Returns total bytes across all layers.
    """
    per_layer = (
        batch_size * seq_len * seq_len  # attention scores
        + batch_size * seq_len * (4 * d_model)  # FFN intermediate (~d_ff)
        + 2 * batch_size * seq_len * d_model  # residuals / norms
    ) * bytes_per_element
    return n_layers * per_layer


# ---------------------------------------------------------------------------
# Layer selection helpers
# ---------------------------------------------------------------------------


def select_uniform_layers(n_layers: int, ratio: float) -> list[int]:
    """Return evenly-spaced layer indices to checkpoint.

    Args:
        n_layers: Total number of transformer layers.
        ratio: Fraction in [0, 1] of layers to checkpoint.

    Returns:
        Sorted list of layer indices (may be empty).
    """
    if ratio <= 0.0 or n_layers == 0:
        return []
    if ratio >= 1.0:
        return list(range(n_layers))

    n_to_checkpoint = max(1, round(ratio * n_layers))
    # Evenly space indices across [0, n_layers)
    indices: list[int] = []
    for k in range(n_to_checkpoint):
        idx = int(round(k * (n_layers - 1) / max(n_to_checkpoint - 1, 1)))
        indices.append(idx)

    # Deduplicate and sort
    return sorted(set(indices))


def select_memory_optimal_layers(
    n_layers: int,
    seq_len: int,
    d_model: int,
    batch_size: int,
    budget_bytes: int,
    bytes_per_element: int = 4,
) -> list[int]:
    """Greedily select layers to checkpoint until activation memory fits budget.

    Layers are ranked by their individual activation cost (all identical in the
    base transformer, so we use a tiebreak on layer index to be deterministic).
    Layers are checkpointed from highest-cost first until total remaining
    activation memory is within budget_bytes.

    Args:
        n_layers: Total transformer layers.
        seq_len: Input sequence length.
        d_model: Model hidden dimension.
        batch_size: Batch size.
        budget_bytes: Target activation memory budget in bytes.
        bytes_per_element: Bytes per fp element.

    Returns:
        Sorted list of layer indices to checkpoint.
    """
    if n_layers == 0:
        return []

    # Per-layer activation cost (same formula as estimate_activation_bytes)
    per_layer_cost = (
        batch_size * seq_len * seq_len
        + batch_size * seq_len * (4 * d_model)
        + 2 * batch_size * seq_len * d_model
    ) * bytes_per_element

    total_cost = n_layers * per_layer_cost
    if total_cost <= budget_bytes:
        return []

    # All layers have equal cost; checkpoint highest-indexed first (arbitrary stable ordering)
    checkpointed: set[int] = set()
    remaining_cost = total_cost

    # Sort by cost desc, tiebreak by idx desc (all same cost here)
    for idx in range(n_layers - 1, -1, -1):
        if remaining_cost <= budget_bytes:
            break
        checkpointed.add(idx)
        remaining_cost -= per_layer_cost

    return sorted(checkpointed)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def compute_recomputation_cost(n_checkpointed: int, n_layers: int) -> float:
    """Estimate extra recomputation cost fraction of the full forward pass.

    Each checkpointed layer's forward is run twice (once during forward,
    once during backward), so the overhead is n_checkpointed / n_layers.

    Returns:
        Float in [0.0, 1.0].
    """
    if n_layers == 0:
        return 0.0
    return min(1.0, n_checkpointed / n_layers)


# ---------------------------------------------------------------------------
# Checkpointing wrappers
# ---------------------------------------------------------------------------


class CheckpointedLayer(nn.Module):
    """Wraps an nn.Module layer to use gradient checkpointing on forward."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # torch.utils.checkpoint.checkpoint requires all non-tensor inputs to
        # be passed via use_reentrant=False path which supports kwargs.
        # We wrap the call in a plain function and pass tensor args through.
        def _fn(*a: Any, **kw: Any) -> Any:
            return self.layer(*a, **kw)

        result = torch.utils.checkpoint.checkpoint(_fn, *args, use_reentrant=False, **kwargs)
        if isinstance(result, tuple) and len(result) == 3 and isinstance(result[2], torch.Tensor):
            # Preserve the historical (output, kv) layer contract while dropping
            # the auxiliary loss term that checkpointing does not need to expose.
            return result[0], result[1]
        return result


def apply_selective_checkpointing(
    model: nn.Module,
    layer_indices: list[int],
) -> nn.Module:
    """Wrap selected layers in model.layers with CheckpointedLayer.

    Modifies model.layers in-place. Non-selected layers are left unchanged.

    Args:
        model: Model with a .layers attribute (nn.ModuleList).
        layer_indices: Indices of layers to wrap.

    Returns:
        The model (same object, modified in-place).
    """
    indices_set = set(layer_indices)
    for idx in indices_set:
        if idx < 0 or idx >= len(model.layers):
            raise IndexError(
                f"Layer index {idx} out of range for model with {len(model.layers)} layers"
            )
        original_layer = model.layers[idx]
        # Don't double-wrap
        if not isinstance(original_layer, CheckpointedLayer):
            model.layers[idx] = CheckpointedLayer(original_layer)
    return model


# ---------------------------------------------------------------------------
# High-level CheckpointSelector
# ---------------------------------------------------------------------------


class CheckpointSelector:
    """Selects and applies gradient checkpointing to a model's layers.

    Usage::

        selector = CheckpointSelector(model, CheckpointSelectorConfig())
        indices = selector.apply(seq_len=512, batch_size=4)
        savings = selector.estimate_memory_savings(seq_len=512, batch_size=4)
    """

    def __init__(self, model: nn.Module, cfg: CheckpointSelectorConfig) -> None:
        self.model = model
        self.cfg = cfg

    def _n_layers(self) -> int:
        return len(self.model.layers)

    def _d_model(self) -> int:
        """Attempt to read d_model from model config; fall back to a heuristic."""
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "d_model"):
            return cfg.d_model
        # Fallback: inspect embedding
        embed = getattr(self.model, "embed", None)
        if embed is not None and hasattr(embed, "embedding_dim"):
            return embed.embedding_dim
        raise AttributeError("Cannot determine d_model from model. Set model.config.d_model.")

    def select_layers(self, seq_len: int, batch_size: int) -> list[int]:
        """Determine which layers to checkpoint based on the configured strategy.

        Args:
            seq_len: Input sequence length.
            batch_size: Training batch size.

        Returns:
            Sorted list of layer indices to checkpoint.
        """
        n_layers = self._n_layers()
        strategy = self.cfg.strategy

        if strategy == "uniform":
            return select_uniform_layers(n_layers, self.cfg.checkpoint_ratio)

        elif strategy == "memory_optimal":
            budget_bytes = int(self.cfg.memory_budget_gb * (1024**3))
            d_model = self._d_model()
            return select_memory_optimal_layers(
                n_layers,
                seq_len,
                d_model,
                batch_size,
                budget_bytes,
                self.cfg.bytes_per_element,
            )

        elif strategy == "adaptive":
            # Adaptive: start with memory_optimal, then fall back to uniform if
            # not enough savings are achieved.
            budget_bytes = int(self.cfg.memory_budget_gb * (1024**3))
            d_model = self._d_model()
            optimal = select_memory_optimal_layers(
                n_layers,
                seq_len,
                d_model,
                batch_size,
                budget_bytes,
                self.cfg.bytes_per_element,
            )
            uniform = select_uniform_layers(n_layers, self.cfg.checkpoint_ratio)
            # Use whichever set checkpoints fewer layers (minimal recomputation)
            return optimal if len(optimal) <= len(uniform) else uniform

        else:
            raise ValueError(
                f"Unknown checkpoint strategy: {self.cfg.strategy!r}. "
                "Expected one of: 'uniform', 'memory_optimal', 'adaptive'."
            )

    def apply(self, seq_len: int, batch_size: int) -> list[int]:
        """Select layers and apply selective checkpointing to the model.

        Args:
            seq_len: Input sequence length.
            batch_size: Training batch size.

        Returns:
            Sorted list of layer indices that were checkpointed.
        """
        indices = self.select_layers(seq_len, batch_size)
        apply_selective_checkpointing(self.model, indices)
        return indices

    def estimate_memory_savings(self, seq_len: int, batch_size: int) -> dict[str, float]:
        """Estimate memory savings from checkpointing.

        Args:
            seq_len: Input sequence length.
            batch_size: Training batch size.

        Returns:
            Dict with keys:
                "baseline_gb"        — activation memory without checkpointing (GB)
                "saved_gb"           — activation memory freed by checkpointing (GB)
                "savings_fraction"   — fraction of baseline memory saved
                "recompute_overhead" — extra recomputation cost in [0, 1]
        """
        n_layers = self._n_layers()
        d_model = self._d_model()

        baseline_bytes = estimate_activation_bytes(
            n_layers, seq_len, d_model, batch_size, self.cfg.bytes_per_element
        )
        baseline_gb = baseline_bytes / (1024**3)

        indices = self.select_layers(seq_len, batch_size)
        n_checkpointed = len(indices)

        # Each checkpointed layer frees its activations
        per_layer_bytes = baseline_bytes / n_layers if n_layers > 0 else 0
        saved_bytes = n_checkpointed * per_layer_bytes
        saved_gb = saved_bytes / (1024**3)

        savings_fraction = saved_bytes / baseline_bytes if baseline_bytes > 0 else 0.0
        recompute_overhead = compute_recomputation_cost(n_checkpointed, n_layers)

        return {
            "baseline_gb": baseline_gb,
            "saved_gb": saved_gb,
            "savings_fraction": savings_fraction,
            "recompute_overhead": recompute_overhead,
        }
