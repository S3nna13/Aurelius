"""Activation checkpointing policy planner.

Given a memory budget, computes which layers to checkpoint to minimize
memory usage while minimizing recomputation overhead.

Strategies:
- NONE: No checkpointing (highest memory, fastest)
- ALL: Checkpoint all layers (lowest memory, 33% slower)
- UNIFORM: Every Nth layer (balance memory vs speed)
- MEMORY_OPTIMAL: Greedy algorithm — checkpoint largest activations first
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointStrategy(Enum):
    NONE = "none"
    ALL = "all"
    UNIFORM = "uniform"           # every N layers
    MEMORY_OPTIMAL = "memory_optimal"  # greedy by activation size


@dataclass
class CheckpointPolicyConfig:
    strategy: CheckpointStrategy = CheckpointStrategy.UNIFORM
    memory_budget_gb: float = 16.0     # target peak memory in GB
    uniform_interval: int = 2          # for UNIFORM: checkpoint every N layers
    recompute_factor: float = 0.33     # FLOPs overhead per checkpointed layer


@dataclass
class LayerMemoryEstimate:
    """Memory estimate for a single transformer layer."""
    layer_idx: int
    activation_bytes: int    # bytes to store forward activations
    param_bytes: int         # bytes for parameters (always stored)
    checkpointed: bool = False

    @property
    def memory_if_stored(self) -> int:
        return self.activation_bytes + self.param_bytes

    @property
    def memory_if_checkpointed(self) -> int:
        return self.param_bytes  # activations freed, only params kept


@dataclass
class CheckpointPlan:
    """Computed checkpointing plan for a model."""
    strategy: CheckpointStrategy
    layer_estimates: list[LayerMemoryEstimate]
    checkpointed_layers: list[int]      # indices of checkpointed layers
    estimated_peak_memory_bytes: int
    estimated_recompute_overhead: float  # fraction of extra FLOPs

    @property
    def n_checkpointed(self) -> int:
        return len(self.checkpointed_layers)

    @property
    def memory_saved_bytes(self) -> int:
        return sum(
            e.activation_bytes for e in self.layer_estimates
            if e.checkpointed
        )


def estimate_layer_memory(
    layer: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype_bytes: int = 2,
) -> LayerMemoryEstimate:
    """Estimate memory for a transformer layer's activations.

    Activations stored during forward (for backward):
    - Input: (B, S, D)
    - After attention: (B, S, D)
    - After FFN: (B, S, D)
    - QKV projections: 3 * (B, S, D)
    Total: ~6 * B * S * D * dtype_bytes

    Params: sum of param.numel() * dtype_bytes
    """
    activation_bytes = 6 * batch_size * seq_len * d_model * dtype_bytes
    param_bytes = sum(p.numel() * dtype_bytes for p in layer.parameters())
    return LayerMemoryEstimate(
        layer_idx=0,  # caller sets this
        activation_bytes=activation_bytes,
        param_bytes=param_bytes,
    )


def _collect_layer_estimates(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype_bytes: int,
) -> list[LayerMemoryEstimate]:
    """Collect LayerMemoryEstimate for each layer in model.layers."""
    estimates: list[LayerMemoryEstimate] = []
    layers = list(model.layers)
    for i, layer in enumerate(layers):
        est = estimate_layer_memory(layer, batch_size, seq_len, d_model, dtype_bytes)
        est.layer_idx = i
        estimates.append(est)
    return estimates


def _estimate_peak_memory(
    estimates: list[LayerMemoryEstimate],
    checkpointed_set: set[int],
    recompute_factor: float,
) -> int:
    """Estimate peak memory bytes given which layers are checkpointed.

    Peak memory = sum of param_bytes for all layers
                + sum of activation_bytes for non-checkpointed layers
    (checkpointed layers free activations after each layer's forward)
    """
    total = 0
    for est in estimates:
        total += est.param_bytes
        if est.layer_idx not in checkpointed_set:
            total += est.activation_bytes
    return total


def compute_checkpoint_plan(
    model: nn.Module,
    cfg: CheckpointPolicyConfig,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype_bytes: int = 2,
) -> CheckpointPlan:
    """Compute which layers to checkpoint given the strategy and memory budget.

    For MEMORY_OPTIMAL: greedy — sort layers by activation size descending,
    checkpoint until estimated peak memory <= memory_budget_gb.

    Returns CheckpointPlan.
    """
    estimates = _collect_layer_estimates(model, batch_size, seq_len, d_model, dtype_bytes)
    n_layers = len(estimates)

    if cfg.strategy == CheckpointStrategy.NONE:
        checkpointed_layers: list[int] = []

    elif cfg.strategy == CheckpointStrategy.ALL:
        checkpointed_layers = list(range(n_layers))

    elif cfg.strategy == CheckpointStrategy.UNIFORM:
        interval = max(1, cfg.uniform_interval)
        checkpointed_layers = [i for i in range(n_layers) if i % interval == 0]

    elif cfg.strategy == CheckpointStrategy.MEMORY_OPTIMAL:
        budget_bytes = int(cfg.memory_budget_gb * (1024 ** 3))
        # Sort layers by activation size descending (greedy: checkpoint biggest first)
        sorted_by_size = sorted(estimates, key=lambda e: e.activation_bytes, reverse=True)
        checkpointed_set: set[int] = set()

        # Check if we're already within budget without any checkpointing
        peak = _estimate_peak_memory(estimates, checkpointed_set, cfg.recompute_factor)
        for est in sorted_by_size:
            if peak <= budget_bytes:
                break
            checkpointed_set.add(est.layer_idx)
            peak = _estimate_peak_memory(estimates, checkpointed_set, cfg.recompute_factor)

        checkpointed_layers = sorted(checkpointed_set)
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")

    # Mark estimates
    checkpointed_set_final = set(checkpointed_layers)
    for est in estimates:
        est.checkpointed = est.layer_idx in checkpointed_set_final

    peak_memory = _estimate_peak_memory(estimates, checkpointed_set_final, cfg.recompute_factor)
    recompute_overhead = cfg.recompute_factor * len(checkpointed_layers) / max(n_layers, 1)

    return CheckpointPlan(
        strategy=cfg.strategy,
        layer_estimates=estimates,
        checkpointed_layers=checkpointed_layers,
        estimated_peak_memory_bytes=peak_memory,
        estimated_recompute_overhead=recompute_overhead,
    )


def apply_checkpoint_plan(
    model: nn.Module,
    plan: CheckpointPlan,
) -> None:
    """Apply the checkpoint plan to the model in-place.

    For each layer in plan.checkpointed_layers:
    Wrap model.layers[i].forward with torch.utils.checkpoint.checkpoint.

    For non-checkpointed layers: ensure standard (non-checkpointed) forward.
    """
    checkpointed_set = set(plan.checkpointed_layers)

    for i, layer in enumerate(model.layers):
        if i in checkpointed_set:
            # Save original forward if not already wrapped
            if not hasattr(layer, '_original_forward'):
                layer._original_forward = layer.forward

            original_forward = layer._original_forward

            def make_checkpointed_forward(orig_fwd):
                def checkpointed_forward(*args, **kwargs):
                    # torch.utils.checkpoint.checkpoint requires a function + tensor args
                    # Flatten args and pass through checkpoint
                    def fn(*a):
                        return orig_fwd(*a, **kwargs)
                    return checkpoint(fn, *args, use_reentrant=False)
                return checkpointed_forward

            layer.forward = make_checkpointed_forward(original_forward)
        else:
            # Restore original forward if it was previously wrapped
            if hasattr(layer, '_original_forward'):
                layer.forward = layer._original_forward
