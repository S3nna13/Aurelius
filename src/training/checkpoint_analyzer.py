"""Checkpoint Analyzer — activation memory profiling and gradient checkpointing strategy recommendation."""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.model.transformer import TransformerBlock


@dataclass
class LayerMemoryStats:
    layer_idx: int
    layer_name: str
    activation_bytes: int  # peak activation memory for this layer (hidden state only)
    param_bytes: int  # parameter memory for this layer
    grad_bytes: int  # gradient memory (= param_bytes for fp32)
    should_checkpoint: bool  # recommendation


@dataclass
class CheckpointAnalysis:
    total_activation_bytes: int
    total_param_bytes: int
    peak_memory_no_checkpoint: int  # activation + param + grad bytes
    peak_memory_with_checkpoint: int  # estimated with recommended strategy
    memory_savings_bytes: int
    memory_savings_pct: float  # 0.0 to 1.0
    layers: list[LayerMemoryStats]
    recommended_layers: list[int]  # layer indices to checkpoint
    strategy: str  # "none" | "uniform" | "selective"


def profile_layer_activations(
    model: nn.Module,
    input_ids: Tensor,
    dtype: torch.dtype = torch.float32,
) -> list[LayerMemoryStats]:
    """Use forward hooks to measure activation tensor sizes per TransformerBlock.

    The hook intercepts the output of each TransformerBlock. The output is a
    (hidden_state, kv_cache) tuple; only hidden_state bytes are counted.

    Returns a list sorted by layer_idx.
    """
    # Collect all TransformerBlock instances with their names
    transformer_blocks: list[tuple[str, nn.Module]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TransformerBlock)
    ]

    # Storage for measured activation sizes keyed by module id
    activation_sizes: dict[int, int] = {}

    def make_hook(module_id: int):
        def hook(module, inputs, output):
            # output is (hidden_state, kv) — measure hidden_state only
            if isinstance(output, (tuple, list)):
                hidden = output[0]
            else:
                hidden = output

            if isinstance(hidden, Tensor):
                n_elements = hidden.numel()
                bytes_per_element = hidden.element_size()
                activation_sizes[module_id] = n_elements * bytes_per_element
            else:
                activation_sizes[module_id] = 0

        return hook

    # Register hooks
    handles = []
    for name, module in transformer_blocks:
        handle = module.register_forward_hook(make_hook(id(module)))
        handles.append(handle)

    # Run a short forward pass with no_grad for profiling
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Build LayerMemoryStats for each block
    result: list[LayerMemoryStats] = []
    for idx, (name, module) in enumerate(transformer_blocks):
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        activation_bytes = activation_sizes.get(id(module), 0)

        result.append(
            LayerMemoryStats(
                layer_idx=idx,
                layer_name=name,
                activation_bytes=activation_bytes,
                param_bytes=param_bytes,
                grad_bytes=param_bytes,  # fp32: grad same size as param
                should_checkpoint=False,  # filled in by analyze_checkpoint_strategy
            )
        )

    # Sort by layer_idx (already in order, but be explicit)
    result.sort(key=lambda s: s.layer_idx)
    return result


def estimate_memory_with_checkpointing(
    layer_stats: list[LayerMemoryStats],
    checkpoint_layers: list[int],
) -> int:
    """Estimate peak memory bytes given a set of checkpointed layer indices.

    With checkpointing, a layer's activations are recomputed during backward,
    so checkpointed layers don't need to store activations — only param + grad.
    Non-checkpointed layers store activations + param + grad.

    Returns the estimated peak memory in bytes.
    """
    checkpoint_set = set(checkpoint_layers)
    total = 0
    for stats in layer_stats:
        if stats.layer_idx in checkpoint_set:
            # No stored activations; still need param + grad
            total += stats.param_bytes + stats.grad_bytes
        else:
            total += stats.activation_bytes + stats.param_bytes + stats.grad_bytes
    return total


def analyze_checkpoint_strategy(
    model: nn.Module,
    input_ids: Tensor,
    memory_budget_gb: float = 16.0,
    dtype: torch.dtype = torch.float32,
) -> CheckpointAnalysis:
    """Profile layer activations and recommend an optimal checkpointing strategy.

    Steps:
    1. Profile layer activations via forward hooks.
    2. Compute total param/grad bytes from model.parameters().
    3. Determine if memory budget is exceeded without checkpointing.
    4. Greedy selection: checkpoint layers with largest activations first until
       within budget (or all layers are checkpointed).
    5. Return full CheckpointAnalysis.
    """
    # 1. Profile activations
    layer_stats = profile_layer_activations(model, input_ids, dtype=dtype)

    # 2. Compute total param bytes (model-wide, not per-layer)
    total_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_grad_bytes = total_param_bytes  # fp32: same as param bytes

    # Total activation bytes
    total_activation_bytes = sum(s.activation_bytes for s in layer_stats)

    # 3. Peak memory without any checkpointing
    peak_no_checkpoint = total_activation_bytes + total_param_bytes + total_grad_bytes

    # Budget in bytes
    budget_bytes = int(memory_budget_gb * 1024**3)

    # 4. Greedy selection — checkpoint layers with largest activations first
    recommended_layers: list[int] = []

    if peak_no_checkpoint > budget_bytes:
        # Sort layers by activation_bytes descending for greedy selection
        sorted_by_activation = sorted(layer_stats, key=lambda s: s.activation_bytes, reverse=True)

        current_checkpoint = []
        for stats in sorted_by_activation:
            current_checkpoint.append(stats.layer_idx)
            estimated = estimate_memory_with_checkpointing(layer_stats, current_checkpoint)
            # Include non-layer model params too
            # estimated already covers per-layer params; add non-layer overhead
            if estimated <= budget_bytes:
                break

        recommended_layers = sorted(current_checkpoint)

    # 5. Estimate memory with recommended strategy
    peak_with_checkpoint = estimate_memory_with_checkpointing(layer_stats, recommended_layers)
    # Add back non-layer model params (params outside TransformerBlocks)
    # (already included since we use total_param_bytes for model-level accounting
    #  but estimate_memory_with_checkpointing uses per-layer params only)
    # Recompute properly: peak_with_checkpoint already sums per-layer param+grad;
    # we need to add the remaining model params (embed, norm, lm_head, etc.)
    layer_param_bytes = sum(s.param_bytes for s in layer_stats)
    non_layer_param_bytes = max(0, total_param_bytes - layer_param_bytes)
    non_layer_grad_bytes = non_layer_param_bytes
    peak_with_checkpoint += non_layer_param_bytes + non_layer_grad_bytes

    # Also add activation memory from non-TransformerBlock parts (approximation: 0)
    # (We only track TransformerBlock activations)

    # Recompute peak_no_checkpoint to be consistent (layer + non-layer)
    peak_no_checkpoint_consistent = (
        sum(s.activation_bytes + s.param_bytes + s.grad_bytes for s in layer_stats)
        + non_layer_param_bytes
        + non_layer_grad_bytes
    )

    memory_savings_bytes = max(0, peak_no_checkpoint_consistent - peak_with_checkpoint)
    memory_savings_pct = (
        memory_savings_bytes / peak_no_checkpoint_consistent
        if peak_no_checkpoint_consistent > 0
        else 0.0
    )

    # Mark should_checkpoint on each layer stat
    recommended_set = set(recommended_layers)
    for s in layer_stats:
        s.should_checkpoint = s.layer_idx in recommended_set

    # Determine strategy label
    n_layers = len(layer_stats)
    n_recommended = len(recommended_layers)
    if n_recommended == 0:
        strategy = "none"
    elif n_recommended == n_layers:
        strategy = "uniform"
    else:
        strategy = "selective"

    return CheckpointAnalysis(
        total_activation_bytes=total_activation_bytes,
        total_param_bytes=total_param_bytes,
        peak_memory_no_checkpoint=peak_no_checkpoint_consistent,
        peak_memory_with_checkpoint=peak_with_checkpoint,
        memory_savings_bytes=memory_savings_bytes,
        memory_savings_pct=memory_savings_pct,
        layers=layer_stats,
        recommended_layers=recommended_layers,
        strategy=strategy,
    )
