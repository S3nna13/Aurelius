"""Flexible activation rematerialization: selective checkpointing and memory/compute trade-off analysis."""  # noqa: E501

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor


@dataclass
class RematerializationConfig:
    """Configuration for selective activation rematerialization.

    Controls which transformer layers are gradient-checkpointed (recomputed
    during backward pass) to trade compute for memory.
    """

    checkpoint_every_n: int = 1
    """Checkpoint every N layers (1 = all layers, 2 = every other, etc.)."""

    checkpoint_layers: list[int] | None = None
    """Explicit layer indices to checkpoint. Overrides checkpoint_every_n when set."""

    use_reentrant: bool = False
    """Passed to torch.utils.checkpoint.checkpoint. False recommended for most cases."""

    offload_to_cpu: bool = False
    """Store activations on CPU between forward and backward. Not implemented here, just stored."""


def should_checkpoint_layer(layer_idx: int, config: RematerializationConfig) -> bool:
    """Determine whether a layer should use gradient checkpointing.

    Args:
        layer_idx: Zero-based index of the layer.
        config: Rematerialization configuration.

    Returns:
        True if the layer should be checkpointed.
    """
    if config.checkpoint_layers is not None:
        return layer_idx in config.checkpoint_layers
    return layer_idx % config.checkpoint_every_n == 0


def checkpoint_forward(module: nn.Module, *args: Any, use_reentrant: bool = False) -> Any:
    """Run a module's forward pass under torch gradient checkpointing.

    Recomputes activations during the backward pass instead of storing them,
    reducing peak activation memory at the cost of extra FLOPs.

    Args:
        module: The nn.Module to execute.
        *args: Positional arguments forwarded to module.
        use_reentrant: Passed to torch.utils.checkpoint.checkpoint.

    Returns:
        Module output (same as module(*args)).
    """
    return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=use_reentrant)


class SelectiveCheckpointWrapper(nn.Module):
    """Wraps an nn.ModuleList and applies gradient checkpointing selectively.

    Each layer is called as ``layer(x)`` and must return a Tensor.
    Whether a given layer is checkpointed is controlled by the
    :class:`RematerializationConfig`.

    Args:
        layers: ModuleList of transformer (or other) layers.
        config: Rematerialization configuration.
    """

    def __init__(self, layers: nn.ModuleList, config: RematerializationConfig) -> None:
        super().__init__()
        self.layers = layers
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        """Run all layers sequentially, checkpointing the selected ones.

        Args:
            x: Input hidden states tensor.

        Returns:
            Output hidden states tensor after all layers.
        """
        for i, layer in enumerate(self.layers):
            if should_checkpoint_layer(i, self.config):
                x = checkpoint_forward(layer, x, use_reentrant=self.config.use_reentrant)
            else:
                x = layer(x)
        return x


def estimate_activation_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate activation memory usage with and without gradient checkpointing.

    The "full" estimate assumes all layer activations are stored simultaneously
    (standard backprop). The "checkpointed" estimate assumes only a single layer's
    boundary activations are retained at any time (full checkpointing).

    Args:
        batch_size: Number of sequences in a batch.
        seq_len: Sequence length (tokens).
        d_model: Model hidden dimension.
        n_layers: Number of transformer layers.
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32).

    Returns:
        Dictionary with keys:
            - ``"full_mb"``: float — full activation memory in megabytes.
            - ``"checkpointed_mb"``: float — checkpointed activation memory in MB.
            - ``"savings_factor"``: float — ratio full / checkpointed (= n_layers).
    """
    bytes_per_mb = 1024.0 * 1024.0
    element_count = batch_size * seq_len * d_model

    full_bytes = element_count * n_layers * dtype_bytes
    checkpointed_bytes = element_count * dtype_bytes  # one layer worth of boundary activations

    full_mb = full_bytes / bytes_per_mb
    checkpointed_mb = checkpointed_bytes / bytes_per_mb
    savings_factor = full_mb / checkpointed_mb  # == n_layers

    return {
        "full_mb": full_mb,
        "checkpointed_mb": checkpointed_mb,
        "savings_factor": savings_factor,
    }


def wrap_model_layers(
    model_layers: nn.ModuleList,
    config: RematerializationConfig,
) -> SelectiveCheckpointWrapper:
    """Convenience factory: wrap a ModuleList with selective checkpointing.

    Args:
        model_layers: The model's layer stack as an nn.ModuleList.
        config: Rematerialization configuration.

    Returns:
        A :class:`SelectiveCheckpointWrapper` ready for use.
    """
    return SelectiveCheckpointWrapper(model_layers, config)


class MemoryProfiler:
    """Profiles peak CUDA memory consumption and wall-clock time of a callable.

    On CPU-only machines (or when CUDA is unavailable) the memory delta is
    reported as 0.0 MB.
    """

    def __init__(self) -> None:
        pass

    def measure(self, fn: Callable, *args: Any) -> tuple[Any, dict[str, float]]:
        """Call *fn* with *args* and record peak memory delta and elapsed time.

        Args:
            fn: Callable to profile.
            *args: Positional arguments forwarded to *fn*.

        Returns:
            A 2-tuple ``(result, stats)`` where *result* is the return value of
            ``fn(*args)`` and *stats* is a dict with:
                - ``"peak_memory_delta_mb"``: float — increase in peak CUDA
                  memory allocated during the call, in megabytes.
                - ``"elapsed_ms"``: float — wall-clock duration in milliseconds.
        """
        has_cuda = torch.cuda.is_available()

        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.max_memory_allocated()
        else:
            before = 0

        t0 = time.perf_counter()
        result = fn(*args)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if has_cuda:
            after = torch.cuda.max_memory_allocated()
            delta_bytes = max(0, after - before)
        else:
            delta_bytes = 0

        stats: dict[str, float] = {
            "peak_memory_delta_mb": delta_bytes / (1024.0 * 1024.0),
            "elapsed_ms": elapsed_ms,
        }
        return result, stats
