"""Gradient checkpointing utilities for Aurelius transformer training.

Trades compute for memory by recomputing activations during the backward pass
instead of storing them. Wraps layers with torch.utils.checkpoint to reduce
peak activation memory.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint


@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing."""

    checkpoint_every: int = 2
    enabled: bool = True
    # Legacy alias kept for backward compat
    checkpoint_every_n_layers: int = 1
    use_reentrant: bool = False


class CheckpointedModule(nn.Module):
    """Wraps any nn.Module with gradient checkpointing."""

    def __init__(self, module: nn.Module, config: CheckpointConfig) -> None:
        super().__init__()
        self.module = module
        self.config = config

    def forward(self, *args, **kwargs):
        if self.config.enabled:
            return torch.utils.checkpoint.checkpoint(
                self.module, *args, use_reentrant=False, **kwargs
            )
        return self.module(*args, **kwargs)


def apply_checkpointing(
    model: nn.Module,
    config: CheckpointConfig,
    layer_names: list[str] | None = None,
) -> nn.Module:
    """Wrap every nn.ModuleList layer (or layers matching layer_names) in CheckpointedModule."""
    for name, child in list(model.named_children()):
        if layer_names is not None:
            if name in layer_names:
                setattr(model, name, CheckpointedModule(child, config))
            else:
                apply_checkpointing(child, config, layer_names)
        else:
            if isinstance(child, nn.ModuleList):
                wrapped = nn.ModuleList([CheckpointedModule(layer, config) for layer in child])
                setattr(model, name, wrapped)
            else:
                apply_checkpointing(child, config, layer_names)
    return model


def checkpoint_forward(fn: Callable, *args, enabled: bool = True) -> Any:
    """Wrapper around torch.utils.checkpoint.checkpoint.

    If enabled, calls fn(*args) through gradient checkpointing so activations
    are recomputed during the backward pass rather than stored. If not enabled,
    calls fn(*args) directly.

    Args:
        fn: Callable to (optionally) checkpoint.
        *args: Positional arguments forwarded to fn.
        enabled: Whether to apply gradient checkpointing.

    Returns:
        Output of fn(*args).
    """
    if not enabled:
        return fn(*args)
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)


class CheckpointedLayer(nn.Module):
    """Thin wrapper that applies gradient checkpointing to any nn.Module layer.

    Args:
        layer: The module to wrap.
        config: CheckpointConfig controlling checkpointing behaviour.
    """

    def __init__(self, layer: nn.Module, config: CheckpointConfig) -> None:
        super().__init__()
        self.layer = layer
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing.

        Args:
            x: Input tensor.

        Returns:
            Output tensor from the wrapped layer.
        """
        return checkpoint_forward(self.layer, x, enabled=self.config.enabled)


class CheckpointedSequential(nn.Module):
    """Sequential container that applies gradient checkpointing to every Nth layer.

    Stores modules as an nn.ModuleList and applies them in order during forward.
    Layer i is checkpointed if (i % checkpoint_every_n_layers == 0) and
    config.enabled is True.

    Args:
        layers: List of nn.Module instances to apply in sequence.
        config: CheckpointConfig controlling checkpointing behaviour.
    """

    def __init__(self, layers: list[nn.Module], config: CheckpointConfig) -> None:
        super().__init__()
        self.module_list = nn.ModuleList(layers)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layers in order, checkpointing every Nth layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying all layers.
        """
        n = self.config.checkpoint_every_n_layers
        for i, module in enumerate(self.module_list):
            should_checkpoint = self.config.enabled and (i % n == 0)
            x = checkpoint_forward(module, x, enabled=should_checkpoint)
        return x


def estimate_activation_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    dtype_bytes: int = 4,
) -> int:
    """Estimate activation memory in bytes WITHOUT gradient checkpointing.

    Formula: batch_size * seq_len * d_model * n_layers * dtype_bytes

    Args:
        batch_size: Number of samples per batch.
        seq_len: Sequence length.
        d_model: Hidden dimension size.
        n_layers: Number of transformer layers.
        dtype_bytes: Bytes per element (default 4 for float32).

    Returns:
        Estimated activation memory in bytes.
    """
    return batch_size * seq_len * d_model * n_layers * dtype_bytes


def estimate_checkpointed_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    checkpoint_every: int = 1,
    dtype_bytes: int = 4,
) -> int:
    """Estimate activation memory in bytes WITH gradient checkpointing.

    Only stores activations at checkpoint boundaries; approximately
    batch_size * seq_len * d_model * ceil(n_layers / checkpoint_every) * dtype_bytes.

    Args:
        batch_size: Number of samples per batch.
        seq_len: Sequence length.
        d_model: Hidden dimension size.
        n_layers: Number of transformer layers.
        checkpoint_every: Checkpoint every Nth layer.
        dtype_bytes: Bytes per element (default 4 for float32).

    Returns:
        Estimated activation memory in bytes with checkpointing.
    """
    n_checkpoints = math.ceil(n_layers / checkpoint_every)
    return batch_size * seq_len * d_model * n_checkpoints * dtype_bytes


def wrap_model_with_checkpointing(
    model: nn.Sequential, config: CheckpointConfig
) -> CheckpointedSequential:
    """Convenience wrapper: convert an nn.Sequential into a CheckpointedSequential.

    Args:
        model: An nn.Sequential whose children will be wrapped.
        config: CheckpointConfig controlling checkpointing behaviour.

    Returns:
        A CheckpointedSequential containing the same layers.
    """
    layers = list(model.children())
    return CheckpointedSequential(layers, config)
