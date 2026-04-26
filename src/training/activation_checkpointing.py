"""Activation (gradient) checkpointing utilities for Aurelius transformer training.

Wraps transformer layers with torch.utils.checkpoint to trade recomputation
for reduced peak activation memory during backpropagation.
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
    """Configuration for activation checkpointing.

    Attributes:
        checkpoint_every_n_layers: Wrap every Nth layer. 1 = all layers, 2 = every other, etc.
        offload_to_cpu: Whether to offload saved activations to CPU memory.
        use_reentrant: Passed to torch.utils.checkpoint.checkpoint. False is recommended
            for newer PyTorch versions and compatibility with torch.compile.
    """

    checkpoint_every_n_layers: int = 1
    offload_to_cpu: bool = False
    use_reentrant: bool = False


def checkpoint_forward(fn: Callable, *args, use_reentrant: bool = False) -> Any:
    """Thin wrapper around torch.utils.checkpoint.checkpoint.

    Calls fn(*args) with gradient checkpointing enabled.

    Args:
        fn: Callable to checkpoint.
        *args: Positional arguments forwarded to fn.
        use_reentrant: Passed to torch.utils.checkpoint.checkpoint.

    Returns:
        Output of fn(*args).
    """
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=use_reentrant)


class CheckpointedSequential(nn.Module):
    """Sequential container that applies gradient checkpointing to every Nth module.

    Stores modules as an nn.ModuleList and applies them in order during forward.
    Module i is checkpointed if i % checkpoint_every_n_layers == 0.

    Args:
        modules: List of nn.Module instances to apply in sequence.
        config: CheckpointConfig controlling checkpointing behavior.
    """

    def __init__(self, modules: list[nn.Module], config: CheckpointConfig) -> None:
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply modules in order, checkpointing every Nth module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying all modules.
        """
        n = self.config.checkpoint_every_n_layers
        for i, module in enumerate(self.module_list):
            if i % n == 0:
                x = torch.utils.checkpoint.checkpoint(
                    module, x, use_reentrant=self.config.use_reentrant
                )
            else:
                x = module(x)
        return x


def estimate_memory_savings(
    n_layers: int,
    d_model: int,
    seq_len: int,
    batch_size: int,
    checkpoint_every_n: int = 1,
) -> dict:
    """Estimate activation memory savings from gradient checkpointing.

    Args:
        n_layers: Number of transformer layers.
        d_model: Hidden dimension.
        seq_len: Sequence length.
        batch_size: Batch size.
        checkpoint_every_n: Checkpoint every Nth layer.

    Returns:
        Dict with keys:
            'activation_bytes_no_checkpoint': int — full float32 activation bytes.
            'activation_bytes_with_checkpoint': int — checkpointed activation bytes.
            'savings_fraction': float — fraction of memory saved.
    """
    bytes_per_element = 4  # float32
    no_checkpoint = bytes_per_element * n_layers * seq_len * batch_size * d_model

    n_checkpointed = math.ceil(n_layers / checkpoint_every_n)
    with_checkpoint = bytes_per_element * n_checkpointed * seq_len * batch_size * d_model

    savings_fraction = 1.0 - (with_checkpoint / no_checkpoint)

    return {
        "activation_bytes_no_checkpoint": int(no_checkpoint),
        "activation_bytes_with_checkpoint": int(with_checkpoint),
        "savings_fraction": float(savings_fraction),
    }


def apply_activation_checkpointing(
    model: nn.Module, module_class: type, config: CheckpointConfig
) -> int:
    """Walk model.modules() and wrap all instances of module_class with checkpointing.

    Replaces each matching module's forward method so that calls go through
    torch.utils.checkpoint.checkpoint.

    Args:
        model: The model to modify in-place.
        module_class: The class of modules to wrap.
        config: CheckpointConfig (use_reentrant is used).

    Returns:
        Count of modules that were wrapped.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, module_class):
            original_forward = module.forward
            module.forward = lambda *args, _orig=original_forward, **kwargs: (
                torch.utils.checkpoint.checkpoint(
                    _orig, *args, use_reentrant=config.use_reentrant, **kwargs
                )
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility
# ---------------------------------------------------------------------------


class CheckpointedLayer(nn.Module):
    """Thin wrapper that applies gradient checkpointing to any nn.Module layer."""

    def __init__(self, layer: nn.Module, use_reentrant: bool = False) -> None:
        super().__init__()
        self.layer = layer
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        if kwargs:

            def _fn(*a):
                return self.layer(*a, **kwargs)
        else:
            _fn = self.layer
        return torch.utils.checkpoint.checkpoint(_fn, *args, use_reentrant=self.use_reentrant)


def wrap_layers_with_checkpointing(model: nn.Module, config: CheckpointConfig) -> int:
    """Iterate over model.layers and wrap every Nth layer with CheckpointedLayer."""
    if not hasattr(model, "layers"):
        raise AttributeError("model must have a 'layers' attribute (nn.ModuleList)")

    n = config.checkpoint_every_n_layers
    wrapped_count = 0

    for i in range(len(model.layers)):
        if i % n == 0:
            model.layers[i] = CheckpointedLayer(model.layers[i], use_reentrant=config.use_reentrant)
            wrapped_count += 1

    return wrapped_count


def get_checkpoint_stats(model: nn.Module) -> dict:
    """Return checkpointing statistics for a model."""
    if not hasattr(model, "layers"):
        raise AttributeError("model must have a 'layers' attribute (nn.ModuleList)")

    total = len(model.layers)
    checkpointed = sum(1 for layer in model.layers if isinstance(layer, CheckpointedLayer))
    ratio = checkpointed / total if total > 0 else 0.0

    return {
        "total_layers": total,
        "checkpointed_layers": checkpointed,
        "checkpoint_ratio": ratio,
    }


class ActivationCheckpointTrainer:
    """Minimal trainer that wraps a model with activation checkpointing and runs train steps."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: CheckpointConfig | None = None,
    ) -> None:
        self.config = config or CheckpointConfig()
        self.model = model
        self.optimizer = optimizer
        self.n_checkpointed_layers = wrap_layers_with_checkpointing(self.model, self.config)
        self.model.train()

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Run a single forward + backward pass."""
        self.optimizer.zero_grad()
        loss, _logits, _pkv = self.model(input_ids, labels=input_ids)

        if loss is None:
            raise RuntimeError("Model returned None loss — ensure labels are passed or seq_len > 1")

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_checkpointed_layers": self.n_checkpointed_layers,
        }
