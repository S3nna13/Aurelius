"""Activation (gradient) checkpointing utilities for Aurelius transformer training.

Wraps transformer layers with torch.utils.checkpoint to trade recomputation
for reduced peak activation memory during backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass
class CheckpointConfig:
    """Configuration for activation checkpointing.

    Attributes:
        checkpoint_every_n_layers: Wrap every Nth layer. 1 = all layers, 2 = every other, etc.
        use_reentrant: Passed to torch.utils.checkpoint.checkpoint. False is recommended
            for newer PyTorch versions and compatibility with torch.compile.
        offload_to_cpu: Whether to offload saved activations to CPU memory. Not yet
            implemented in torch.utils.checkpoint natively; reserved for future use.
    """

    checkpoint_every_n_layers: int = 1
    use_reentrant: bool = False
    offload_to_cpu: bool = False


class CheckpointedLayer(nn.Module):
    """Thin wrapper that applies gradient checkpointing to any nn.Module layer.

    The wrapped layer's forward is called via torch.utils.checkpoint.checkpoint,
    which discards intermediate activations during the forward pass and recomputes
    them during backward, reducing peak memory at the cost of extra compute.

    Usage:
        layer = TransformerBlock(config)
        ckpt_layer = CheckpointedLayer(layer, use_reentrant=False)
        output = ckpt_layer(*inputs)
    """

    def __init__(self, layer: nn.Module, use_reentrant: bool = False) -> None:
        super().__init__()
        self.layer = layer
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        # torch.utils.checkpoint.checkpoint requires all inputs to be tensors
        # (or at least that's how use_reentrant=False works). We wrap the call
        # in a closure that captures keyword arguments.
        if kwargs:
            def _fn(*a):
                return self.layer(*a, **kwargs)
        else:
            _fn = self.layer

        return checkpoint(_fn, *args, use_reentrant=self.use_reentrant)


def wrap_layers_with_checkpointing(model: nn.Module, config: CheckpointConfig) -> int:
    """Iterate over model.layers and wrap every Nth layer with CheckpointedLayer.

    Args:
        model: A model that exposes a ``layers`` attribute (nn.ModuleList).
        config: CheckpointConfig controlling which layers to wrap.

    Returns:
        Number of layers that were wrapped.
    """
    if not hasattr(model, "layers"):
        raise AttributeError("model must have a 'layers' attribute (nn.ModuleList)")

    n = config.checkpoint_every_n_layers
    wrapped_count = 0

    for i in range(len(model.layers)):
        # Wrap layer at index i if (i+1) is divisible by n, i.e. layers 0, n-1, 2n-1 …
        # Simpler: wrap layer i when (i % n == 0)
        if i % n == 0:
            model.layers[i] = CheckpointedLayer(
                model.layers[i], use_reentrant=config.use_reentrant
            )
            wrapped_count += 1

    return wrapped_count


def estimate_memory_savings(
    n_layers: int,
    d_model: int,
    seq_len: int,
    batch_size: int,
) -> tuple[float, float]:
    """Estimate activation memory with and without checkpointing.

    The formula is a simplified estimate of the float32 activation memory
    proportional to layer count:

        full        = n_layers * seq_len * batch_size * d_model * 4 bytes / 1e6  (MB)
        checkpointed = full / n_layers  (only ~1 layer's activations kept at a time)

    Args:
        n_layers: Number of transformer layers.
        d_model: Hidden dimension.
        seq_len: Sequence length.
        batch_size: Batch size.

    Returns:
        Tuple of (full_mb, checkpointed_mb) — estimated activation memory in MB.
    """
    full_mb: float = n_layers * seq_len * batch_size * d_model * 4 / 1e6
    checkpointed_mb: float = full_mb / n_layers
    return full_mb, checkpointed_mb


class ActivationCheckpointTrainer:
    """Minimal trainer that wraps a model with activation checkpointing and runs train steps.

    Args:
        model: An AureliusTransformer (or any model following the ``loss, logits, pkv = model(input_ids)``
            API that exposes a ``layers`` nn.ModuleList).
        optimizer: A PyTorch optimizer.
        config: CheckpointConfig for layer wrapping.
    """

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
        """Run a single forward + backward pass.

        The model API is: ``loss, logits, pkv = model(input_ids)``
        Labels are derived by shifting input_ids right by one position
        (standard next-token prediction).

        Args:
            input_ids: (batch, seq_len) integer token ids.

        Returns:
            dict with keys:
                ``loss`` (float): scalar loss value.
                ``n_checkpointed_layers`` (int): number of checkpointed layers.
        """
        self.optimizer.zero_grad()

        # Use input_ids as both input and labels (standard LM shift handled inside model)
        loss, _logits, _pkv = self.model(input_ids, labels=input_ids)

        if loss is None:
            raise RuntimeError("Model returned None loss — ensure labels are passed or seq_len > 1")

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_checkpointed_layers": self.n_checkpointed_layers,
        }


def get_checkpoint_stats(model: nn.Module) -> dict:
    """Return checkpointing statistics for a model.

    Args:
        model: A model with a ``layers`` nn.ModuleList, some of which may be
            wrapped in CheckpointedLayer.

    Returns:
        dict with keys:
            ``total_layers`` (int): total number of layers.
            ``checkpointed_layers`` (int): number of CheckpointedLayer wrappers.
            ``checkpoint_ratio`` (float): fraction of layers that are checkpointed.
    """
    if not hasattr(model, "layers"):
        raise AttributeError("model must have a 'layers' attribute (nn.ModuleList)")

    total = len(model.layers)
    checkpointed = sum(
        1 for layer in model.layers if isinstance(layer, CheckpointedLayer)
    )
    ratio = checkpointed / total if total > 0 else 0.0

    return {
        "total_layers": total,
        "checkpointed_layers": checkpointed,
        "checkpoint_ratio": ratio,
    }
