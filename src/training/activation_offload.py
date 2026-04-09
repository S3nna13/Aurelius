"""Activation offloading and memory-efficient training: CPU offload, selective checkpointing, and peak memory tracking."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OffloadConfig:
    """Configuration for activation offloading and memory-efficient training."""

    offload_to_cpu: bool = True
    """Offload activations to CPU during forward pass."""

    pin_memory: bool = True
    """Pin CPU tensors for faster H2D transfer."""

    checkpoint_layers: list[int] = field(default_factory=list)
    """Which layers to checkpoint. Empty list uses checkpoint_ratio instead."""

    checkpoint_ratio: float = 0.5
    """Fraction of layers to checkpoint when checkpoint_layers is empty."""

    profile_memory: bool = False
    """Enable memory profiling."""

    dtype: str = "float32"
    """Dtype string: 'float32' | 'float16' | 'bfloat16'."""


# ---------------------------------------------------------------------------
# dtype helper
# ---------------------------------------------------------------------------

def get_dtype(dtype_str: str) -> torch.dtype:
    """Map a dtype string to a torch.dtype.

    Args:
        dtype_str: One of "float32", "float16", "bfloat16".

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If the string is not recognised.
    """
    _map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in _map:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. Choose from {list(_map.keys())}."
        )
    return _map[dtype_str]


# ---------------------------------------------------------------------------
# CPU offload wrapper
# ---------------------------------------------------------------------------

class CPUOffloadTensor:
    """Wraps a tensor, stores it on CPU, and restores it to the original device on demand.

    Args:
        tensor: Tensor to offload.
        pin_memory: If True, pin the CPU copy for faster H2D transfers.
    """

    def __init__(self, tensor: Tensor, pin_memory: bool = True) -> None:
        cpu_tensor = tensor.detach().cpu()
        if pin_memory and cpu_tensor.is_floating_point():
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except Exception:
                pass  # silently fall back if pinning not supported
        self._cpu_tensor = cpu_tensor
        self._device = tensor.device
        self._requires_grad = tensor.requires_grad

    def restore(self) -> Tensor:
        """Move the stored CPU tensor back to the original device.

        Returns:
            Tensor on the original device.
        """
        t = self._cpu_tensor.to(self._device)
        if self._requires_grad:
            t = t.requires_grad_(True)
        return t


# ---------------------------------------------------------------------------
# Selective checkpointing helpers
# ---------------------------------------------------------------------------

def selective_checkpoint_layers(n_layers: int, config: OffloadConfig) -> list[int]:
    """Determine which layer indices to checkpoint.

    Args:
        n_layers: Total number of transformer layers.
        config: OffloadConfig controlling checkpoint behaviour.

    Returns:
        Sorted list of layer indices to apply gradient checkpointing to.
    """
    if config.checkpoint_layers:
        return sorted(config.checkpoint_layers)

    ratio = config.checkpoint_ratio
    if ratio <= 0.0:
        return []
    step = round(1 / ratio)
    return [i for i in range(n_layers) if i % step == 0]


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Track peak GPU memory usage across training steps."""

    def __init__(self) -> None:
        self.peak_mb: float = 0.0
        self.snapshots: list[float] = []

    def snapshot(self) -> float:
        """Record current GPU memory and update peak.

        Returns:
            Current allocated GPU memory in MB.
        """
        if torch.cuda.is_available():
            current_mb = torch.cuda.memory_allocated() / 1e6
        else:
            current_mb = 0.0

        if current_mb > self.peak_mb:
            self.peak_mb = current_mb
        self.snapshots.append(current_mb)
        return current_mb

    def reset(self) -> None:
        """Clear all recorded history."""
        self.peak_mb = 0.0
        self.snapshots = []

    def summary(self) -> dict[str, float]:
        """Return a summary of recorded memory statistics.

        Returns:
            Dict with keys: peak_mb, mean_mb, n_snapshots.
        """
        n = len(self.snapshots)
        mean_mb = sum(self.snapshots) / n if n > 0 else 0.0
        return {
            "peak_mb": self.peak_mb,
            "mean_mb": mean_mb,
            "n_snapshots": float(n),
        }


# ---------------------------------------------------------------------------
# Gradient checkpoint wrapper
# ---------------------------------------------------------------------------

class GradientCheckpointWrapper(nn.Module):
    """Wraps a module to apply gradient checkpointing during training.

    Args:
        module: The module to wrap.
        enabled: If False the wrapper is a no-op.
    """

    def __init__(self, module: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.module = module
        self.enabled = enabled

    def forward(self, *args, **kwargs):
        if self.enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module, *args, **kwargs, use_reentrant=False
            )
        return self.module(*args, **kwargs)


# ---------------------------------------------------------------------------
# Apply selective checkpointing to a ModuleList
# ---------------------------------------------------------------------------

def apply_selective_checkpointing(
    model_layers: nn.ModuleList, config: OffloadConfig
) -> nn.ModuleList:
    """Wrap selected layers with GradientCheckpointWrapper.

    Args:
        model_layers: ModuleList of transformer layers.
        config: OffloadConfig specifying which layers to checkpoint.

    Returns:
        New ModuleList with selected layers wrapped.
    """
    n_layers = len(model_layers)
    checkpoint_indices = set(selective_checkpoint_layers(n_layers, config))

    new_layers: list[nn.Module] = []
    for i, layer in enumerate(model_layers):
        if i in checkpoint_indices:
            new_layers.append(GradientCheckpointWrapper(layer, enabled=True))
        else:
            new_layers.append(layer)

    return nn.ModuleList(new_layers)


# ---------------------------------------------------------------------------
# Memory-efficient trainer
# ---------------------------------------------------------------------------

class MemoryEfficientTrainer:
    """Trainer with memory optimisations (offloading, checkpointing, tracking).

    Args:
        model: The neural network to train.
        optimizer: A PyTorch optimizer.
        config: OffloadConfig with training options.
    """

    def __init__(self, model: nn.Module, optimizer, config: OffloadConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.tracker = MemoryTracker()

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict[str, float]:
        """Perform a single training step.

        Args:
            input_ids: Token id tensor of shape (batch, seq_len).
            labels: Target token id tensor of shape (batch, seq_len).

        Returns:
            Dict with keys: loss (float), memory_mb (float).
        """
        self.model.train()
        self.optimizer.zero_grad()

        self.tracker.snapshot()  # before forward

        loss, _logits, _pkv = self.model(input_ids, labels=labels)

        self.tracker.snapshot()  # after forward

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "memory_mb": self.tracker.peak_mb,
        }

    def get_memory_stats(self) -> dict[str, float]:
        """Return memory statistics from the tracker.

        Returns:
            Dict from MemoryTracker.summary().
        """
        return self.tracker.summary()
