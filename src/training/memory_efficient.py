"""Memory-efficient training utilities for Aurelius.

Provides gradient checkpointing wrappers, chunked cross-entropy loss,
gradient accumulation, memory tracking, and a memory-efficient trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MemEffConfig:
    """Configuration for memory-efficient training utilities."""

    use_checkpointing: bool = True
    chunk_size: int = 1
    offload_optimizer: bool = False
    mixed_precision: bool = False


# ---------------------------------------------------------------------------
# Checkpointed forward
# ---------------------------------------------------------------------------


def checkpointed_forward(module: nn.Module, *args: Any) -> Tensor:
    """Wrapper around torch.utils.checkpoint.checkpoint.

    Recomputes activations during backward pass to save memory.
    """
    return checkpoint(module, *args, use_reentrant=False)


# ---------------------------------------------------------------------------
# Chunked cross-entropy
# ---------------------------------------------------------------------------


def chunked_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    chunk_size: int,
    ignore_index: int = -100,
) -> Tensor:
    """Compute cross-entropy loss in chunks along the sequence dimension.

    Avoids materialising the full (B*S, V) loss tensor at once by processing
    the sequence in chunks of size `chunk_size`.

    Args:
        logits: (B, S, V) — raw unnormalized scores.
        labels: (B, S) — target token ids.
        chunk_size: Number of sequence positions processed per chunk.
        ignore_index: Token id to ignore in the loss.

    Returns:
        Scalar cross-entropy loss.
    """
    B, S, V = logits.shape
    total_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    n_valid = 0

    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        chunk_logits = logits[:, start:end, :].reshape(-1, V)
        chunk_labels = labels[:, start:end].reshape(-1)

        # Count non-ignored tokens in this chunk
        valid_mask = chunk_labels != ignore_index
        n_valid += valid_mask.sum().item()

        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss = total_loss + chunk_loss

    if n_valid == 0:
        return total_loss  # 0, avoids division by zero
    return total_loss / n_valid


# ---------------------------------------------------------------------------
# Gradient accumulator
# ---------------------------------------------------------------------------


class GradientAccumulator:
    """Accumulates gradients across micro-batches.

    The caller is responsible for calling optimizer.step() when step()
    returns True.

    Args:
        model: The model whose parameters will receive gradients.
        n_accumulate: Number of micro-batches per optimizer step.
    """

    def __init__(self, model: nn.Module, n_accumulate: int) -> None:
        self.model = model
        self.n_accumulate = n_accumulate
        self._current_step = 0

    def step(self, loss: Tensor) -> bool:
        """Backward pass with scaled loss.

        Args:
            loss: Scalar loss tensor for the current micro-batch.

        Returns:
            True when it is time for the caller to call optimizer.step()
            (i.e., every n_accumulate calls).
        """
        scaled = loss / self.n_accumulate
        scaled.backward()
        self._current_step += 1
        return (self._current_step % self.n_accumulate) == 0

    def zero_grad(self) -> None:
        """Zero gradients on all model parameters."""
        for p in self.model.parameters():
            p.grad = None

    @property
    def current_step(self) -> int:
        """Total number of micro-batch steps taken so far."""
        return self._current_step


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------


class MemoryTracker:
    """Tracks peak CUDA memory usage.

    On CPU-only environments all values will be 0.0, which is handled
    gracefully.
    """

    def __init__(self) -> None:
        self._peak_mb: float = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._peak_mb = 0.0

    def record(self) -> dict[str, float]:
        """Capture current and peak memory usage.

        Returns:
            dict with keys "allocated_mb", "reserved_mb", "peak_mb" (all MB).
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
        else:
            allocated = 0.0
            reserved = 0.0
            peak = 0.0
        self._peak_mb = peak
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "peak_mb": peak,
        }

    def __enter__(self) -> MemoryTracker:
        self.reset()
        return self

    def __exit__(self, *args: Any) -> None:
        self.record()


# ---------------------------------------------------------------------------
# Memory-efficient trainer
# ---------------------------------------------------------------------------


class MemEfficientTrainer:
    """Trainer that applies memory-efficient techniques.

    Combines chunked cross-entropy, optional gradient checkpointing
    (delegated to AureliusConfig.use_gradient_checkpointing), and memory
    tracking per training step.

    Args:
        model: AureliusTransformer instance.
        config: MemEffConfig controlling which techniques are active.
        optimizer: PyTorch optimizer attached to model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MemEffConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self._tracker = MemoryTracker()

    def train_step(self, input_ids: Tensor) -> dict[str, float]:
        """Run one training step.

        Computes chunked cross-entropy over the model's logits, calls
        backward, and steps the optimizer.

        Args:
            input_ids: (B, S) integer token ids.

        Returns:
            dict with "loss" (float) and "peak_mb" (float).
        """
        self.model.train()
        self.optimizer.zero_grad()

        with self._tracker:
            # model(input_ids) returns (loss, logits, pkv) — loss may be None
            # since we are computing our own chunked CE loss we pass no labels.
            _, logits, _ = self.model(input_ids)

            # Build shifted labels: predict next token
            B, S, V = logits.shape
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = chunked_cross_entropy(
                shift_logits,
                shift_labels,
                chunk_size=self.config.chunk_size,
            )

            loss.backward()

        self.optimizer.step()

        stats = self._tracker.record()
        return {
            "loss": loss.item(),
            "peak_mb": stats["peak_mb"],
        }
