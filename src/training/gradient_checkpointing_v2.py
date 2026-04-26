"""Gradient checkpointing utilities for Aurelius — v2.

Provides selective checkpointing, segment-level control, and memory profiling
using pure native PyTorch (torch.utils.checkpoint).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing.

    Attributes:
        enabled: Whether gradient checkpointing is active.
        checkpoint_every: Wrap every Nth layer (1 = every layer, 2 = every other, …).
        use_reentrant: Passed to ``torch.utils.checkpoint.checkpoint``. ``False``
            is recommended for newer PyTorch versions and ``torch.compile``
            compatibility.
    """

    enabled: bool = True
    checkpoint_every: int = 1
    use_reentrant: bool = False


# ---------------------------------------------------------------------------
# CheckpointedLayer
# ---------------------------------------------------------------------------


class CheckpointedLayer(nn.Module):
    """Wraps any ``nn.Module`` to apply gradient checkpointing during training.

    When ``config.enabled`` is ``True`` **and** the model is in training mode
    the forward call is routed through
    ``torch.utils.checkpoint.checkpoint``, causing activations to be
    recomputed during the backward pass instead of stored.  In eval mode or
    when checkpointing is disabled the wrapped module is called directly.

    Args:
        module: The inner module to wrap.
        config: :class:`CheckpointConfig` controlling checkpointing behaviour.
    """

    def __init__(self, module: nn.Module, config: CheckpointConfig) -> None:
        super().__init__()
        self._module = module
        self.config = config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wrapped(self) -> nn.Module:
        """Return the inner (unwrapped) module."""
        return self._module

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Forward pass with optional gradient checkpointing.

        If ``config.enabled`` is ``True`` and the layer is in training mode
        the call is checkpointed; otherwise it is a direct pass-through.

        Note:
            ``torch.utils.checkpoint.checkpoint`` does not support keyword
            arguments directly.  Keyword arguments are captured via closure so
            that the checkpoint wrapper receives only positional tensors.
        """
        if self.config.enabled and self.training:
            # Capture kwargs in a closure — checkpoint only accepts *args.
            def run(*a):
                return self._module(*a, **kwargs)

            return torch.utils.checkpoint.checkpoint(
                run, *args, use_reentrant=self.config.use_reentrant
            )
        return self._module(*args, **kwargs)


# ---------------------------------------------------------------------------
# SelectiveCheckpointing
# ---------------------------------------------------------------------------


class SelectiveCheckpointing:
    """Wraps a list of layers with selective gradient checkpointing.

    Only layers whose index satisfies ``i % config.checkpoint_every == 0``
    are wrapped in :class:`CheckpointedLayer`.

    Args:
        layers: Flat list of ``nn.Module`` instances (e.g. transformer blocks).
        config: :class:`CheckpointConfig` controlling which layers to wrap.
    """

    def __init__(self, layers: list[nn.Module], config: CheckpointConfig) -> None:
        self._layers = list(layers)
        self.config = config

    def wrap(self) -> list[CheckpointedLayer]:
        """Return a list where qualifying layers are wrapped.

        Layer *i* is wrapped in :class:`CheckpointedLayer` when
        ``i % config.checkpoint_every == 0``; other layers are also wrapped
        (so the returned list has the same length and type), but they pass
        through with an effectively disabled-per-layer config clone.

        Returns:
            A list of :class:`CheckpointedLayer` objects the same length as
            the original layer list.
        """
        result: list[CheckpointedLayer] = []
        for i, layer in enumerate(self._layers):
            should_checkpoint = i % self.config.checkpoint_every == 0
            # Build a per-layer config that respects both the global flag and
            # the selective criterion.
            layer_config = CheckpointConfig(
                enabled=self.config.enabled and should_checkpoint,
                checkpoint_every=self.config.checkpoint_every,
                use_reentrant=self.config.use_reentrant,
            )
            result.append(CheckpointedLayer(layer, layer_config))
        return result

    @staticmethod
    def unwrap(layers: list[CheckpointedLayer]) -> list[nn.Module]:
        """Extract the original inner modules from a list of :class:`CheckpointedLayer`.

        Args:
            layers: List of :class:`CheckpointedLayer` instances returned by
                :meth:`wrap`.

        Returns:
            List of the original ``nn.Module`` objects.
        """
        return [cl.wrapped for cl in layers]


# ---------------------------------------------------------------------------
# MemoryStats
# ---------------------------------------------------------------------------


@dataclass
class MemoryStats:
    """Snapshot of current PyTorch GPU memory usage.

    All values are in megabytes.  On CPU-only machines all fields are ``0.0``.

    Attributes:
        peak_allocated_mb: Peak memory allocated by tensors since the last
            reset (or program start).
        current_allocated_mb: Memory currently held by live tensors.
        reserved_mb: Memory reserved (cached) by the CUDA allocator.
    """

    peak_allocated_mb: float
    current_allocated_mb: float
    reserved_mb: float

    @classmethod
    def capture(cls) -> MemoryStats:
        """Capture a memory snapshot from the current PyTorch allocator state.

        Uses ``torch.cuda`` statistics when a CUDA device is available;
        returns zeros on CPU-only setups.

        Returns:
            A :class:`MemoryStats` instance reflecting the current state.
        """
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            current = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
        else:
            peak = 0.0
            current = 0.0
            reserved = 0.0
        return cls(
            peak_allocated_mb=peak,
            current_allocated_mb=current,
            reserved_mb=reserved,
        )


# ---------------------------------------------------------------------------
# CheckpointingBenchmark
# ---------------------------------------------------------------------------


class CheckpointingBenchmark:
    """Measures forward/backward timing and memory usage for a given model.

    Args:
        model: The ``nn.Module`` to benchmark.
        config: :class:`CheckpointConfig` applied during benchmarking.
    """

    def __init__(self, model: nn.Module, config: CheckpointConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_forward(self, x: torch.Tensor, n_iterations: int = 3) -> dict[str, float]:
        """Benchmark forward and backward passes.

        Runs ``n_iterations`` forward+backward cycles **without**
        checkpointing (the model is not re-wrapped here; the caller controls
        that) and reports average timing and peak memory.

        Args:
            x: Input tensor fed to ``self.model``.
            n_iterations: Number of warm+timed iterations.

        Returns:
            A dict with keys:

            * ``"forward_ms"`` — average forward time in milliseconds.
            * ``"backward_ms"`` — average backward time in milliseconds.
            * ``"peak_memory_mb"`` — peak CUDA memory in MB (0 on CPU).
        """
        self.model.train()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        forward_times: list[float] = []
        backward_times: list[float] = []

        for _ in range(n_iterations):
            # Zero any existing gradients.
            self.model.zero_grad(set_to_none=True)

            # ---- Forward ----
            t0 = time.perf_counter()
            out = self.model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            forward_times.append((t1 - t0) * 1000.0)

            # Reduce to scalar for backward.
            loss = out.sum()

            # ---- Backward ----
            t2 = time.perf_counter()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            backward_times.append((t3 - t2) * 1000.0)

        stats = MemoryStats.capture()

        return {
            "forward_ms": sum(forward_times) / len(forward_times),
            "backward_ms": sum(backward_times) / len(backward_times),
            "peak_memory_mb": stats.peak_allocated_mb,
        }

    @staticmethod
    def estimate_savings(original_layers: int, checkpointed_every: int) -> float:
        """Theoretical fraction of activation memory freed by checkpointing.

        Formula: ``1 - 1 / checkpointed_every``

        For ``checkpointed_every=2`` half the activations are freed → ``0.5``.
        For ``checkpointed_every=4`` three-quarters are freed → ``0.75``.

        Args:
            original_layers: Total number of layers (unused in formula but
                kept for API symmetry / future extension).
            checkpointed_every: Checkpoint interval *N*.

        Returns:
            Fraction of activations freed (between 0.0 and 1.0).
        """
        return 1.0 - 1.0 / checkpointed_every
