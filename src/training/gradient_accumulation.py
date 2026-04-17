"""Gradient accumulation for training with large effective batch sizes.

Supports micro-batching, gradient synchronization, and loss scaling for
memory-constrained hardware.
"""

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# MicroBatchSplitter
# ---------------------------------------------------------------------------

class MicroBatchSplitter:
    """Split a batch into micro-batches along dim=0."""

    def __init__(self, micro_batch_size: int) -> None:
        if micro_batch_size < 1:
            raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")
        self.micro_batch_size = micro_batch_size

    def split(self, *tensors: Tensor) -> list[tuple[Tensor, ...]]:
        """Split each tensor along dim=0 into chunks of micro_batch_size.

        Returns a list of tuples; each tuple contains one slice per input tensor.
        The last micro-batch may be smaller than micro_batch_size if
        batch_size % micro_batch_size != 0.
        """
        if not tensors:
            return []
        batch_size = tensors[0].size(0)
        for t in tensors[1:]:
            if t.size(0) != batch_size:
                raise ValueError("All tensors must have the same size along dim=0")

        chunks_per_tensor = [
            torch.split(t, self.micro_batch_size, dim=0) for t in tensors
        ]
        n = len(chunks_per_tensor[0])
        return [tuple(chunks_per_tensor[j][i] for j in range(len(tensors))) for i in range(n)]

    def n_micro_batches(self, batch_size: int) -> int:
        """Return ceil(batch_size / micro_batch_size)."""
        return math.ceil(batch_size / self.micro_batch_size)

    def effective_batch_size(self, n_accumulation_steps: int) -> int:
        """Return micro_batch_size * n_accumulation_steps."""
        return self.micro_batch_size * n_accumulation_steps


# ---------------------------------------------------------------------------
# GradientAccumulator
# ---------------------------------------------------------------------------

class GradientAccumulator:
    """Accumulate gradients over multiple backward passes before updating.

    Usage::

        accum = GradientAccumulator(model, optimizer, n_accumulation_steps=4)
        for i, (loss, is_last) in enumerate(compute_losses()):
            accum.step(loss, is_last=is_last)
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 n_accumulation_steps: int) -> None:
        if n_accumulation_steps < 1:
            raise ValueError(f"n_accumulation_steps must be >= 1, got {n_accumulation_steps}")
        self.model = model
        self.optimizer = optimizer
        self.n_accumulation_steps = n_accumulation_steps
        self._accumulated_steps: int = 0

    # Context-manager support (no-op enter/exit; step() drives the logic)
    def __enter__(self) -> "GradientAccumulator":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    def step(self, loss: Tensor, is_last: bool = False) -> None:
        """Scale loss, call backward, and optionally perform an optimizer step.

        Parameters
        ----------
        loss:
            The raw loss for this micro-batch.
        is_last:
            When True, clip gradients, call optimizer.step(), and zero_grad().
        """
        scaled = loss / self.n_accumulation_steps
        scaled.backward()
        self._accumulated_steps += 1

        if is_last:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulated_steps = 0

    @property
    def accumulated_steps(self) -> int:
        """Number of backward passes since the last optimizer update."""
        return self._accumulated_steps

    def grad_norm(self) -> float:
        """Total L2 norm of all current gradients."""
        total_sq: float = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().norm(2).item() ** 2
        return math.sqrt(total_sq)


# ---------------------------------------------------------------------------
# VirtualBatchTrainer
# ---------------------------------------------------------------------------

class VirtualBatchTrainer:
    """Train with a large virtual batch via gradient accumulation.

    The model sees ``virtual_batch_size`` examples per optimizer step, but
    only ``micro_batch_size`` examples are held in memory at once.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 virtual_batch_size: int, micro_batch_size: int) -> None:
        if virtual_batch_size % micro_batch_size != 0:
            raise ValueError(
                f"virtual_batch_size ({virtual_batch_size}) must be divisible "
                f"by micro_batch_size ({micro_batch_size})"
            )
        self.model = model
        self.optimizer = optimizer
        self.virtual_batch_size = virtual_batch_size
        self.micro_batch_size = micro_batch_size
        self._splitter = MicroBatchSplitter(micro_batch_size)
        self._accumulator = GradientAccumulator(model, optimizer, self.n_accumulation_steps)

    @property
    def n_accumulation_steps(self) -> int:
        """Number of micro-batches per optimizer update."""
        return self.virtual_batch_size // self.micro_batch_size

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict:
        """Run one virtual-batch training step.

        Returns
        -------
        dict with keys:
            - ``loss``: mean loss (float)
            - ``grad_norm``: L2 gradient norm after clipping (float)
            - ``n_micro_batches``: number of micro-batches processed (int)
        """
        self.optimizer.zero_grad()
        micro_batches = self._splitter.split(input_ids, labels)
        n = len(micro_batches)

        total_loss: float = 0.0
        for i, (mb_ids, mb_labels) in enumerate(micro_batches):
            is_last = (i == n - 1)
            output = self.model(mb_ids)
            loss = nn.functional.cross_entropy(
                output.view(-1, output.size(-1)), mb_labels.view(-1)
            )
            total_loss += loss.item()
            self._accumulator.step(loss, is_last=is_last)

        return {
            "loss": total_loss / n,
            "grad_norm": self._accumulator.grad_norm(),
            "n_micro_batches": n,
        }

    def verify_equivalence(self, model: nn.Module, input_ids: Tensor,
                           labels: Tensor, tol: float = 1e-4) -> bool:
        """Check that accumulated gradients match full-batch gradients within tol.

        Parameters
        ----------
        model:
            A fresh copy of the model (same architecture, same parameters).
        input_ids:
            Full batch of input IDs.
        labels:
            Full batch of labels.
        tol:
            Maximum allowed max absolute difference between gradients.

        Returns
        -------
        True if all gradient differences are within ``tol``.
        """
        # Full-batch backward on the reference model
        ref_optim = torch.optim.SGD(model.parameters(), lr=0.0)
        ref_optim.zero_grad()
        out = model(input_ids)
        loss = nn.functional.cross_entropy(
            out.view(-1, out.size(-1)), labels.view(-1)
        )
        loss.backward()
        ref_grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        # Accumulated gradients are already stored in self.model after train_step
        accum_grads = {
            name: p.grad.clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }

        # Re-run train_step so gradients are freshly accumulated
        self.optimizer.zero_grad()
        micro_batches = self._splitter.split(input_ids, labels)
        n = len(micro_batches)
        for i, (mb_ids, mb_labels) in enumerate(micro_batches):
            out = self.model(mb_ids)
            loss_mb = nn.functional.cross_entropy(
                out.view(-1, out.size(-1)), mb_labels.view(-1)
            )
            scaled = loss_mb / n
            scaled.backward()

        accum_grads = {
            name: p.grad.clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }

        for name, ref_g in ref_grads.items():
            if name not in accum_grads:
                return False
            diff = (ref_g - accum_grads[name]).abs().max().item()
            if diff > tol:
                return False
        return True


# ---------------------------------------------------------------------------
# LossScaler
# ---------------------------------------------------------------------------

class LossScaler:
    """Mixed-precision loss scaling for stable float16 training simulation.

    Tracks the current loss scale and adjusts it based on whether inf/NaN
    values are detected in gradients.
    """

    def __init__(self, init_scale: float = 65536.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000) -> None:
        self._scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._steps_since_reset: int = 0

    def scale(self, loss: Tensor) -> Tensor:
        """Return loss multiplied by the current scale factor."""
        return loss * self._scale

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        """Divide all parameter gradients by the current scale factor."""
        inv_scale = 1.0 / self._scale
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(inv_scale)

    def update(self, found_inf: bool) -> None:
        """Update the scale factor based on whether inf/NaN was found.

        Parameters
        ----------
        found_inf:
            True if any gradient was inf or NaN this step.
        """
        if found_inf:
            self._scale = self._scale * self.backoff_factor
            self._steps_since_reset = 0
        else:
            self._steps_since_reset += 1
            if self._steps_since_reset >= self.growth_interval:
                self._scale = self._scale * self.growth_factor
                self._steps_since_reset = 0

    def get_scale(self) -> float:
        """Return the current loss scale."""
        return self._scale

    def is_inf_or_nan(self, tensor: Tensor) -> bool:
        """Return True if tensor contains any inf or NaN values."""
        return bool(torch.isinf(tensor).any() or torch.isnan(tensor).any())


# ---------------------------------------------------------------------------
# AccumulationStats
# ---------------------------------------------------------------------------

class AccumulationStats:
    """Track training statistics across accumulation steps."""

    def __init__(self) -> None:
        self._micro_losses: list[float] = []
        self._micro_grad_norms: list[float] = []
        self._optimizer_steps: int = 0
        self._micro_batches_this_update: int = 0
        self._micro_batches_per_update: list[int] = []

    def record_micro_batch(self, loss: float, grad_norm: float | None = None) -> None:
        """Record stats for one micro-batch forward/backward pass."""
        self._micro_losses.append(float(loss))
        if grad_norm is not None:
            self._micro_grad_norms.append(float(grad_norm))
        self._micro_batches_this_update += 1

    def record_optimizer_step(self) -> None:
        """Record that an optimizer step occurred."""
        self._optimizer_steps += 1
        self._micro_batches_per_update.append(self._micro_batches_this_update)
        self._micro_batches_this_update = 0

    def mean_loss(self) -> float:
        """Mean loss over all recorded micro-batches since last reset."""
        if not self._micro_losses:
            return 0.0
        return sum(self._micro_losses) / len(self._micro_losses)

    def steps_per_update(self) -> float:
        """Mean number of micro-batches per optimizer step."""
        if not self._micro_batches_per_update:
            return 0.0
        return sum(self._micro_batches_per_update) / len(self._micro_batches_per_update)

    def reset(self) -> None:
        """Clear all recorded statistics."""
        self._micro_losses = []
        self._micro_grad_norms = []
        self._optimizer_steps = 0
        self._micro_batches_this_update = 0
        self._micro_batches_per_update = []
