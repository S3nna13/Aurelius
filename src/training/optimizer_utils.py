"""Optimizer utilities for stable LLM training.

Provides parameter group splitting, gradient norm computation and clipping,
learning-rate scheduling, AdamW construction, and gradient accumulation.

Usage::

    config = OptimizerConfig(lr=3e-4, warmup_steps=100, total_steps=1000)
    optimizer = build_optimizer(model, config)

    accumulator = GradientAccumulator(accumulation_steps=4, model=model)
    for step, (x, y) in enumerate(dataloader):
        loss = criterion(model(x), y)
        should_step = accumulator.accumulate(loss)
        if should_step:
            lr_mult = get_lr_schedule(step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * lr_mult
            optimizer.step()
            optimizer.zero_grad()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Hyperparameter configuration for the optimizer and LR schedule."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 100_000
    schedule: str = "cosine"  # "cosine" | "linear" | "constant"

    def __post_init__(self) -> None:
        valid = {"cosine", "linear", "constant"}
        if self.schedule not in valid:
            raise ValueError(f"schedule must be one of {valid}, got {self.schedule!r}")


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------


def get_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split model parameters into two groups for differential weight decay.

    Parameters with ndim >= 2 (weights) receive ``weight_decay``; parameters
    with ndim < 2 (biases, LayerNorm scale/bias) receive ``weight_decay=0``.

    Parameters
    ----------
    model:
        The PyTorch module whose parameters should be grouped.
    weight_decay:
        Weight-decay coefficient for 2-D+ parameter tensors.

    Returns
    -------
    List of two dicts suitable for passing to an optimizer constructor.
    """
    wd_params: list[torch.Tensor] = []
    no_wd_params: list[torch.Tensor] = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            wd_params.append(p)
        else:
            no_wd_params.append(p)

    return [
        {"params": wd_params, "weight_decay": weight_decay},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Gradient norm helpers
# ---------------------------------------------------------------------------


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the global L2 gradient norm across all parameters with gradients.

    Parameters
    ----------
    model:
        The PyTorch module to inspect.

    Returns
    -------
    float
        sqrt(sum of squared per-gradient L2 norms), or 0.0 if no grads exist.
    """
    total_sq: float = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total_sq += g.norm(2).item() ** 2
    return float(math.sqrt(total_sq))


def clip_grad_norm_custom(model: nn.Module, max_norm: float) -> float:
    """Clip gradients by global L2 norm, returning the pre-clip norm.

    Implements clipping from scratch without calling
    ``torch.nn.utils.clip_grad_norm_``.

    Parameters
    ----------
    model:
        The module whose gradients will be clipped in-place.
    max_norm:
        Maximum allowed global gradient norm.

    Returns
    -------
    float
        The global gradient norm *before* clipping.
    """
    pre_clip = compute_grad_norm(model)

    if pre_clip > max_norm and pre_clip > 0.0:
        scale = max_norm / pre_clip
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(scale)

    return pre_clip


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def get_lr_schedule(step: int, config: OptimizerConfig) -> float:
    """Compute the LR multiplier at the given training step.

    The multiplier is applied to ``config.lr`` to obtain the actual learning
    rate.  During the warmup phase the multiplier rises linearly from 0 to 1.
    After warmup, the schedule continues according to ``config.schedule``:

    * ``"cosine"``   — cosine decay from 1.0 to 0.1 × lr
    * ``"linear"``   — linear decay from 1.0 to 0.0
    * ``"constant"`` — stays at 1.0

    Parameters
    ----------
    step:
        Current training step (0-indexed).
    config:
        Optimizer/schedule configuration.

    Returns
    -------
    float
        Multiplier in [0, 1] (approximately — cosine minimum is 0.1).
    """
    warmup = config.warmup_steps
    total = config.total_steps

    # Linear warmup
    if step < warmup:
        return float(step) / max(1, warmup)

    # Post-warmup
    progress = float(step - warmup) / max(1, total - warmup)
    progress = min(progress, 1.0)

    if config.schedule == "cosine":
        # Decay from 1.0 to min_lr_ratio (0.1)
        min_ratio = 0.1
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    elif config.schedule == "linear":
        return 1.0 - progress
    else:  # "constant"
        return 1.0


# ---------------------------------------------------------------------------
# Build optimizer
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    """Construct an AdamW optimizer with weight-decay param groups.

    Parameters
    ----------
    model:
        The model to optimize.
    config:
        Optimizer configuration.

    Returns
    -------
    torch.optim.AdamW
    """
    param_groups = get_param_groups(model, config.weight_decay)
    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


class GradientAccumulator:
    """Accumulate gradients over multiple micro-steps before an optimizer step.

    Each call to :meth:`accumulate` scales the loss by ``1/accumulation_steps``
    and calls ``backward()``.  The method returns ``True`` every
    ``accumulation_steps`` calls, signalling that the caller should invoke
    ``optimizer.step()`` and ``optimizer.zero_grad()``.

    Parameters
    ----------
    accumulation_steps:
        Number of micro-batches to accumulate before a parameter update.
    model:
        The model being trained (unused internally but stored for reference).
    """

    def __init__(self, accumulation_steps: int, model: nn.Module) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        self._accumulation_steps = accumulation_steps
        self._model = model
        self._call_count: int = 0
        self._complete_cycles: int = 0

    def accumulate(self, loss: torch.Tensor) -> bool:
        """Scale and backward the loss; return True when a step should be taken.

        Parameters
        ----------
        loss:
            Scalar loss tensor for this micro-batch.

        Returns
        -------
        bool
            ``True`` if ``accumulation_steps`` micro-batches have been
            accumulated and the caller should call ``optimizer.step()``.
        """
        scaled = loss / self._accumulation_steps
        scaled.backward()
        self._call_count += 1

        if self._call_count % self._accumulation_steps == 0:
            self._complete_cycles += 1
            return True
        return False

    def step_count(self) -> int:
        """Return the number of complete accumulation cycles completed so far."""
        return self._complete_cycles

    def reset(self) -> None:
        """Reset the internal call counter without zeroing gradients."""
        self._call_count = 0
        self._complete_cycles = 0


# ---------------------------------------------------------------------------
# Effective learning rate
# ---------------------------------------------------------------------------


def compute_effective_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate from the first parameter group.

    Parameters
    ----------
    optimizer:
        Any PyTorch optimizer.

    Returns
    -------
    float
        Learning rate of the first param group.
    """
    return float(optimizer.param_groups[0]["lr"])
