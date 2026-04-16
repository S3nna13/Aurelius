"""Online learning v2: incremental model updates on streaming data with catastrophic forgetting prevention.

Provides:
  OnlineConfig, StreamingBuffer, compute_parameter_distance,
  compute_online_ewc_loss, OnlineLearner, compute_forgetting_metric.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OnlineConfig:
    """Configuration for the online learner."""
    buffer_size: int = 1000
    update_frequency: int = 100
    lr: float = 1e-4
    forgetting_penalty: float = 0.0
    ewc_lambda: float = 0.0
    momentum: float = 0.0
    min_loss_threshold: float = 0.0  # skip update if loss < this


# ---------------------------------------------------------------------------
# StreamingBuffer
# ---------------------------------------------------------------------------

class StreamingBuffer:
    """Fixed-size circular buffer for streaming data (FIFO eviction)."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._data: List[Any] = []
        self._head: int = 0  # index of next write position when full

    def add(self, item: Any) -> None:
        """Add item; evict oldest if at capacity."""
        if len(self._data) < self._capacity:
            self._data.append(item)
        else:
            self._data[self._head] = item
            self._head = (self._head + 1) % self._capacity

    def sample(self, n: int) -> List[Any]:
        """Random sample of min(n, len(buffer)) items without replacement."""
        k = min(n, len(self._data))
        return random.sample(self._data, k)

    def __len__(self) -> int:
        return len(self._data)

    def is_full(self) -> bool:
        return len(self._data) == self._capacity


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_parameter_distance(params1: List[Tensor], params2: List[Tensor]) -> float:
    """L2 distance between two parameter lists."""
    if len(params1) != len(params2):
        raise ValueError("Parameter lists must have the same length")
    total = torch.tensor(0.0)
    for p1, p2 in zip(params1, params2):
        diff = (p1.float() - p2.float()).reshape(-1)
        total = total + torch.dot(diff, diff)
    return float(total.sqrt().item())


def compute_online_ewc_loss(
    model: nn.Module,
    fisher_diags: List[Tensor],
    anchors: List[Tensor],
    lambda_: float,
) -> Tensor:
    """EWC regularisation: lambda/2 * sum_i( F_i * (theta_i - theta_i*)^2 ).

    Returns a scalar tensor.
    """
    params = [p for p in model.parameters()]
    if len(params) != len(fisher_diags) or len(params) != len(anchors):
        raise ValueError("params, fisher_diags, and anchors must all have the same length")

    loss = torch.zeros(1, dtype=torch.float32)
    for p, f, a in zip(params, fisher_diags, anchors):
        loss = loss + (f * (p - a) ** 2).sum()
    return (lambda_ / 2.0) * loss.squeeze()


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------

class OnlineLearner:
    """Incrementally updates a model on streaming (x, y) pairs."""

    def __init__(self, model: nn.Module, config: OnlineConfig) -> None:
        self.model = model
        self.config = config
        # Store initial parameter values for drift calculation
        self._init_params: List[Tensor] = [
            p.detach().clone() for p in model.parameters()
        ]
        # AdamW optimizer
        self._optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
        )

    def observe(self, x: Tensor, y: Tensor) -> float:
        """Single gradient update on (x, y). Returns loss as float.

        Skips the parameter update if loss < min_loss_threshold.
        """
        self.model.train()
        self._optimizer.zero_grad()

        output = self.model(x)
        # Support both (batch, classes) + (batch,) and scalar outputs
        if output.shape == y.shape:
            loss = nn.functional.mse_loss(output, y.float())
        else:
            loss = nn.functional.cross_entropy(output, y)

        loss_val = float(loss.item())

        if loss_val < self.config.min_loss_threshold:
            return loss_val

        loss.backward()
        self._optimizer.step()
        return loss_val

    def observe_batch(self, xs: List[Tensor], ys: List[Tensor]) -> List[float]:
        """Observe each (x, y) pair; return list of losses."""
        return [self.observe(x, y) for x, y in zip(xs, ys)]

    def get_parameter_drift(self) -> float:
        """L2 distance from initial parameters."""
        current = [p.detach().clone() for p in self.model.parameters()]
        return compute_parameter_distance(current, self._init_params)

    def save_checkpoint(self) -> dict:
        """Return a copy of the current model state dict."""
        return copy.deepcopy(self.model.state_dict())

    def reset_to_checkpoint(self, checkpoint: dict) -> None:
        """Restore model to a previously saved state dict."""
        self.model.load_state_dict(checkpoint)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def compute_forgetting_metric(
    losses_before: List[float],
    losses_after: List[float],
) -> float:
    """Mean(losses_after - losses_before).

    Positive  => forgetting (losses went up).
    Negative  => improvement (losses went down).
    """
    if len(losses_before) != len(losses_after):
        raise ValueError("losses_before and losses_after must have the same length")
    if not losses_before:
        return 0.0
    diffs = [a - b for a, b in zip(losses_after, losses_before)]
    return sum(diffs) / len(diffs)
