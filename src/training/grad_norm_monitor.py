"""
src/training/grad_norm_monitor.py

Gradient norm monitoring: track per-layer and global gradient norms,
detect gradient explosions, and apply adaptive clipping based on
running statistics.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GradNormStats:
    """Statistics for gradient norms at a single training step.

    Attributes:
        global_norm:  L2 norm across all parameters with gradients.
        layer_norms:  Per-parameter-name L2 norm of .grad.
        step:         Training step index.
        clipped:      Whether gradient clipping was applied this step.
    """

    global_norm: float
    layer_norms: Dict[str, float]
    step: int
    clipped: bool = False


def compute_grad_norms(model: nn.Module, step: int = 0) -> GradNormStats:
    """Compute per-layer and global L2 gradient norms for a model.

    Parameters with None gradients are skipped.

    Args:
        model: PyTorch module whose parameters have been populated with .grad.
        step:  Current training step (used to populate GradNormStats.step).

    Returns:
        GradNormStats with global_norm, layer_norms dict, and step.
    """
    layer_norms: Dict[str, float] = {}
    sq_sum = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norm = param.grad.detach().norm(2).item()
        layer_norms[name] = norm
        sq_sum += norm ** 2

    global_norm = math.sqrt(sq_sum)
    return GradNormStats(global_norm=global_norm, layer_norms=layer_norms, step=step)


class GradNormMonitor:
    """Tracks gradient norm history and provides adaptive clipping support.

    Maintains a rolling window of global gradient norms and exposes
    percentile-based adaptive clip values to detect and respond to
    gradient explosions.

    Args:
        window_size:     Maximum number of historical global norms to retain.
                         Default: 100.
        clip_percentile: Percentile of the history used as the adaptive clip
                         baseline. Default: 95.0.
    """

    def __init__(self, window_size: int = 100, clip_percentile: float = 95.0) -> None:
        self.window_size = window_size
        self.clip_percentile = clip_percentile
        self.history: deque[float] = deque(maxlen=window_size)

    def update(self, stats: GradNormStats) -> None:
        """Append the global norm from stats to the rolling history.

        Args:
            stats: GradNormStats from the current step.
        """
        self.history.append(stats.global_norm)

    def adaptive_clip_value(self) -> float:
        """Return the percentile-based adaptive clip threshold.

        Returns:
            Clip threshold based on clip_percentile of history,
            or float('inf') if history is empty.
        """
        if not self.history:
            return float("inf")
        return float(np.percentile(list(self.history), self.clip_percentile))

    def should_clip(self, stats: GradNormStats, multiplier: float = 2.0) -> bool:
        """Determine whether gradient clipping should be applied.

        Returns True when the current global norm exceeds
        multiplier * adaptive_clip_value().

        Args:
            stats:      GradNormStats from the current step.
            multiplier: How many times the adaptive clip value the current
                        norm must exceed to trigger clipping. Default: 2.0.

        Returns:
            True if clipping is warranted, False otherwise.
        """
        clip_val = self.adaptive_clip_value()
        if clip_val == float("inf"):
            return False
        return stats.global_norm > multiplier * clip_val

    def summary(self) -> dict:
        """Return summary statistics over the norm history.

        Returns:
            Dict with keys 'mean', 'std', 'max', 'min'. All values are 0.0
            if history is empty.
        """
        if not self.history:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        arr = np.array(list(self.history), dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "min": float(arr.min()),
        }


def clip_grad_norm_adaptive(
    model: nn.Module,
    monitor: GradNormMonitor,
    multiplier: float = 2.0,
) -> Tuple[float, bool]:
    """Apply adaptive gradient norm clipping based on running statistics.

    Computes the current global gradient norm and clips if the norm
    exceeds multiplier * monitor.adaptive_clip_value().

    Args:
        model:      PyTorch module with populated .grad attributes.
        monitor:    GradNormMonitor with history of previous norms.
        multiplier: Clip trigger threshold relative to adaptive clip value.
                    Default: 2.0.

    Returns:
        Tuple of (actual_norm, was_clipped).
    """
    stats = compute_grad_norms(model)
    was_clipped = monitor.should_clip(stats, multiplier=multiplier)

    if was_clipped:
        clip_val = monitor.adaptive_clip_value()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)

    return stats.global_norm, was_clipped
