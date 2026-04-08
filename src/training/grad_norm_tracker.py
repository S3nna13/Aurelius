"""Gradient norm tracker with EMA-based spike detection and adaptive clipping.

Usage:
    tracker = GradNormTracker(model, GradNormConfig())
    loss.backward()
    metrics = tracker.step()          # call after backward, before optimizer.step()
    norm_before = clip_by_adaptive_norm(model, tracker)
    optimizer.step()
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class GradNormConfig:
    ema_alpha: float = 0.99       # EMA smoothing factor for norm history
    spike_threshold: float = 3.0  # flag if global_norm > spike_threshold * ema_norm
    window_size: int = 100        # rolling window size for statistics
    track_per_layer: bool = True  # track individual parameter grad norms


class GradNormTracker:
    """Tracks per-layer and global gradient norms during training.

    Call ``step()`` after ``loss.backward()`` and before ``optimizer.step()``.
    """

    def __init__(self, model: nn.Module, cfg: GradNormConfig) -> None:
        self.model = model
        self.cfg = cfg

        self._step: int = 0
        self._ema_norm: float | None = None            # lazily initialised
        self._window: deque[float] = deque(maxlen=cfg.window_size)
        self._spike_flags: deque[bool] = deque(maxlen=cfg.window_size)
        self._last_layer_norms: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> dict[str, float]:
        """Compute norms, update EMA/window, return metrics dict.

        Must be called after ``loss.backward()`` and before
        ``optimizer.step()``.
        """
        global_norm = compute_global_grad_norm(self.model)

        # Update EMA
        if self._ema_norm is None:
            self._ema_norm = global_norm
        else:
            alpha = self.cfg.ema_alpha
            self._ema_norm = alpha * self._ema_norm + (1.0 - alpha) * global_norm

        # Spike detection (guard against ema_norm == 0)
        if self._ema_norm > 0.0:
            is_spike = global_norm > self.cfg.spike_threshold * self._ema_norm
        else:
            is_spike = False

        # Update rolling window
        self._window.append(global_norm)
        self._spike_flags.append(is_spike)

        # Per-layer norms
        if self.cfg.track_per_layer:
            self._last_layer_norms = _compute_per_param_norms(self.model)

        self._step += 1

        return {
            "global_norm": float(global_norm),
            "ema_norm": float(self._ema_norm),
            "is_spike": bool(is_spike),
            "step": self._step,
            "recommended_clip": float(2.0 * self._ema_norm),
        }

    def get_per_layer_norms(self) -> dict[str, float]:
        """Return ``{param_name: grad_norm}`` for all params with gradients.

        Returns the snapshot captured during the most recent ``step()`` call
        (or computes fresh if ``track_per_layer`` was disabled).
        """
        if self.cfg.track_per_layer:
            return dict(self._last_layer_norms)
        return _compute_per_param_norms(self.model)

    def get_stats(self) -> dict[str, float]:
        """Aggregate statistics over the rolling window.

        Returns
        -------
        dict with keys: mean_norm, std_norm, max_norm, min_norm,
                        n_spikes (int stored as float), spike_rate.
        """
        if not self._window:
            return {
                "mean_norm": 0.0,
                "std_norm": 0.0,
                "max_norm": 0.0,
                "min_norm": 0.0,
                "n_spikes": 0,
                "spike_rate": 0.0,
            }

        norms = list(self._window)
        n = len(norms)
        mean = sum(norms) / n
        variance = sum((x - mean) ** 2 for x in norms) / n
        std = math.sqrt(variance)
        n_spikes = sum(self._spike_flags)
        spike_rate = n_spikes / n if n > 0 else 0.0

        return {
            "mean_norm": float(mean),
            "std_norm": float(std),
            "max_norm": float(max(norms)),
            "min_norm": float(min(norms)),
            "n_spikes": int(n_spikes),
            "spike_rate": float(spike_rate),
        }

    def reset(self) -> None:
        """Clear history and reset step counter (e.g. at the start of a new epoch)."""
        self._step = 0
        self._ema_norm = None
        self._window.clear()
        self._spike_flags.clear()
        self._last_layer_norms = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ema_norm(self) -> float:
        """Current EMA of the global gradient norm (0.0 before first step)."""
        return float(self._ema_norm) if self._ema_norm is not None else 0.0


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def compute_global_grad_norm(model: nn.Module) -> float:
    """Compute sqrt(sum of squared grad norms) across all parameters.

    Returns 0.0 if no parameter has a gradient.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total_sq)


def clip_by_adaptive_norm(
    model: nn.Module,
    tracker: GradNormTracker,
    multiplier: float = 2.0,
) -> float:
    """Clip gradients using ``tracker.ema_norm * multiplier`` as the threshold.

    Parameters
    ----------
    model:
        The model whose gradients are to be clipped.
    tracker:
        A ``GradNormTracker`` that has had at least one ``step()`` recorded so
        that ``ema_norm`` is non-zero.
    multiplier:
        Scales the EMA norm to produce the clip threshold (default 2.0).

    Returns
    -------
    float
        The actual global gradient norm *before* clipping.
    """
    pre_clip_norm = compute_global_grad_norm(model)
    max_norm = tracker.ema_norm * multiplier
    if max_norm > 0.0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    return pre_clip_norm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_per_param_norms(model: nn.Module) -> dict[str, float]:
    """Return ``{param_name: grad_norm}`` for all params that have gradients."""
    result: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            result[name] = float(p.grad.detach().float().norm(2).item())
    return result
