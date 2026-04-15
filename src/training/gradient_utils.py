"""Gradient utilities: norm computation, clipping, monitoring, and diagnostics.

Usage::

    config = GradientConfig(max_norm=1.0)
    monitor = GradientMonitor(config)
    loss.backward()
    metrics = monitor.record(model.parameters())
    optimizer.step()
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GradientConfig:
    """Configuration for gradient clipping and monitoring."""

    max_norm: float = 1.0
    norm_type: float = 2.0
    clip_value: Optional[float] = None
    warn_threshold: float = 10.0


# ---------------------------------------------------------------------------
# Standalone gradient norm helpers
# ---------------------------------------------------------------------------


def _iter_params(
    parameters: Union[nn.Module, Iterable[torch.Tensor]],
) -> Iterator[torch.Tensor]:
    """Yield individual parameter tensors from a module or an iterable."""
    if isinstance(parameters, nn.Module):
        yield from parameters.parameters()
    else:
        yield from parameters


def compute_grad_norm(
    parameters: Union[nn.Module, Iterable[torch.Tensor]],
    norm_type: float = 2.0,
) -> float:
    """Compute the global gradient norm across all parameters that have gradients.

    This is read-only (no clipping is applied).  The result is equivalent to
    the pre-clip norm returned by ``torch.nn.utils.clip_grad_norm_``.

    Parameters
    ----------
    parameters:
        A model or an iterable of parameter tensors.
    norm_type:
        Type of the norm (e.g. 2.0 for L2, ``float('inf')`` for max-norm).

    Returns
    -------
    float
        Global gradient norm.
    """
    grads = [p.grad.detach() for p in _iter_params(parameters) if p.grad is not None]
    if not grads:
        return 0.0

    if norm_type == float("inf"):
        return float(max(g.abs().max() for g in grads))

    total = sum(g.float().norm(norm_type).item() ** norm_type for g in grads)
    return float(total ** (1.0 / norm_type))


def clip_grad_norm(
    parameters: Union[nn.Module, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradient norm in-place; return the pre-clip global norm.

    Parameters
    ----------
    parameters:
        A model or an iterable of parameter tensors.
    max_norm:
        Maximum allowed norm.
    norm_type:
        Type of the norm.

    Returns
    -------
    float
        Global gradient norm *before* clipping.
    """
    params = list(_iter_params(parameters))
    pre_clip = compute_grad_norm(params, norm_type)
    nn.utils.clip_grad_norm_(
        [p for p in params if p.grad is not None],
        max_norm=max_norm,
        norm_type=norm_type,
    )
    return pre_clip


def clip_grad_value(
    parameters: Union[nn.Module, Iterable[torch.Tensor]],
    clip_value: float,
) -> None:
    """Clip each gradient element to the range ``[-clip_value, clip_value]``.

    Parameters
    ----------
    parameters:
        A model or an iterable of parameter tensors.
    clip_value:
        Absolute clipping bound.
    """
    nn.utils.clip_grad_value_(
        [p for p in _iter_params(parameters) if p.grad is not None],
        clip_value=clip_value,
    )


# ---------------------------------------------------------------------------
# GradientMonitor
# ---------------------------------------------------------------------------


class GradientMonitor:
    """Records gradient norms and optionally applies clipping each step.

    Parameters
    ----------
    config:
        Controls clipping behaviour and warning thresholds.
    """

    def __init__(self, config: GradientConfig) -> None:
        self.config = config
        self._history: List[Dict[str, float]] = []

    def record(
        self,
        parameters: Union[nn.Module, Iterable[torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute norm, apply clipping if configured, and record the step.

        Parameters
        ----------
        parameters:
            A model or an iterable of parameter tensors.

        Returns
        -------
        dict with keys:
            * ``grad_norm``   — norm before any clipping
            * ``clipped_norm``— norm after clipping (same as ``grad_norm`` when
                                no clipping was applied)
            * ``was_clipped`` — 1.0 if clipping was applied, else 0.0
            * ``clip_ratio``  — ``grad_norm / max_norm`` (0.0 when max_norm ≤ 0)
        """
        params = list(_iter_params(parameters))
        grad_norm = compute_grad_norm(params, self.config.norm_type)

        was_clipped = False

        # Apply value clipping first (if configured)
        if self.config.clip_value is not None:
            clip_grad_value(params, self.config.clip_value)

        # Apply norm clipping
        if self.config.max_norm > 0 and grad_norm > self.config.max_norm:
            clip_grad_norm(params, self.config.max_norm, self.config.norm_type)
            was_clipped = True

        clipped_norm = compute_grad_norm(params, self.config.norm_type)

        clip_ratio = (
            grad_norm / self.config.max_norm if self.config.max_norm > 0 else 0.0
        )

        entry: Dict[str, float] = {
            "grad_norm": grad_norm,
            "clipped_norm": clipped_norm,
            "was_clipped": float(was_clipped),
            "clip_ratio": clip_ratio,
        }
        self._history.append(entry)
        return entry

    def get_history(self) -> List[Dict[str, float]]:
        """Return all recorded entries."""
        return list(self._history)

    def get_stats(self) -> Dict[str, float]:
        """Aggregate statistics over all recorded steps.

        Returns
        -------
        dict with keys:
            * ``mean_grad_norm``  — mean of recorded gradient norms
            * ``max_grad_norm``   — maximum recorded gradient norm
            * ``clip_frequency``  — fraction of steps where clipping occurred
            * ``n_steps``         — total number of recorded steps
        """
        if not self._history:
            return {
                "mean_grad_norm": 0.0,
                "max_grad_norm": 0.0,
                "clip_frequency": 0.0,
                "n_steps": 0.0,
            }

        norms = [e["grad_norm"] for e in self._history]
        n = len(norms)
        clipped = sum(1 for e in self._history if e["was_clipped"] > 0.0)

        return {
            "mean_grad_norm": float(sum(norms) / n),
            "max_grad_norm": float(max(norms)),
            "clip_frequency": float(clipped / n),
            "n_steps": float(n),
        }

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Gradient issue detection
# ---------------------------------------------------------------------------


def detect_gradient_issues(
    parameters: Union[nn.Module, Iterable[torch.Tensor]],
) -> Dict[str, bool]:
    """Detect common gradient problems.

    Parameters
    ----------
    parameters:
        A model or an iterable of parameter tensors.

    Returns
    -------
    dict with keys:
        * ``has_nan``       — any parameter gradient contains NaN
        * ``has_inf``       — any parameter gradient contains Inf
        * ``has_zero_grad`` — at least one parameter has no gradient (``None``)
        * ``any_issue``     — logical OR of the above
    """
    has_nan = False
    has_inf = False
    has_zero_grad = False

    for p in _iter_params(parameters):
        if p.grad is None:
            has_zero_grad = True
        else:
            g = p.grad.detach()
            if torch.isnan(g).any():
                has_nan = True
            if torch.isinf(g).any():
                has_inf = True

    any_issue = has_nan or has_inf or has_zero_grad
    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "has_zero_grad": has_zero_grad,
        "any_issue": any_issue,
    }


# ---------------------------------------------------------------------------
# Per-layer gradient norms
# ---------------------------------------------------------------------------


def compute_layer_grad_norms(model: nn.Module) -> Dict[str, float]:
    """Return per-parameter gradient L2 norms for all params with gradients.

    Parameters
    ----------
    model:
        The PyTorch module to inspect.

    Returns
    -------
    dict
        ``{parameter_name: grad_norm}`` for every named parameter that has a
        gradient.
    """
    result: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            result[name] = float(p.grad.detach().float().norm(2).item())
    return result


# ---------------------------------------------------------------------------
# AdaptiveGradClipper
# ---------------------------------------------------------------------------


class AdaptiveGradClipper:
    """Clip gradient norm based on a running percentile of historical norms.

    The clipper maintains a sliding window of observed norms and uses a
    configurable percentile of that window as the clipping threshold.

    Parameters
    ----------
    percentile:
        Percentile (0–100) of the norm window to use as the threshold.
    window:
        Maximum number of historical norms to keep.
    """

    def __init__(self, percentile: float = 95.0, window: int = 100) -> None:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.percentile = percentile
        self.window = window
        self._norms: deque[float] = deque(maxlen=window)

    def update(self, norm: float) -> None:
        """Record a new observed gradient norm.

        Parameters
        ----------
        norm:
            The gradient norm value to add to the history.
        """
        self._norms.append(float(norm))

    def get_clip_threshold(self) -> float:
        """Return the current percentile threshold.

        Returns 0.0 when no norms have been recorded yet.
        """
        if not self._norms:
            return 0.0

        sorted_norms = sorted(self._norms)
        n = len(sorted_norms)
        # Linear interpolation percentile (same convention as numpy)
        idx = (self.percentile / 100.0) * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(sorted_norms[lo])
        frac = idx - lo
        return float(sorted_norms[lo] * (1.0 - frac) + sorted_norms[hi] * frac)

    def clip(
        self,
        parameters: Union[nn.Module, Iterable[torch.Tensor]],
    ) -> float:
        """Compute gradient norm, clip to percentile threshold, return pre-clip norm.

        If no history is available (threshold == 0), no clipping is applied.

        Parameters
        ----------
        parameters:
            A model or an iterable of parameter tensors.

        Returns
        -------
        float
            The gradient norm *before* clipping.
        """
        params = list(_iter_params(parameters))
        pre_clip = compute_grad_norm(params)
        threshold = self.get_clip_threshold()
        if threshold > 0.0:
            clip_grad_norm(params, max_norm=threshold)
        return pre_clip
