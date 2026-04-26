"""Activation and gradient norm tracking for diagnosing training health.

Tracks running statistics of activations and gradients layer-by-layer,
enabling early detection of vanishing/exploding gradients and dead layers.

Implements:
  - TrackingConfig     — configuration dataclass
  - RunningStats       — exponential moving average tracker
  - compute_tensor_stats — per-tensor diagnostic summary
  - LayerStatsHook     — forward/backward hook wrapper
  - ActivationTracker  — model-level hook manager
  - compute_gradient_norm_per_layer — per-parameter gradient norms
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrackingConfig:
    """Configuration for activation and gradient tracking."""

    track_activations: bool = True
    track_gradients: bool = True
    ema_decay: float = 0.99
    log_interval: int = 100
    percentiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


# ---------------------------------------------------------------------------
# Running statistics (EMA)
# ---------------------------------------------------------------------------


class RunningStats:
    """Exponential moving average tracker for mean and variance.

    Uses Welford-style online EMA update:
        mean_new  = decay * mean + (1 - decay) * value
        var_new   = decay * var  + (1 - decay) * (value - mean_new)^2
    """

    def __init__(self, decay: float = 0.99) -> None:
        self._decay = decay
        self._mean: float = 0.0
        self._var: float = 0.0
        self._initialised: bool = False

    # ------------------------------------------------------------------
    def update(self, value: float) -> None:
        """Update EMA mean and variance with a new scalar observation."""
        if not self._initialised:
            self._mean = value
            self._var = 0.0
            self._initialised = True
            return
        d = self._decay
        diff = value - self._mean
        self._mean = d * self._mean + (1.0 - d) * value
        self._var = d * self._var + (1.0 - d) * diff * diff

    def mean(self) -> float:
        """Return current EMA mean."""
        return self._mean

    def std(self) -> float:
        """Return sqrt of current EMA variance (always >= 0)."""
        return math.sqrt(max(self._var, 0.0))

    def reset(self) -> None:
        """Reset tracker to its initial state."""
        self._mean = 0.0
        self._var = 0.0
        self._initialised = False


# ---------------------------------------------------------------------------
# Tensor statistics
# ---------------------------------------------------------------------------


def compute_tensor_stats(x: Tensor) -> dict:
    """Return a diagnostic summary dict for tensor *x*.

    Keys
    ----
    mean         : float  — arithmetic mean of all elements
    std          : float  — standard deviation of all elements
    norm         : float  — L2 norm
    max_abs      : float  — maximum absolute value
    fraction_zero: float  — fraction of elements exactly equal to 0
    fraction_nan : float  — fraction of elements that are NaN
    """
    # Work with a float32 view to keep computations stable
    x_f = x.detach().float()
    numel = x_f.numel()

    if numel == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "norm": 0.0,
            "max_abs": 0.0,
            "fraction_zero": 0.0,
            "fraction_nan": 0.0,
        }

    nan_mask = torch.isnan(x_f)
    fraction_nan = nan_mask.float().mean().item()

    # Replace NaNs with 0 for numeric computations
    x_clean = x_f.clone()
    x_clean[nan_mask] = 0.0

    mean = x_clean.mean().item()
    std = x_clean.std(unbiased=False).item() if numel > 1 else 0.0
    norm = x_clean.norm(p=2).item()
    max_abs = x_clean.abs().max().item()
    fraction_zero = (x_clean == 0.0).float().mean().item()

    return {
        "mean": mean,
        "std": std,
        "norm": norm,
        "max_abs": max_abs,
        "fraction_zero": fraction_zero,
        "fraction_nan": fraction_nan,
    }


# ---------------------------------------------------------------------------
# Per-layer hook wrapper
# ---------------------------------------------------------------------------


class LayerStatsHook:
    """Attach forward (and optionally backward) hooks to a single module."""

    def __init__(self, name: str, config: TrackingConfig) -> None:
        self._name = name
        self._config = config
        self._activation_stats: dict | None = None
        self._gradient_stats: dict | None = None
        self._fwd_handle = None
        self._bwd_handle = None

    # ------------------------------------------------------------------
    def register(self, module: nn.Module) -> None:
        """Register hooks on *module*."""
        if self._config.track_activations:
            self._fwd_handle = module.register_forward_hook(self._forward_hook)
        if self._config.track_gradients:
            self._bwd_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs, output) -> None:
        # output may be a tensor or a tuple; grab the first tensor
        if isinstance(output, Tensor):
            tensor = output
        elif isinstance(output, (tuple, list)):
            tensor = next((t for t in output if isinstance(t, Tensor)), None)
        else:
            tensor = None

        if tensor is not None:
            self._activation_stats = compute_tensor_stats(tensor)

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
        # grad_output contains gradients w.r.t. the module's output
        tensor = None
        if isinstance(grad_output, (tuple, list)):
            tensor = next((g for g in grad_output if isinstance(g, Tensor)), None)
        elif isinstance(grad_output, Tensor):
            tensor = grad_output

        if tensor is not None:
            self._gradient_stats = compute_tensor_stats(tensor)

    # ------------------------------------------------------------------
    def remove(self) -> None:
        """Remove all registered hooks."""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return latest captured statistics.

        Returns
        -------
        dict with keys:
            "activation": dict | None
            "gradient"  : dict | None
        """
        return {
            "activation": self._activation_stats,
            "gradient": self._gradient_stats,
        }


# ---------------------------------------------------------------------------
# Model-level tracker
# ---------------------------------------------------------------------------


class ActivationTracker:
    """Attach and manage :class:`LayerStatsHook` instances across a model."""

    def __init__(self, model: nn.Module, config: TrackingConfig) -> None:
        self._model = model
        self._config = config
        self._hooks: dict[str, LayerStatsHook] = {}

    # ------------------------------------------------------------------
    def attach(self, layer_names: list[str]) -> None:
        """Attach hooks to named modules whose names are in *layer_names*.

        Silently skips names that do not match any module.
        """
        named = {name: module for name, module in self._model.named_modules()}
        for name in layer_names:
            if name in named and name not in self._hooks:
                hook = LayerStatsHook(name, self._config)
                hook.register(named[name])
                self._hooks[name] = hook

    def detach(self) -> None:
        """Remove all hooks from tracked modules."""
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    def get_all_stats(self) -> dict[str, dict]:
        """Return {layer_name: stats_dict} for every tracked layer."""
        return {name: hook.get_stats() for name, hook in self._hooks.items()}

    # ------------------------------------------------------------------
    def detect_anomalies(self) -> list[str]:
        """Return names of layers with anomalous activation statistics.

        Anomaly criteria (activation stats):
          - norm > 100       (exploding activations)
          - fraction_nan > 0 (NaN propagation)
          - std == 0         (dead / constant layer), only when mean != 0
            so pure-zero layers are also flagged via norm == 0 only if
            we explicitly check std == 0 regardless of mean.
        """
        anomalous: list[str] = []
        for name, hook in self._hooks.items():
            stats = hook.get_stats()
            act = stats.get("activation")
            if act is None:
                continue
            if act["norm"] > 100:
                anomalous.append(name)
            elif act["fraction_nan"] > 0:
                anomalous.append(name)
            elif act["std"] == 0.0:
                anomalous.append(name)
        return anomalous


# ---------------------------------------------------------------------------
# Gradient norm per parameter
# ---------------------------------------------------------------------------


def compute_gradient_norm_per_layer(model: nn.Module) -> dict[str, float]:
    """Return {param_name: grad_l2_norm} for all parameters with gradients."""
    result: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            result[name] = param.grad.detach().float().norm(p=2).item()
    return result
