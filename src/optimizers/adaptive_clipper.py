"""Adaptive gradient clipping for stable training."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class AdaptiveGradientClipper:
    """Clip gradients based on running statistics of gradient norm.

    Adapts the clip threshold based on the observed gradient norm history,
    growing the threshold when norms are consistently high.
    """

    initial_threshold: float = 1.0
    growth_factor: float = 1.05
    decay_factor: float = 0.95
    window_size: int = 100
    _norm_history: list[float] | None = None

    def __post_init__(self) -> None:
        self._norm_history = []

    def clip(self, total_norm: float) -> float:
        if self._norm_history is None:
            self._norm_history = []
        self._norm_history.append(total_norm)
        if len(self._norm_history) > self.window_size:
            self._norm_history.pop(0)

        avg_norm = sum(self._norm_history) / len(self._norm_history)
        threshold = self.initial_threshold
        ratio = avg_norm / max(threshold, 1e-8)
        if ratio > 1.5:
            threshold *= self.growth_factor
        elif ratio < 0.5:
            threshold *= self.decay_factor
        return threshold


class ZClip:
    """Adaptive gradient clipping via EMA z-score anomaly detection.

    Maintains EMA of gradient norms and EMA of squared norms to compute
    running mean and standard deviation. Clips only when the current norm
    is a statistical outlier (z-score > z_threshold).

    Reference: ZClip (2025)
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        beta: float | None = None,
        ema_alpha: float | None = None,
        min_std: float = 1e-8,
        fallback_clip: float | None = None,
        warmup_steps: int = 0,
    ) -> None:
        if beta is not None and ema_alpha is not None:
            raise ValueError("Only one of beta or ema_alpha should be provided")
        if beta is None and ema_alpha is None:
            beta = 0.99
        elif ema_alpha is not None:
            beta = 1.0 - ema_alpha

        self.z_threshold = z_threshold
        self.beta = beta
        self.min_std = min_std
        self.fallback_clip = fallback_clip
        self.warmup_steps = warmup_steps

        self._ema: float = 0.0
        self._ema_sq: float = 0.0
        self._step: int = 0

    def _z_score(self, norm: float) -> tuple[float, float, float]:
        """Compute z-score using current EMA state (before update).

        Returns (mean, std, z_score).
        """
        if self._step == 0:
            mean = norm
            std = 0.0
        else:
            mean = self._ema
            var = self._ema_sq - mean * mean
            std = math.sqrt(max(var, 0.0))
        std = max(std, self.min_std)
        z = (norm - mean) / std
        return mean, std, z

    def _update(self, norm: float) -> None:
        """Update EMAs with the given norm."""
        if self._step == 0:
            self._ema = norm
            self._ema_sq = norm * norm
        else:
            self._ema = self.beta * self._ema + (1.0 - self.beta) * norm
            self._ema_sq = self.beta * self._ema_sq + (1.0 - self.beta) * (norm * norm)
        self._step += 1

    def clip(self, total_norm: float) -> float:
        """Return the (possibly clipped) norm.

        If the norm is a z-score outlier, returns the clipped value.
        Otherwise returns the original norm.
        """
        if not math.isfinite(total_norm):
            return total_norm

        if self._step < self.warmup_steps and self.fallback_clip is not None:
            clipped = min(total_norm, self.fallback_clip)
            self._update(clipped)
            return clipped

        mean, std, z = self._z_score(total_norm)
        if z > self.z_threshold:
            clipped = mean + self.z_threshold * std
        else:
            clipped = total_norm

        self._update(clipped)
        return clipped

    def clip_grad_norm_(self, parameters: Iterable[torch.nn.Parameter]) -> float:
        """Compute gradient norm and adaptively clip in-place.

        Args:
            parameters: Iterable of parameters whose gradients to clip.

        Returns:
            The pre-clip gradient norm.
        """
        params_with_grad = [p for p in parameters if p.grad is not None]
        if not params_with_grad:
            return 0.0

        total_norm_sq = sum(p.grad.detach().float().norm() ** 2 for p in params_with_grad)
        norm = float(total_norm_sq**0.5)

        if not math.isfinite(norm):
            return norm

        if self._step < self.warmup_steps and self.fallback_clip is not None:
            if norm > self.fallback_clip:
                clip_coef = self.fallback_clip / max(norm, 1e-6)
                for p in params_with_grad:
                    p.grad.detach().mul_(clip_coef)
            self._update(min(norm, self.fallback_clip))
            return norm

        mean, std, z = self._z_score(norm)
        if z > self.z_threshold:
            clip_value = mean + self.z_threshold * std
            clip_coef = clip_value / max(norm, 1e-6)
            for p in params_with_grad:
                p.grad.detach().mul_(clip_coef)

        self._update(min(norm, mean + self.z_threshold * std) if z > self.z_threshold else norm)
        return norm


class AdaGC:
    """Per-tensor adaptive gradient clipping using z-scores.

    Computes z-score per parameter tensor rather than globally.
    Each parameter's gradient norm is compared against its own EMA
    history, and clipped independently if it's an outlier.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        beta: float = 0.99,
        min_std: float = 1e-8,
    ) -> None:
        self.z_threshold = z_threshold
        self.beta = beta
        self.min_std = min_std

        # Per-tensor state: maps id(tensor) -> (ema, ema_sq, step)
        self._state: dict[int, tuple[float, float, int]] = {}

    def _z_score(self, tensor_id: int, norm: float) -> tuple[float, float, float]:
        """Compute z-score using current per-tensor EMA state."""
        if tensor_id not in self._state:
            mean = norm
            std = 0.0
        else:
            ema, ema_sq, _step = self._state[tensor_id]
            mean = ema
            var = ema_sq - mean * mean
            std = math.sqrt(max(var, 0.0))
        std = max(std, self.min_std)
        z = (norm - mean) / std
        return mean, std, z

    def _update(self, tensor_id: int, norm: float) -> None:
        """Update per-tensor EMAs with the given norm."""
        if tensor_id not in self._state:
            self._state[tensor_id] = (norm, norm * norm, 1)
        else:
            ema, ema_sq, step = self._state[tensor_id]
            ema = self.beta * ema + (1.0 - self.beta) * norm
            ema_sq = self.beta * ema_sq + (1.0 - self.beta) * (norm * norm)
            step += 1
            self._state[tensor_id] = (ema, ema_sq, step)

    def clip_grads_(self, parameters: Iterable[torch.nn.Parameter]) -> dict[int, float]:
        """Clip gradients per tensor and return dict of pre-clip norms.

        Args:
            parameters: Iterable of parameters whose gradients to clip.

        Returns:
            Mapping from parameter id to pre-clip gradient norm.
        """
        params_with_grad = [p for p in parameters if p.grad is not None]
        norms: dict[int, float] = {}

        for p in params_with_grad:
            tid = id(p)
            norm = float(p.grad.detach().float().norm())
            norms[tid] = norm

            if not math.isfinite(norm):
                continue

            mean, std, z = self._z_score(tid, norm)
            if z > self.z_threshold:
                clip_value = mean + self.z_threshold * std
                clip_coef = clip_value / max(norm, 1e-6)
                p.grad.detach().mul_(clip_coef)
                self._update(tid, clip_value)
            else:
                self._update(tid, norm)

        return norms


ADAPTIVE_CLIPPER = AdaptiveGradientClipper()
