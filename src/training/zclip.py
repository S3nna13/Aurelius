"""ZClip — adaptive gradient clipping via EMA z-score anomaly detection.

Replaces fixed max_norm clipping. Tracks running mean/variance of gradient
norms via EMA and clips only when the current norm is a statistical outlier
(z-score > threshold). Falls back to fixed clipping on warmup steps.

Reference: "ZClip: Adaptive Spike Mitigation for LLM Pre-Training"
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


class ZClip:
    """Adaptive gradient clipping via EMA z-score anomaly detection.

    Tracks the running mean and variance of gradient norms using EMA.
    Clips only when the current norm is a statistical outlier (z-score >
    z_threshold). During warmup, falls back to fixed max_norm clipping.

    Args:
        params: Iterable of parameter tensors (same as optimizer params).
        z_threshold: How many std devs above mean triggers a clip (default 2.5).
        ema_alpha: EMA smoothing factor for mean and variance (default 0.01).
        min_warmup_steps: Steps before z-score is used; uses fallback_clip
            during warmup (default 100).
        fallback_clip: max_norm used during warmup (default 1.0).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        z_threshold: float = 2.5,
        ema_alpha: float = 0.01,
        min_warmup_steps: int = 100,
        fallback_clip: float = 1.0,
    ) -> None:
        self.params = list(params)
        self.z_threshold = z_threshold
        self.ema_alpha = ema_alpha
        self.min_warmup_steps = min_warmup_steps
        self.fallback_clip = fallback_clip

        # EMA state
        self._ema_mean: float = 0.0
        self._ema_var: float = 1.0
        self._step: int = 0

    def clip_grad_norm_(self, parameters: Iterable[torch.nn.Parameter]) -> float:
        """Compute gradient norm and adaptively clip if norm is an outlier.

        Args:
            parameters: Iterable of parameters whose gradients to clip.

        Returns:
            The pre-clip gradient norm (as a Python float).
        """
        params_with_grad = [p for p in parameters if p.grad is not None]

        if not params_with_grad:
            return 0.0

        # Compute total gradient norm (L2 across all parameters)
        total_norm_sq = sum(
            p.grad.detach().float().norm() ** 2
            for p in params_with_grad
        )
        norm = float(total_norm_sq ** 0.5)

        alpha = self.ema_alpha

        if self._step < self.min_warmup_steps:
            # Warmup: fixed clipping to fallback_clip
            if norm > self.fallback_clip:
                clip_coef = self.fallback_clip / max(norm, 1e-6)
                for p in params_with_grad:
                    p.grad.detach().mul_(clip_coef)
            clipped_norm = min(norm, self.fallback_clip)

            # Update EMA with the (possibly clipped) norm
            self._ema_var = (1 - alpha) * (self._ema_var + alpha * (norm - self._ema_mean) ** 2)
            self._ema_mean = (1 - alpha) * self._ema_mean + alpha * clipped_norm
        else:
            # Post-warmup: z-score based clipping
            std = math.sqrt(self._ema_var + 1e-8)
            z_score = (norm - self._ema_mean) / std

            if z_score > self.z_threshold:
                # Clip to mean + z_threshold * std
                clip_value = self._ema_mean + self.z_threshold * std
                clip_coef = clip_value / max(norm, 1e-6)
                for p in params_with_grad:
                    p.grad.detach().mul_(clip_coef)
                clipped_norm = clip_value
            else:
                clipped_norm = norm

            # Update EMA with the (possibly clipped) norm
            self._ema_var = (1 - alpha) * (self._ema_var + alpha * (norm - self._ema_mean) ** 2)
            self._ema_mean = (1 - alpha) * self._ema_mean + alpha * clipped_norm

        self._step += 1
        return norm
