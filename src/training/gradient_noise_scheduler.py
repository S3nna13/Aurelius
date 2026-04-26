"""Gradient noise scheduler with configurable decay and noise distributions."""

from __future__ import annotations

import threading

import torch


class GradientNoiseScheduler:
    """Add scaled noise to gradients with a decaying scale schedule.

    Supports Gaussian and Laplace noise, combined with exponential, linear,
    or constant decay.

    Args:
        noise_type: Distribution of the injected noise. One of ``"gaussian"``
            or ``"laplace"``.
        initial_scale: Magnitude of the noise at step 0. Must be > 0.
        decay: Decay schedule for the noise scale. One of ``"exponential"``,
            ``"linear"``, or ``"constant"``.
        decay_rate: Rate controlling the speed of decay. Interpretation depends
            on *decay* (e.g. the base for exponential decay).
    """

    _VALID_NOISE_TYPES = {"gaussian", "laplace"}
    _VALID_DECAYS = {"exponential", "linear", "constant"}

    def __init__(
        self,
        noise_type: str = "gaussian",
        initial_scale: float = 0.01,
        decay: str = "exponential",
        decay_rate: float = 0.99,
    ) -> None:
        if noise_type not in self._VALID_NOISE_TYPES:
            raise ValueError(
                f"Unsupported noise_type '{noise_type}'. Must be one of {self._VALID_NOISE_TYPES}."
            )
        if decay not in self._VALID_DECAYS:
            raise ValueError(f"Unsupported decay '{decay}'. Must be one of {self._VALID_DECAYS}.")
        if initial_scale <= 0:
            raise ValueError(f"initial_scale must be > 0, got {initial_scale}.")

        self.noise_type = noise_type
        self.initial_scale = initial_scale
        self.decay = decay
        self.decay_rate = decay_rate
        self._lock = threading.Lock()

    def get_scale(self, step: int) -> float:
        """Return the noise scale for a given training step.

        Args:
            step: Current training step (non-negative integer).

        Returns:
            The noise scale after applying the configured decay.
        """
        if self.decay == "exponential":
            return self.initial_scale * (self.decay_rate**step)
        if self.decay == "linear":
            return self.initial_scale * max(0.0, 1.0 - step * self.decay_rate)
        # constant
        return self.initial_scale

    def add_noise(
        self,
        gradients: list[torch.Tensor],
        step: int,
    ) -> list[torch.Tensor]:
        """Add scaled noise to each gradient tensor in-place.

        Args:
            gradients: List of gradient tensors.
            step: Current training step used to compute the noise scale.

        Returns:
            The list of noisy gradient tensors (mutated in-place).
        """
        with self._lock:
            scale = self.get_scale(step)
            for grad in gradients:
                if self.noise_type == "gaussian":
                    noise = torch.randn_like(grad) * scale
                else:  # laplace
                    u = torch.rand_like(grad) - 0.5
                    noise = -scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
                grad.add_(noise)
        return gradients


NOISE_SCHEDULER_REGISTRY: dict[str, GradientNoiseScheduler] = {}
