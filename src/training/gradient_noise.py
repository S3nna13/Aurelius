"""Gradient Noise Injection for escaping sharp minima.

Neelakantan et al. 2015 — https://arxiv.org/abs/1511.06807

Adds annealed Gaussian noise to gradients at each step:
    σ_t² = eta / (1 + t)^gamma

This helps the optimizer escape saddle points and sharp local minima
during neural network training.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class GradientNoiseSchedule:
    """Annealed Gaussian noise schedule for gradient perturbation.

    Computes the noise variance at step ``t`` as:

        σ_t² = eta / (1 + t)^gamma

    Args:
        eta: Initial noise variance (σ² at t=0 equals eta).
        gamma: Decay exponent; Neelakantan et al. recommend 0.55.
    """

    def __init__(self, eta: float = 0.01, gamma: float = 0.55) -> None:
        self.eta = eta
        self.gamma = gamma

    def variance(self, step: int) -> float:
        """Return σ_t² = eta / (1 + step)^gamma."""
        return self.eta / (1 + step) ** self.gamma

    def std(self, step: int) -> float:
        """Return σ_t = sqrt(variance(step))."""
        return math.sqrt(self.variance(step))

    def noise_tensor(
        self,
        shape: tuple[int, ...],
        step: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample a Gaussian noise tensor scaled to σ_t.

        Args:
            shape: Desired tensor shape.
            step: Current training step used to compute σ_t.
            generator: Optional ``torch.Generator`` for reproducibility.

        Returns:
            Float32 tensor of the given shape with values ~ N(0, σ_t²).
        """
        return torch.randn(shape, generator=generator) * self.std(step)


class GradientNoiseOptimizer:
    """Optimizer wrapper that injects annealed Gaussian noise into gradients.

    Before each ``step()`` call the wrapper adds noise sampled from
    N(0, σ_t²) to every parameter gradient, where σ_t is determined by
    ``schedule``.

    Args:
        optimizer: Any ``torch.optim.Optimizer`` instance to wrap.
        schedule: A :class:`GradientNoiseSchedule` controlling noise magnitude.
        seed: Integer seed for the internal ``torch.Generator``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule: GradientNoiseSchedule,
        seed: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.schedule = schedule
        self._step_count: int = 0
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Inject gradient noise then delegate to the wrapped optimizer."""
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                noise = self.schedule.noise_tensor(
                    param.grad.shape,
                    self._step_count,
                    generator=self._generator,
                )
                param.grad.data.add_(noise)

        self.optimizer.step()
        self._step_count += 1

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero (or None-out) all parameter gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable state dict including the step counter."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "eta": self.schedule.eta,
            "gamma": self.schedule.gamma,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a previously saved dict."""
        self._step_count = state["step_count"]
        self.optimizer.load_state_dict(state["optimizer"])

    # ------------------------------------------------------------------
    # Pass-through properties
    # ------------------------------------------------------------------

    @property
    def param_groups(self):
        """Expose the wrapped optimizer's param_groups."""
        return self.optimizer.param_groups


class GradientNoiseCallback:
    """Utility to record and inspect noise magnitude over training.

    Args:
        None
    """

    def __init__(self) -> None:
        self.noise_std_history: list[float] = []

    def record(self, schedule: GradientNoiseSchedule, step: int) -> float:
        """Append the current noise std to history and return it.

        Args:
            schedule: The active :class:`GradientNoiseSchedule`.
            step: Current training step.

        Returns:
            The noise standard deviation at ``step``.
        """
        std = schedule.std(step)
        self.noise_std_history.append(std)
        return std

    def plot_schedule(self, n_steps: int, schedule: GradientNoiseSchedule) -> list[float]:
        """Return the noise std for each step in ``[0, n_steps)``.

        Note: This method returns a plain list and does NOT produce any
        matplotlib plots.

        Args:
            n_steps: Number of steps to evaluate.
            schedule: The :class:`GradientNoiseSchedule` to query.

        Returns:
            List of noise standard deviations, one per step.
        """
        return [schedule.std(t) for t in range(n_steps)]
