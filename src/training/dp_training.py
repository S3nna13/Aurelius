"""Differentially Private SGD (DP-SGD) for privacy-preserving LLM fine-tuning.

Implements per-sample gradient clipping and calibrated Gaussian noise addition
following the Abadi et al. (2016) DP-SGD algorithm with RDP accounting.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DPConfig:
    """Configuration for differentially private training."""

    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5
    target_epsilon: float = 8.0
    accounting_method: str = "rdp"


class PrivacyAccountant:
    """Tracks privacy budget consumption using Rényi Differential Privacy (RDP)."""

    def __init__(self, config: DPConfig, dataset_size: int, batch_size: int) -> None:
        self.config = config
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.sampling_rate = self._compute_sampling_rate()

    def _compute_sampling_rate(self) -> float:
        """Return the batch sampling rate q = batch_size / dataset_size."""
        return self.batch_size / self.dataset_size

    def compute_rdp(self, steps: int, alpha: float = 2.0) -> float:
        """Compute RDP divergence accumulated over `steps` training steps.

        Uses the Gaussian mechanism with subsampling approximation:
            RDP ≈ steps * q² * α / (2σ²)
        where q is the sampling rate and σ is the noise multiplier.
        """
        return steps * (self.sampling_rate**2) * alpha / (2.0 * self.config.noise_multiplier**2)

    def compute_epsilon(self, steps: int) -> float:
        """Convert accumulated RDP to (ε, δ)-DP via:
            ε = rdp + log(1/δ) / (α - 1)
        Uses order α=2. Clamps to 1000.0 to avoid overflow.
        At steps=0, returns 0.0 (no privacy budget consumed).
        """
        if steps == 0:
            return 0.0
        alpha = 2.0
        rdp = self.compute_rdp(steps, alpha=alpha)
        epsilon = rdp + math.log(1.0 / self.config.delta) / (alpha - 1.0)
        return min(epsilon, 1000.0)

    def is_budget_exceeded(self, steps: int) -> bool:
        """Return True if the privacy budget (target_epsilon) has been exceeded."""
        return self.compute_epsilon(steps) > self.config.target_epsilon


def clip_gradients(parameters: Iterable, max_grad_norm: float) -> float:
    """Clip each parameter's gradient by norm and return the average gradient norm before clipping.

    Args:
        parameters: Iterable of nn.Parameter objects with populated .grad attributes.
        max_grad_norm: Maximum allowed gradient norm per parameter.

    Returns:
        Average gradient norm across all parameters before clipping.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return 0.0

    norms = []
    for p in params_with_grad:
        grad_norm = p.grad.data.norm(2).item()
        norms.append(grad_norm)
        # Clip in-place
        nn.utils.clip_grad_norm_([p], max_grad_norm)

    return sum(norms) / len(norms)


def add_dp_noise(
    parameters: Iterable,
    noise_multiplier: float,
    max_grad_norm: float,
    batch_size: int,
) -> None:
    """Add calibrated Gaussian noise to each parameter's gradient.

    Noise std is: noise_std = noise_multiplier * max_grad_norm / batch_size

    Args:
        parameters: Iterable of nn.Parameter objects with populated .grad attributes.
        noise_multiplier: Ratio of noise std to sensitivity (max_grad_norm).
        max_grad_norm: Clipping norm (sensitivity).
        batch_size: Number of samples in the batch (used for normalization).
    """
    noise_std = noise_multiplier * max_grad_norm / batch_size
    for p in parameters:
        if p.grad is not None:
            p.grad.data.add_(torch.randn_like(p.grad.data) * noise_std)


class DPTrainer:
    """Trainer that applies DP-SGD: per-sample gradient clipping + Gaussian noise."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: DPConfig,
        dataset_size: int,
        batch_size: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.batch_size = batch_size
        self.accountant = PrivacyAccountant(config, dataset_size, batch_size)
        self._step_count: int = 0

    def train_step(self, input_ids: Tensor) -> dict:
        """Perform one DP-SGD training step.

        1. Forward pass to get loss (uses plain tuple API: loss, logits, pkv = model(input_ids))
        2. Backward pass to populate gradients
        3. Clip gradients per-sample style
        4. Add calibrated Gaussian noise
        5. Optimizer step

        Returns:
            dict with keys: loss, epsilon, grad_norm, steps, budget_exceeded
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward — model returns (loss, logits, past_key_values)
        # Use next-token prediction: labels are input_ids shifted left by 1
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        loss, logits, pkv = self.model(inputs, labels=labels)

        # Backward
        loss.backward()

        # Clip gradients and record norm before clipping
        grad_norm = clip_gradients(self.model.parameters(), self.config.max_grad_norm)

        # Add DP noise
        add_dp_noise(
            self.model.parameters(),
            self.config.noise_multiplier,
            self.config.max_grad_norm,
            self.batch_size,
        )

        self.optimizer.step()
        self._step_count += 1

        epsilon = self.accountant.compute_epsilon(self._step_count)
        budget_exceeded = self.accountant.is_budget_exceeded(self._step_count)

        return {
            "loss": loss.item(),
            "epsilon": epsilon,
            "grad_norm": grad_norm,
            "steps": self._step_count,
            "budget_exceeded": budget_exceeded,
        }

    def get_privacy_spent(self) -> tuple[float, float]:
        """Return (epsilon, delta) for the current number of steps taken."""
        epsilon = self.accountant.compute_epsilon(self._step_count)
        return (epsilon, self.config.delta)
