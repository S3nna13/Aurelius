"""
Continual learning utilities for preventing catastrophic forgetting.

Supports: EWC (Elastic Weight Consolidation), Experience Replay,
Knowledge Distillation, and a no-regularization baseline.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CLConfig:
    """Configuration for continual learning."""

    ewc_lambda: float = 5000.0
    replay_buffer_size: int = 1000
    n_replay_per_step: int = 16
    distill_alpha: float = 0.5
    method: str = "ewc"  # "ewc" | "replay" | "distill" | "none"


# ---------------------------------------------------------------------------
# Fisher Information
# ---------------------------------------------------------------------------


def compute_fisher_information(
    model: nn.Module,
    dataloader: Iterable,
    loss_fn: Callable,
    n_samples: int = 100,
) -> dict[str, Tensor]:
    """
    Estimate diagonal Fisher information via accumulated squared gradients.

    For each batch/sample: compute loss, backward, accumulate grad^2.
    Returns a dict mapping param_name -> fisher tensor (same shape as param),
    normalised by n_samples.
    """
    model.eval()

    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    seen = 0
    for batch in dataloader:
        if seen >= n_samples:
            break

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = None

        model.zero_grad()

        if y is not None:
            loss = loss_fn(x, y)
        else:
            loss = loss_fn(x)

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2

        seen += 1

    for name in fisher:
        fisher[name] /= max(seen, 1)

    return fisher


# ---------------------------------------------------------------------------
# EWC Regularizer
# ---------------------------------------------------------------------------


class EWCRegularizer:
    """
    Elastic Weight Consolidation penalty.

    penalty = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
    """

    def __init__(
        self,
        model: nn.Module,
        fisher: dict[str, Tensor],
        ref_params: dict[str, Tensor],
        lam: float = 5000.0,
    ) -> None:
        self.fisher = fisher
        self.ref_params = ref_params
        self.lam = lam

    def penalty(self, model: nn.Module) -> Tensor:
        """Return scalar EWC penalty for the current model parameters."""
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)

        for name, param in model.named_parameters():
            if name in self.fisher and name in self.ref_params:
                f = self.fisher[name].to(device)
                ref = self.ref_params[name].to(device)
                loss = loss + (f * (param - ref) ** 2).sum()

        return (self.lam / 2.0) * loss


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------


class ExperienceReplayBuffer:
    """FIFO experience replay buffer storing (x, y) pairs."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._xs: list[Tensor] = []
        self._ys: list[Tensor] = []

    def add(self, x: Tensor, y: Tensor) -> None:
        """Add samples; evict oldest when buffer is full (FIFO)."""
        batch_size = x.shape[0]
        for i in range(batch_size):
            if len(self._xs) >= self.max_size:
                self._xs.pop(0)
                self._ys.pop(0)
            self._xs.append(x[i].detach().cpu())
            self._ys.append(y[i].detach().cpu())

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        """Return up to n random (x, y) pairs stacked into tensors."""
        available = len(self._xs)
        k = min(n, available)
        indices = random.sample(range(available), k)
        xs = torch.stack([self._xs[i] for i in indices])
        ys = torch.stack([self._ys[i] for i in indices])
        return xs, ys

    def __len__(self) -> int:
        return len(self._xs)


# ---------------------------------------------------------------------------
# Clone Model
# ---------------------------------------------------------------------------


def clone_model(model: nn.Module) -> nn.Module:
    """Return a deep copy of model with the same parameter values."""
    return copy.deepcopy(model)


# ---------------------------------------------------------------------------
# Distillation Loss
# ---------------------------------------------------------------------------


def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    """
    KL divergence loss for knowledge distillation.

    KL(teacher || student) at the given temperature, scaled by T^2.
    Returns a scalar tensor.
    """
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (T**2)


# ---------------------------------------------------------------------------
# Continual Trainer
# ---------------------------------------------------------------------------


class ContinualTrainer:
    """
    Wraps a model with continual-learning logic.

    Supports methods: "ewc", "replay", "distill", "none".
    """

    def __init__(self, model: nn.Module, config: CLConfig) -> None:
        self.model = model
        self.config = config

        self._ref_params: dict[str, Tensor] | None = None
        self._fisher: dict[str, Tensor] | None = None
        self._ewc: EWCRegularizer | None = None
        self._teacher: nn.Module | None = None

        self._replay_buffer = ExperienceReplayBuffer(config.replay_buffer_size)

    def register_task(
        self,
        task_id: int,
        fisher: dict[str, Tensor] | None = None,
    ) -> None:
        """
        Snapshot current parameters as the reference for the next task.
        Optionally accepts pre-computed Fisher information.
        """
        self._ref_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self._fisher = fisher

        if fisher is not None:
            self._ewc = EWCRegularizer(
                self.model,
                fisher=fisher,
                ref_params=self._ref_params,
                lam=self.config.ewc_lambda,
            )
        else:
            self._ewc = None

        self._teacher = clone_model(self.model)
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        self._teacher.eval()

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        loss_fn: Callable,
    ) -> dict[str, float]:
        """
        Compute task loss + optional regularisation.

        Returns dict with keys: "loss", "task_loss", "reg_loss".
        """
        self.model.train()

        task_loss = loss_fn(x, y)

        reg_loss = torch.tensor(0.0, device=task_loss.device)

        method = self.config.method

        if method == "ewc" and self._ewc is not None:
            reg_loss = self._ewc.penalty(self.model)

        elif method == "replay" and len(self._replay_buffer) > 0:
            rx, ry = self._replay_buffer.sample(self.config.n_replay_per_step)
            rx = rx.to(x.device)
            ry = ry.to(x.device)
            reg_loss = loss_fn(rx, ry)

        elif method == "distill" and self._teacher is not None:
            with torch.no_grad():
                teacher_logits = self._teacher(x)
            student_logits = self.model(x)
            reg_loss = (
                compute_distillation_loss(student_logits, teacher_logits)
                * self.config.distill_alpha
            )

        total_loss = task_loss + reg_loss

        self._replay_buffer.add(x, y)

        return {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "reg_loss": reg_loss.item(),
        }

    def get_forgetting(
        self,
        model: nn.Module,
        ref_params: dict[str, Tensor],
    ) -> float:
        """Mean L2 distance between current and reference parameters."""
        total = 0.0
        count = 0
        for name, param in model.named_parameters():
            if name in ref_params:
                diff = (param.detach() - ref_params[name].to(param.device)).norm().item()
                total += diff
                count += 1
        return total / max(count, 1)
