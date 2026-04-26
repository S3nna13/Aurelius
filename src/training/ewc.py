"""Elastic Weight Consolidation (EWC) for continual learning.

Kirkpatrick et al., 2017 - arXiv:1612.00796

Prevents catastrophic forgetting by anchoring important parameters
(measured via Fisher information) to their values after learning a
previous task.

Penalty: lambda/2 * sum_i F_i * (theta_i - theta*_i)^2
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class EWCConfig:
    ewc_lambda: float = 1000.0  # regularization strength
    n_fisher_samples: int = 200  # samples to estimate Fisher
    fisher_type: str = "empirical"  # "empirical" | "diagonal"


def compute_fisher_diagonal(
    model: nn.Module,
    data_iter: Iterator[Tensor],  # yields input_ids batches
    n_samples: int,
) -> dict[str, Tensor]:
    """Estimate diagonal Fisher Information Matrix.

    For each batch (up to n_samples total batches):
        1. Forward pass, compute log probability (negative cross-entropy
           treating input_ids as both input and shifted labels).
        2. Backward to get gradients.
        3. Fisher[param] += grad^2 (element-wise).

    Returns dict mapping param name -> Fisher diagonal (same shape as param),
    averaged over sampled batches.
    """
    model.eval()

    # Initialise accumulators
    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(param.data)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    n_batches = 0
    for batch in data_iter:
        if n_batches >= n_samples:
            break

        # Support both dict batches and plain tensors
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        else:
            input_ids = batch

        model.zero_grad()

        # Forward pass: unpack (loss, logits, pkv) — AureliusTransformer API
        loss, logits, _ = model(input_ids)

        if loss is None or not loss.requires_grad:
            # Compute cross-entropy manually
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            )

        loss.backward()

        # Accumulate squared gradients (Fisher diagonal estimate)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)

        n_batches += 1

    # Average over batches and detach from computation graph
    n_batches = max(1, n_batches)
    fisher = {name: (f / n_batches).detach() for name, f in fisher.items()}

    model.train()
    logger.info("Computed Fisher diagonal over %d batches", n_batches)
    return fisher


def ewc_penalty(
    model: nn.Module,
    fisher: dict[str, Tensor],
    optimal_params: dict[str, Tensor],
    ewc_lambda: float,
) -> Tensor:
    """Compute EWC regularization penalty.

    penalty = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    Returns scalar tensor.
    """
    device = next(model.parameters()).device
    penalty = torch.tensor(0.0, device=device)

    for name, param in model.named_parameters():
        if name not in fisher:
            continue
        f = fisher[name].to(device)
        theta_star = optimal_params[name].to(device)
        penalty = penalty + (f * (param - theta_star).pow(2)).sum()

    return ewc_lambda / 2.0 * penalty


class EWCTrainer:
    """Continual learning trainer using EWC regularization."""

    def __init__(
        self,
        model: nn.Module,
        config: EWCConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self._fisher: dict[str, Tensor] | None = None
        self._optimal_params: dict[str, Tensor] | None = None

    def consolidate(self, data_iter: Iterator[Tensor]) -> None:
        """After finishing task T, compute Fisher and save optimal params.

        Call this before training on task T+1.
        """
        self._fisher = compute_fisher_diagonal(
            self.model,
            data_iter,
            self.config.n_fisher_samples,
        )
        self._optimal_params = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        logger.info("EWCTrainer: consolidated task with %d parameters", len(self._fisher))

    def train_step(self, input_ids: Tensor) -> dict:
        """Train step with EWC regularization (if consolidated).

        loss = task_loss + ewc_penalty  (if Fisher available)

        Returns dict with keys: 'task_loss', 'ewc_loss', 'total_loss'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Task loss: language modelling (next-token prediction)
        task_loss_tensor, logits, _ = self.model(input_ids, labels=input_ids)

        if task_loss_tensor is None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            task_loss_tensor = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            )

        # EWC regularisation
        if self._fisher is not None and self._optimal_params is not None:
            ewc_loss_tensor = ewc_penalty(
                self.model,
                self._fisher,
                self._optimal_params,
                self.config.ewc_lambda,
            )
        else:
            device = next(self.model.parameters()).device
            ewc_loss_tensor = torch.tensor(0.0, device=device)

        total_loss = task_loss_tensor + ewc_loss_tensor
        total_loss.backward()
        self.optimizer.step()

        return {
            "task_loss": task_loss_tensor.item(),
            "ewc_loss": ewc_loss_tensor.item(),
            "total_loss": total_loss.item(),
        }

    def is_consolidated(self) -> bool:
        """True if Fisher has been computed (consolidate() was called)."""
        return self._fisher is not None


class TaskSequence:
    """Manages a sequence of tasks for continual learning evaluation."""

    def __init__(self) -> None:
        self._tasks: list[dict] = []

    def add_task(self, name: str, data: list[Tensor]) -> None:
        """Register a named task with its training data."""
        self._tasks.append({"name": name, "data": data})

    def get_task(self, name: str) -> list[Tensor]:
        """Retrieve task data by name.

        Raises:
            KeyError: If no task with the given name exists.
        """
        for task in self._tasks:
            if task["name"] == name:
                return task["data"]
        raise KeyError(f"Task '{name}' not found")

    def task_names(self) -> list[str]:
        """Return ordered list of registered task names."""
        return [t["name"] for t in self._tasks]

    def __len__(self) -> int:
        return len(self._tasks)


# Backward-compatible alias
class EWC:
    """Legacy EWC interface (model, config) compatible with continual.py."""

    def __init__(self, model: nn.Module, config: EWCConfig) -> None:
        self.model = model
        self.config = config
        self._fisher: dict[str, Tensor] | None = None
        self._optimal_params: dict[str, Tensor] | None = None

    def compute_fisher(self, data_iter) -> None:
        self._fisher = compute_fisher_diagonal(
            self.model, iter(data_iter), self.config.n_fisher_samples
        )
        self._optimal_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}

    def penalty(self, model: nn.Module) -> Tensor:
        if self._fisher is None or self._optimal_params is None:
            return torch.tensor(0.0)
        return ewc_penalty(model, self._fisher, self._optimal_params, self.config.ewc_lambda)
