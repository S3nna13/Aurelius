"""Online learning: streaming updates with experience replay and catastrophic forgetting prevention."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class OnlineLearningConfig:
    """Configuration for online / streaming learner."""
    replay_buffer_size: int = 1000
    replay_batch_size: int = 16
    ewc_lambda: float = 1.0
    drift_detection_window: int = 50
    drift_threshold: float = 2.0   # z-score threshold for drift detection
    learning_rate: float = 1e-4


class ReplayBuffer:
    """FIFO experience replay buffer storing dicts with input_ids and labels."""

    def __init__(self, max_size: int) -> None:
        self._buffer: list[dict] = []
        self._max_size = max_size

    def add(self, item: dict) -> None:
        """Add item dict; evict oldest when full (FIFO)."""
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)
        self._buffer.append(item)

    def sample(self, n: int) -> list[dict]:
        """Random sample without replacement (with replacement if n > len)."""
        if len(self._buffer) == 0:
            return []
        if n > len(self._buffer):
            return random.choices(self._buffer, k=n)
        return random.sample(self._buffer, n)

    def __len__(self) -> int:
        return len(self._buffer)


def detect_concept_drift(
    loss_history: list[float],
    window: int,
    threshold: float,
) -> bool:
    """Detect concept drift by comparing mean loss in the two halves of a window.

    Args:
        loss_history: Recent loss values (most recent last).
        window: Number of steps to consider.
        threshold: z-score threshold above which drift is flagged.

    Returns:
        True if z_score = (mean2 - mean1) / (std1 + 1e-8) > threshold, else False.
        Returns False if we don't yet have `window` data points.
    """
    if len(loss_history) < window:
        return False

    recent = loss_history[-window:]
    half = window // 2
    first_half = recent[:half]
    second_half = recent[half:]

    mean1 = sum(first_half) / len(first_half)
    mean2 = sum(second_half) / len(second_half)

    variance1 = sum((x - mean1) ** 2 for x in first_half) / len(first_half)
    std1 = variance1 ** 0.5

    z_score = (mean2 - mean1) / (std1 + 1e-8)
    return z_score > threshold


def compute_fisher_diagonal(
    model: nn.Module,
    data_batch: list[dict],
    loss_fn: Callable,
) -> dict[str, Tensor]:
    """Compute diagonal Fisher information E[grad^2] for each parameter.

    Runs forward+backward for each sample in data_batch and accumulates
    squared gradients as an estimate of the Fisher diagonal.

    Args:
        model: The neural network module.
        data_batch: List of dicts with "input_ids" and "labels" keys.
        loss_fn: Callable(model, input_ids, labels) -> scalar loss tensor.

    Returns:
        Dict mapping param_name -> Fisher diagonal tensor (same shape as param).
    """
    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(param.data)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    model.eval()
    n_samples = 0

    for sample in data_batch:
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        model.zero_grad()
        loss = loss_fn(model, input_ids, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2)

        n_samples += 1

    # Normalize by number of samples
    if n_samples > 0:
        fisher = {name: f / n_samples for name, f in fisher.items()}

    model.train()
    return fisher


def ewc_penalty(
    model: nn.Module,
    fisher: dict[str, Tensor],
    optimal_params: dict[str, Tensor],
    ewc_lambda: float,
) -> Tensor:
    """Compute EWC penalty to prevent catastrophic forgetting.

    Penalty = ewc_lambda/2 * sum_i F_i * (theta_i - theta*_i)^2

    Args:
        model: Current model.
        fisher: Dict of Fisher diagonal tensors per parameter name.
        optimal_params: Dict of optimal (reference) parameter tensors.
        ewc_lambda: Penalty strength.

    Returns:
        Scalar penalty tensor.
    """
    device = next(model.parameters()).device
    penalty = torch.tensor(0.0, device=device)

    for name, param in model.named_parameters():
        if name not in fisher or name not in optimal_params:
            continue
        f = fisher[name].to(device)
        opt = optimal_params[name].to(device)
        penalty = penalty + (f * (param - opt).pow(2)).sum()

    return ewc_lambda / 2.0 * penalty


class OnlineLearner:
    """Online learner with experience replay, EWC, and drift detection."""

    def __init__(
        self,
        model: nn.Module,
        config: OnlineLearningConfig,
        optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self._loss_history: list[float] = []
        self._fisher: dict[str, Tensor] = {}
        self._optimal_params: dict[str, Tensor] = {}

    def _loss_fn(self, model: nn.Module, input_ids: Tensor, labels: Tensor) -> Tensor:
        """Compute cross-entropy loss manually using model logits."""
        _, logits, _ = model(input_ids)
        # Shift for causal LM: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
        )

    def update(self, input_ids: Tensor, labels: Tensor) -> dict:
        """Process one streaming batch: replay + EWC + drift detection.

        Args:
            input_ids: (batch, seq_len) token indices.
            labels: (batch, seq_len) target token ids.

        Returns:
            Dict with keys: task_loss, replay_loss, drift_detected, buffer_size.
        """
        # Add current sample to replay buffer
        self.replay_buffer.add({"input_ids": input_ids, "labels": labels})

        self.model.train()
        self.optimizer.zero_grad()

        # Compute task loss on current batch
        task_loss_tensor = self._loss_fn(self.model, input_ids, labels)
        task_loss_val = task_loss_tensor.item()

        # Sample from replay buffer and compute replay loss
        replay_loss_tensor = torch.tensor(0.0, device=task_loss_tensor.device)
        replay_loss_val = 0.0

        if len(self.replay_buffer) > 0:
            n_replay = min(self.config.replay_batch_size, len(self.replay_buffer))
            replay_samples = self.replay_buffer.sample(n_replay)
            replay_losses = []
            for sample in replay_samples:
                r_ids = sample["input_ids"].to(task_loss_tensor.device)
                r_lbl = sample["labels"].to(task_loss_tensor.device)
                r_loss = self._loss_fn(self.model, r_ids, r_lbl)
                replay_losses.append(r_loss)
            if replay_losses:
                replay_loss_tensor = torch.stack(replay_losses).mean()
                replay_loss_val = replay_loss_tensor.item()

        # Accumulate loss history and check for concept drift
        self._loss_history.append(task_loss_val)
        drift_detected = detect_concept_drift(
            self._loss_history,
            self.config.drift_detection_window,
            self.config.drift_threshold,
        )

        # Combine losses
        total_loss = task_loss_tensor + replay_loss_tensor

        # Backward and optimizer step
        total_loss.backward()
        self.optimizer.step()

        return {
            "task_loss": task_loss_val,
            "replay_loss": replay_loss_val,
            "drift_detected": drift_detected,
            "buffer_size": len(self.replay_buffer),
        }

    def consolidate(self) -> None:
        """Save current params as optimal_params and compute Fisher on replay buffer."""
        # Save current parameters as the optimal reference
        self._optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Sample up to 10 items from replay buffer for Fisher computation
        n_fisher = min(10, len(self.replay_buffer))
        if n_fisher == 0:
            self._fisher = {}
            return

        fisher_samples = self.replay_buffer.sample(n_fisher)
        self._fisher = compute_fisher_diagonal(
            self.model,
            fisher_samples,
            self._loss_fn,
        )
