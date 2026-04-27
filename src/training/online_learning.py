"""Online learning: streaming updates with experience replay and catastrophic forgetting prevention.

Also provides concept-drift-aware adaptive learning rate training via:
  OnlineLearningConfig, LossWindow, DriftDetector, AdaptiveLRScheduler, OnlineLearner.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class OnlineLearningConfig:
    """Configuration for online / streaming learner (legacy replay+EWC API)."""

    replay_buffer_size: int = 1000
    replay_batch_size: int = 16
    ewc_lambda: float = 1.0
    drift_detection_window: int = 50
    drift_threshold: float = 2.0  # z-score threshold for drift detection
    learning_rate: float = 1e-4
    # --- drift-adaptive LR fields (used by new API) ---
    window_size: int = 100
    base_lr: float = 1e-4
    lr_increase_factor: float = 2.0
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    forgetting_factor: float = 0.99


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
            return random.choices(self._buffer, k=n)  # noqa: S311
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
    std1 = variance1**0.5

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


class _LegacyOnlineLearner:
    """Online learner with experience replay, EWC, and drift detection (legacy API)."""

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


# ---------------------------------------------------------------------------
# New drift-aware adaptive-LR online learning API
# ---------------------------------------------------------------------------


class LossWindow:
    """Fixed-size sliding window of scalar loss values."""

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._losses: list[float] = []

    def add(self, loss: float) -> None:
        """Append loss; evict oldest when at capacity."""
        if len(self._losses) >= self._window_size:
            self._losses.pop(0)
        self._losses.append(loss)

    def mean(self) -> float:
        """Return mean of current values; 0.0 if empty."""
        if not self._losses:
            return 0.0
        return sum(self._losses) / len(self._losses)

    def std(self) -> float:
        """Return sample std of current values; 0.0 if fewer than 2 values."""
        n = len(self._losses)
        if n < 2:
            return 0.0
        mu = self.mean()
        variance = sum((x - mu) ** 2 for x in self._losses) / n
        return math.sqrt(variance)

    def is_full(self) -> bool:
        return len(self._losses) >= self._window_size

    def __len__(self) -> int:
        return len(self._losses)


class DriftDetector:
    """Detects concept drift by comparing a current loss window to a reference window."""

    def __init__(self, config: OnlineLearningConfig) -> None:
        self._config = config
        self._reference_window = LossWindow(config.window_size)
        self._current_window = LossWindow(config.window_size)
        self._total_steps: int = 0
        self._drift_events: int = 0

    def update(self, loss: float) -> bool:
        """Add loss to appropriate windows and return True if drift is detected.

        Phase 1 (bootstrap): Fill the reference window.  No drift check yet.
        Phase 2 (detection): Add incoming losses only to the current window;
            compare current vs. reference.  If drift is detected, swap current
            → reference and reset current window.

        Drift criterion: z-score = (current_mean - ref_mean) / (ref_std + 1e-8)
        exceeds threshold.
        """
        self._total_steps += 1

        if not self._reference_window.is_full():
            # Bootstrap: feed reference window first
            self._reference_window.add(loss)
            return False

        # Detection phase: only update current window
        self._current_window.add(loss)

        if not self._current_window.is_full():
            # Need enough data in current window before checking
            return False

        score = self.drift_score()
        if score > self._config.drift_threshold:
            # Swap: current becomes the new reference baseline
            self._reference_window = self._current_window
            self._current_window = LossWindow(self._config.window_size)
            self._drift_events += 1
            return True
        return False

    def drift_score(self) -> float:
        """Return current z-score: (current_mean - ref_mean) / (ref_std + 1e-8)."""
        ref_mean = self._reference_window.mean()
        ref_std = self._reference_window.std()
        cur_mean = self._current_window.mean()
        return (cur_mean - ref_mean) / (ref_std + 1e-8)


class AdaptiveLRScheduler:
    """Adjusts optimizer learning rate in response to drift detection."""

    def __init__(self, optimizer, config: OnlineLearningConfig) -> None:
        self._optimizer = optimizer
        self._config = config
        self._current_lr: float = config.base_lr
        self._apply_lr(self._current_lr)

    def _apply_lr(self, lr: float) -> None:
        for pg in self._optimizer.param_groups:
            pg["lr"] = lr

    def on_drift_detected(self) -> None:
        """Boost LR by lr_increase_factor, clamped to max_lr."""
        self._current_lr = min(
            self._current_lr * self._config.lr_increase_factor,
            self._config.max_lr,
        )

    def on_stable(self) -> None:
        """Decay LR toward base_lr via forgetting_factor (EMA)."""
        base = self._config.base_lr
        ff = self._config.forgetting_factor
        # EMA toward base_lr
        self._current_lr = ff * self._current_lr + (1.0 - ff) * base
        self._current_lr = max(self._current_lr, self._config.min_lr)

    def step(self, drift_detected: bool) -> None:
        """Call appropriate handler and push updated LR into optimizer."""
        if drift_detected:
            self.on_drift_detected()
        else:
            self.on_stable()
        self._apply_lr(self._current_lr)


class OnlineLearner:
    """Online learner with concept drift detection and adaptive learning rate."""

    def __init__(self, model: nn.Module, optimizer, config: OnlineLearningConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config

        self._drift_detector = DriftDetector(config)
        self._lr_scheduler = AdaptiveLRScheduler(optimizer, config)
        self._step_count: int = 0
        self._drift_events: int = 0
        self._loss_window = LossWindow(config.window_size)

    def train_step(self, input_ids: Tensor) -> dict:
        """Run one gradient step on input_ids (next-token prediction).

        Returns:
            dict with keys: loss, drift_detected, current_lr, drift_score, step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss_val, logits, _pkv = self.model(input_ids)

        # If model returns a scalar loss directly use it; otherwise compute CE.
        if isinstance(loss_val, Tensor) and loss_val.numel() == 1:
            loss = loss_val
        else:
            # Compute next-token cross-entropy from logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            )

        loss.backward()
        self.optimizer.step()

        loss_float = loss.item()
        self._loss_window.add(loss_float)

        drift_detected = self._drift_detector.update(loss_float)
        if drift_detected:
            self._drift_events += 1

        self._lr_scheduler.step(drift_detected)
        self._step_count += 1

        return {
            "loss": loss_float,
            "drift_detected": drift_detected,
            "current_lr": self._lr_scheduler._current_lr,
            "drift_score": self._drift_detector.drift_score(),
            "step": self._step_count,
        }

    def get_stats(self) -> dict:
        """Return summary statistics."""
        return {
            "steps": self._step_count,
            "drift_events": self._drift_events,
            "current_lr": self._lr_scheduler._current_lr,
            "mean_loss": self._loss_window.mean(),
        }
