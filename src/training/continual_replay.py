"""Continual learning with experience replay: DER++, ring buffer replay, and selective memory consolidation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReplayConfig:
    """Configuration for continual replay training."""

    buffer_size: int = 1000                # max examples in replay buffer
    replay_fraction: float = 0.25          # fraction of batch from replay
    alpha: float = 0.1                     # DER++ logit replay weight
    beta: float = 0.5                      # DER++ CE on replay weight
    ewc_lambda: float = 0.01              # EWC regularization strength
    strategy: str = "der++"               # "der++" | "er" | "ewc_only"
    reservoir_sampling: bool = True        # reservoir sampling for buffer updates


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class ReplayExample:
    """A single stored example in the replay buffer."""

    input_ids: Tensor          # shape (T,)
    labels: Tensor             # shape (T,) with -100 for masked tokens
    logits: Optional[Tensor]   # stored teacher logits (T, V), used by DER++
    task_id: int
    timestamp: int


# ---------------------------------------------------------------------------
# Ring buffer / reservoir replay buffer
# ---------------------------------------------------------------------------


class RingBufferReplay:
    """Fixed-size replay buffer with optional reservoir sampling."""

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self._buffer: list[ReplayExample] = []
        self._ptr: int = 0        # next write position (ring mode)
        self._n_seen: int = 0     # total examples seen (reservoir mode)

    def add(self, example: ReplayExample) -> None:
        """Add a single example to the buffer."""
        self._n_seen += 1

        if self.config.reservoir_sampling:
            if len(self._buffer) < self.config.buffer_size:
                self._buffer.append(example)
            else:
                # Knuth's Algorithm R
                idx = random.randint(0, self._n_seen - 1)
                if idx < self.config.buffer_size:
                    self._buffer[idx] = example
        else:
            # Ring buffer: overwrite at _ptr
            if len(self._buffer) < self.config.buffer_size:
                self._buffer.append(example)
            else:
                self._buffer[self._ptr] = example
            self._ptr = (self._ptr + 1) % self.config.buffer_size

    def sample(self, n: int) -> list[ReplayExample]:
        """Randomly sample min(n, len(buffer)) examples."""
        k = min(n, len(self._buffer))
        if k == 0:
            return []
        return random.sample(self._buffer, k)

    def __len__(self) -> int:
        return len(self._buffer)

    def task_distribution(self) -> dict[int, int]:
        """Count examples per task_id."""
        dist: dict[int, int] = {}
        for ex in self._buffer:
            dist[ex.task_id] = dist.get(ex.task_id, 0) + 1
        return dist


# ---------------------------------------------------------------------------
# DER++ loss
# ---------------------------------------------------------------------------


def der_loss(
    current_logits: Tensor,
    replay_logits: Tensor,
    replay_labels: Tensor,
    alpha: float = 0.1,
    beta: float = 0.5,
) -> tuple[Tensor, dict]:
    """Compute DER++ loss.

    DER++ = alpha * MSE(current_logits_on_replay, stored_logits)
           + beta * CE(current, replay_labels)

    Args:
        current_logits:  (B_replay, T, V) model output on replay inputs.
        replay_logits:   (B_replay, T, V) stored dark experience logits.
        replay_labels:   (B_replay, T) labels (-100 for masked tokens).
        alpha:           weight on MSE distillation loss.
        beta:            weight on cross-entropy supervision loss.

    Returns:
        (total_loss, {"mse_loss": float, "ce_loss": float})
    """
    mse = F.mse_loss(current_logits, replay_logits)

    B, T, V = current_logits.shape
    ce = F.cross_entropy(
        current_logits.view(B * T, V),
        replay_labels.view(B * T),
        ignore_index=-100,
    )

    total = alpha * mse + beta * ce
    return total, {"mse_loss": mse.item(), "ce_loss": ce.item()}


# ---------------------------------------------------------------------------
# EWC helpers
# ---------------------------------------------------------------------------


def compute_ewc_penalty(
    model: nn.Module,
    fisher_info: dict[str, Tensor],
    old_params: dict[str, Tensor],
    ewc_lambda: float = 0.01,
) -> Tensor:
    """Compute EWC regularization penalty.

    Penalty = ewc_lambda * sum_i fisher_i * (param_i - old_param_i)^2

    Args:
        model:       Current model.
        fisher_info: {name: importance tensor} same shape as params.
        old_params:  {name: param value from previous task}.
        ewc_lambda:  Regularization strength.

    Returns:
        Scalar penalty tensor.
    """
    device = next(model.parameters()).device
    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in fisher_info and name in old_params:
            fisher = fisher_info[name].to(param.device)
            old = old_params[name].to(param.device)
            penalty = penalty + (fisher * (param - old).pow(2)).sum()
    return ewc_lambda * penalty


def estimate_fisher_diagonal(
    model: nn.Module,
    data_loader: list[Tensor],
    n_samples: int = 50,
) -> dict[str, Tensor]:
    """Estimate diagonal Fisher information via squared gradients.

    Args:
        model:       The model to estimate Fisher for.
        data_loader: List of input_ids tensors.
        n_samples:   Maximum number of samples to process.

    Returns:
        {param_name: mean_grad_squared} for params with requires_grad=True.
    """
    model.train()
    accum: dict[str, Tensor] = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    count = 0

    for batch in data_loader:
        if count >= n_samples:
            break

        if batch.dim() == 1:
            batch = batch.unsqueeze(0)

        model.zero_grad()
        loss, logits, _ = model(input_ids=batch, labels=batch)
        if loss is None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), batch.view(B * T))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                accum[name] += param.grad.detach().pow(2)

        count += 1

    if count > 0:
        for name in accum:
            accum[name] /= count

    return accum


# ---------------------------------------------------------------------------
# Continual Replay Trainer
# ---------------------------------------------------------------------------


class ContinualReplayTrainer:
    """Trainer combining current-task loss with replay-based continual learning.

    Supports DER++ (dark experience replay with logit distillation),
    plain ER (experience replay with CE only), and EWC-only modes.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ReplayConfig,
        optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.buffer = RingBufferReplay(config)
        self.fisher_info: dict[str, Tensor] | None = None
        self.old_params: dict[str, Tensor] | None = None
        self._step_count: int = 0

    def train_step(
        self,
        input_ids: Tensor,
        labels: Tensor,
        task_id: int = 0,
    ) -> dict[str, float]:
        """Single training step with optional replay and EWC.

        Args:
            input_ids: (B, T) current-task input tokens.
            labels:    (B, T) current-task labels (-100 for masked).
            task_id:   Integer task identifier.

        Returns:
            {"loss": float, "replay_loss": float, "ewc_loss": float}
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward on current batch
        ce_loss, current_logits, _ = self.model(input_ids=input_ids, labels=labels)
        if ce_loss is None:
            B, T, V = current_logits.shape
            ce_loss = F.cross_entropy(
                current_logits.view(B * T, V),
                labels.view(B * T),
                ignore_index=-100,
            )

        total_loss = ce_loss
        replay_loss_val = 0.0
        ewc_loss_val = 0.0

        # Replay
        n_replay = max(1, int(input_ids.shape[0] * self.config.replay_fraction))
        if len(self.buffer) >= n_replay and self.config.strategy != "ewc_only":
            replay_examples = self.buffer.sample(n_replay)
            seq_len = input_ids.shape[1]

            replay_ids_list = []
            replay_labels_list = []
            replay_logits_list = []
            has_stored_logits = (
                self.config.strategy == "der++"
                and all(ex.logits is not None for ex in replay_examples)
            )

            for ex in replay_examples:
                ids = ex.input_ids[:seq_len]
                lbl = ex.labels[:seq_len]
                T_ex = ids.shape[0]
                if T_ex < seq_len:
                    pad_len = seq_len - T_ex
                    ids = F.pad(ids, (0, pad_len), value=0)
                    lbl = F.pad(lbl, (0, pad_len), value=-100)
                replay_ids_list.append(ids)
                replay_labels_list.append(lbl)

                if has_stored_logits:
                    stored = ex.logits[:seq_len]
                    T_ex2 = stored.shape[0]
                    if T_ex2 < seq_len:
                        stored = F.pad(stored, (0, 0, 0, seq_len - T_ex2), value=0.0)
                    replay_logits_list.append(stored)

            replay_ids = torch.stack(replay_ids_list).to(input_ids.device)
            replay_labels = torch.stack(replay_labels_list).to(input_ids.device)

            _, replay_current_logits, _ = self.model(input_ids=replay_ids)

            if has_stored_logits:
                stored_logits = torch.stack(replay_logits_list).to(input_ids.device)
                r_loss, _ = der_loss(
                    replay_current_logits,
                    stored_logits,
                    replay_labels,
                    alpha=self.config.alpha,
                    beta=self.config.beta,
                )
            else:
                B_r, T, V = replay_current_logits.shape
                r_loss = F.cross_entropy(
                    replay_current_logits.view(B_r * T, V),
                    replay_labels.view(B_r * T),
                    ignore_index=-100,
                )

            total_loss = total_loss + r_loss
            replay_loss_val = r_loss.item()

        # EWC penalty
        if (
            self.config.strategy in ("der++", "ewc_only")
            and self.fisher_info is not None
            and self.old_params is not None
        ):
            ewc_pen = compute_ewc_penalty(
                self.model,
                self.fisher_info,
                self.old_params,
                ewc_lambda=self.config.ewc_lambda,
            )
            total_loss = total_loss + ewc_pen
            ewc_loss_val = ewc_pen.item()

        total_loss.backward()
        self.optimizer.step()

        # Store current batch in replay buffer with logits for DER++
        with torch.no_grad():
            _, store_logits, _ = self.model(input_ids=input_ids)

        B = input_ids.shape[0]
        for i in range(B):
            stored_logit = (
                store_logits[i].detach().cpu()
                if self.config.strategy == "der++"
                else None
            )
            ex = ReplayExample(
                input_ids=input_ids[i].detach().cpu(),
                labels=labels[i].detach().cpu(),
                logits=stored_logit,
                task_id=task_id,
                timestamp=self._step_count,
            )
            self.buffer.add(ex)

        self._step_count += 1

        return {
            "loss": total_loss.item(),
            "replay_loss": replay_loss_val,
            "ewc_loss": ewc_loss_val,
        }

    def consolidate_task(self, data_loader: list[Tensor]) -> None:
        """Estimate Fisher information and save current params for EWC.

        Should be called after finishing a task, before starting the next.

        Args:
            data_loader: List of input_ids tensors for the completed task.
        """
        self.fisher_info = estimate_fisher_diagonal(
            self.model,
            data_loader,
            n_samples=50,
        )
        self.old_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
