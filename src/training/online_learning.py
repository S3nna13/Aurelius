"""Online learning and continual pretraining: streaming data, catastrophic forgetting prevention."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class OnlineLearningConfig:
    """Configuration for online / continual-pretraining learner."""
    buffer_size: int = 1000
    replay_ratio: float = 0.3
    lr_warmup_steps: int = 100
    forgetting_threshold: float = 0.1
    use_ewc: bool = True
    ewc_lambda: float = 0.4


class StreamingDataBuffer:
    """Circular buffer for streaming online data."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self._buffer: list[tuple[Tensor, Tensor]] = []
        self._capacity: int = capacity
        self._rng: random.Random = random.Random(seed)

    def add(self, input_ids: Tensor, labels: Tensor) -> None:
        if len(self._buffer) >= self._capacity:
            self._buffer.pop(0)
        self._buffer.append((input_ids, labels))

    def sample(self, n: int) -> list[tuple[Tensor, Tensor]]:
        if len(self._buffer) == 0:
            return []
        replace = len(self._buffer) < n
        if replace:
            return [self._rng.choice(self._buffer) for _ in range(n)]
        return self._rng.sample(self._buffer, n)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int = 10) -> bool:
        return len(self._buffer) >= min_size


class OnlineLearner:
    """Continual learning with experience replay and EWC regularization."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: OnlineLearningConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.buffer: StreamingDataBuffer = StreamingDataBuffer(config.buffer_size)
        self.fisher_diagonal: dict[str, Tensor] | None = None
        self.reference_params: dict[str, Tensor] | None = None
        self.step_count: int = 0

    def compute_fisher(self, calibration_batches: list[tuple[Tensor, Tensor]]) -> None:
        """Estimate diagonal Fisher information via squared gradients."""
        self.model.eval()

        self.reference_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        fisher: dict[str, Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        n_batches = max(1, len(calibration_batches))
        for input_ids, labels in calibration_batches:
            self.model.zero_grad()
            loss, _, _ = self.model(input_ids=input_ids, labels=labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

        self.fisher_diagonal = {name: f / n_batches for name, f in fisher.items()}
        self.model.train()

    def ewc_penalty(self) -> Tensor:
        """EWC penalty: fisher * (param - ref)^2 * lambda / 2, summed over params."""
        if self.fisher_diagonal is None or self.reference_params is None:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        device = next(self.model.parameters()).device
        penalty = torch.tensor(0.0, device=device)

        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.fisher_diagonal:
                continue
            fisher = self.fisher_diagonal[name].to(device)
            ref = self.reference_params[name].to(device)
            penalty = penalty + (fisher * (param - ref).pow(2)).sum()

        return self.config.ewc_lambda / 2.0 * penalty

    def train_on_batch(
        self, input_ids: Tensor, labels: Tensor
    ) -> dict[str, float]:
        """Add to buffer, mix with replay, forward + EWC, backward, step."""
        self.buffer.add(input_ids, labels)
        self.model.train()
        self.optimizer.zero_grad()

        loss, _, _ = self.model(input_ids=input_ids, labels=labels)

        replay_loss_val = 0.0
        if self.buffer.is_ready():
            n_replay = max(1, int(input_ids.shape[0] * self.config.replay_ratio))
            replay_samples = self.buffer.sample(n_replay)
            replay_loss = torch.tensor(0.0, device=loss.device)
            for r_ids, r_lbl in replay_samples:
                r_ids = r_ids.to(loss.device)
                r_lbl = r_lbl.to(loss.device)
                r_loss, _, _ = self.model(input_ids=r_ids, labels=r_lbl)
                replay_loss = replay_loss + r_loss
            replay_loss = replay_loss / len(replay_samples)
            replay_loss_val = replay_loss.item()
        else:
            replay_loss = torch.tensor(0.0, device=loss.device)

        ewc_loss = torch.tensor(0.0, device=loss.device)
        if self.config.use_ewc:
            ewc_loss = self.ewc_penalty()
        ewc_loss_val = ewc_loss.item()

        total_loss = loss + replay_loss + ewc_loss
        total_loss.backward()
        self.optimizer.step()
        self.step_count += 1

        return {
            "loss": loss.item(),
            "replay_loss": replay_loss_val,
            "ewc_loss": ewc_loss_val,
            "buffer_size": len(self.buffer),
        }

    def evaluate_forgetting(
        self,
        old_batches: list[tuple[Tensor, Tensor]],
        baseline_losses: list[float],
    ) -> dict[str, float]:
        """Compare current losses on old batches vs baselines."""
        self.model.eval()
        forgetting_values: list[float] = []

        with torch.no_grad():
            for (input_ids, labels), baseline in zip(old_batches, baseline_losses):
                loss, _, _ = self.model(input_ids=input_ids, labels=labels)
                delta = loss.item() - baseline
                forgetting_values.append(delta)

        self.model.train()

        mean_forgetting = sum(forgetting_values) / max(1, len(forgetting_values))
        max_forgetting = max(forgetting_values) if forgetting_values else 0.0
        catastrophic = max_forgetting > self.config.forgetting_threshold

        return {
            "mean_forgetting": mean_forgetting,
            "max_forgetting": max_forgetting,
            "catastrophic": catastrophic,
        }


class DataStreamSimulator:
    """Simulates a stream of task-shifted synthetic data."""

    def __init__(
        self,
        vocab_size: int = 256,
        seq_len: int = 16,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def next_batch(
        self, task_id: int, batch_size: int = 4
    ) -> tuple[Tensor, Tensor]:
        """Generate a synthetic batch biased toward a token range for this task.

        Returns (input_ids, labels) each of shape (batch_size, seq_len).
        labels = input_ids shifted by 1 (causal LM).
        """
        n_tasks = task_id + 2
        chunk = max(1, self.vocab_size // n_tasks)
        lo = task_id * chunk
        hi = min(lo + chunk, self.vocab_size)

        input_ids = torch.randint(lo, hi, (batch_size, self.seq_len), generator=self._rng)
        labels = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)
        return input_ids, labels


def cosine_similarity_params(params_a: dict, params_b: dict) -> float:
    """Cosine similarity between two flattened parameter vectors. Returns float in [-1, 1]."""
    shared_keys = [k for k in params_a if k in params_b]
    if not shared_keys:
        return 0.0

    vec_a = torch.cat([params_a[k].flatten().float() for k in shared_keys])
    vec_b = torch.cat([params_b[k].flatten().float() for k in shared_keys])

    dot = (vec_a * vec_b).sum()
    norm_a = vec_a.norm()
    norm_b = vec_b.norm()

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (dot / (norm_a * norm_b)).item()
