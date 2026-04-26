"""Multi-task learning with task routing and shared representations.

Provides task-specific heads on a shared backbone transformer, a soft task
router, weighted multi-task loss, and a trainer that orchestrates them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Configuration for a single task in the multi-task setup."""

    task_name: str
    task_type: str  # "classification" | "generation" | "regression"
    n_classes: int = 2
    weight: float = 1.0


# ---------------------------------------------------------------------------
# TaskHead
# ---------------------------------------------------------------------------


class TaskHead(nn.Module):
    """Task-specific projection head on top of a shared backbone.

    For classification and regression: linear d_model -> n_classes.
    For generation: identity pass-through (returns hidden states unchanged).
    """

    def __init__(self, d_model: int, task_config: TaskConfig) -> None:
        super().__init__()
        self.task_config = task_config
        self.task_type = task_config.task_type

        if task_config.task_type in ("classification", "regression"):
            self.proj = nn.Linear(d_model, task_config.n_classes, bias=True)
        else:
            # generation: pass-through
            self.proj = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to task output space.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            (B, T, n_classes) for classification/regression,
            (B, T, d_model)  for generation.
        """
        if self.proj is not None:
            return self.proj(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# MultiTaskRouter
# ---------------------------------------------------------------------------


class MultiTaskRouter(nn.Module):
    """Soft task router: produces per-sample task weight distribution.

    Mean-pools hidden states over the sequence dimension, applies a linear
    layer, then softmax to obtain a probability over n_tasks.
    """

    def __init__(self, d_model: int, n_tasks: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_tasks, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute soft task weights.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            (B, n_tasks) task weights summing to 1 along dim=1.
        """
        pooled = hidden_states.mean(dim=1)  # (B, d_model)
        logits = self.linear(pooled)  # (B, n_tasks)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# compute_multitask_loss
# ---------------------------------------------------------------------------


def compute_multitask_loss(
    task_losses: dict[str, torch.Tensor],
    task_weights: dict[str, float],
) -> torch.Tensor:
    """Compute weighted sum of per-task losses: sum_i(w_i * L_i).

    Args:
        task_losses:  Mapping task_name -> scalar loss tensor.
        task_weights: Mapping task_name -> scalar weight (float).

    Returns:
        Scalar tensor representing the combined loss.
    """
    total: torch.Tensor | None = None
    for name, loss in task_losses.items():
        w = task_weights.get(name, 1.0)
        weighted = w * loss
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("task_losses must not be empty")
    return total


# ---------------------------------------------------------------------------
# MultiTaskTrainer
# ---------------------------------------------------------------------------


class MultiTaskTrainer:
    """Trains a shared backbone together with per-task heads.

    The backbone must satisfy the Aurelius API:
        loss, logits, pkv = model(input_ids)

    Task heads are stored in a registry keyed by task_name.
    """

    def __init__(
        self,
        shared_model: nn.Module,
        task_configs: list[TaskConfig],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.shared_model = shared_model
        self.optimizer = optimizer

        if not hasattr(shared_model, "config"):
            raise ValueError("shared_model must expose a .config attribute with d_model")
        self._d_model: int = shared_model.config.d_model

        # Registry: task_name -> TaskHead
        self.task_heads: dict[str, TaskHead] = {}
        for tc in task_configs:
            self.add_task_head(tc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task_head(self, task_config: TaskConfig) -> None:
        """Register a new TaskHead for the given task."""
        self.task_heads[task_config.task_name] = TaskHead(self._d_model, task_config)

    def train_step(self, input_ids: torch.Tensor, task_name: str) -> dict:
        """One gradient update for a given task.

        Args:
            input_ids: (B, S) integer token ids.
            task_name: Which registered task to train.

        Returns:
            {"loss": float, "task": str}
        """
        if task_name not in self.task_heads:
            raise KeyError(f"Unknown task: {task_name!r}")

        self.shared_model.train()
        self.optimizer.zero_grad()

        hidden = self._extract_hidden(input_ids)
        head = self.task_heads[task_name]
        loss = self._task_loss(hidden, head, input_ids) * head.task_config.weight

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "task": task_name}

    def get_task_losses(
        self,
        input_ids: torch.Tensor,
        task_names: list[str],
    ) -> dict[str, float]:
        """Compute loss for each task without backward pass.

        Args:
            input_ids:  (B, S) integer token ids.
            task_names: Task names to evaluate.

        Returns:
            Mapping task_name -> float loss.
        """
        self.shared_model.eval()
        hidden = self._extract_hidden_no_grad(input_ids)

        result: dict[str, float] = {}
        for name in task_names:
            if name not in self.task_heads:
                raise KeyError(f"Unknown task: {name!r}")
            head = self.task_heads[name]
            with torch.no_grad():
                loss = self._task_loss(hidden, head, input_ids)
            result[name] = loss.item()
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract post-norm hidden states via a forward hook (with grad)."""
        captured: list[torch.Tensor] = []

        def hook(module, inp, output):  # noqa: ARG001
            t = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(t)

        target = getattr(self.shared_model, "norm", None) or self.shared_model.layers[-1]
        handle = target.register_forward_hook(hook)
        try:
            self.shared_model(input_ids)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Failed to capture hidden states")
        return captured[0]

    def _extract_hidden_no_grad(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states without gradient tracking."""
        with torch.no_grad():
            return self._extract_hidden(input_ids)

    def _task_loss(
        self,
        hidden: torch.Tensor,
        head: TaskHead,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar loss for one task given post-backbone hidden states."""
        output = head(hidden)  # (B, T, C)
        B, T, C = output.shape
        tc = head.task_config

        if tc.task_type == "classification":
            labels = (input_ids[:, 1:] % tc.n_classes).long()  # (B, T-1)
            logits = output[:, :-1, :].contiguous()  # (B, T-1, C)
            return F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1))

        elif tc.task_type == "regression":
            return output.pow(2).mean()

        else:
            # generation: project back to vocab via lm_head if available
            lm_head = getattr(self.shared_model, "lm_head", None)
            logits = lm_head(output) if lm_head is not None else output
            vocab_size = logits.size(-1)
            labels = input_ids[:, 1:].long()
            logits = logits[:, :-1, :].contiguous()
            return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
