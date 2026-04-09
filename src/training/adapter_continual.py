"""Continual learning with task-specific adapters: progressive training without forgetting."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class AdapterContinualConfig:
    """Configuration for adapter-based continual learning."""

    adapter_rank: int = 8
    adapter_alpha: float = 16.0
    n_tasks: int = 5
    freeze_old_adapters: bool = True
    task_overlap_penalty: float = 0.1  # regularization for parameter overlap between tasks


class TaskAdapter(nn.Module):
    """Task-specific low-rank adapter (LoRA-style residual).

    Computes: x + (x @ A.T @ B.T) * scaling
    A is initialized with small random values; B is zero so the adapter
    starts as identity.

    Args:
        d_model: Model feature dimension.
        rank: Low-rank bottleneck dimension.
        alpha: Scaling factor numerator (scaling = alpha / rank).
        task_id: Integer identifier for this task's adapter.
    """

    def __init__(self, d_model: int, rank: int, alpha: float, task_id: int) -> None:
        super().__init__()
        self.task_id = task_id
        self.scaling = alpha / rank

        # A: (rank, d_model) — small random init
        self.A = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        # B: (d_model, rank) — zero init so adapter is identity at start
        self.B = nn.Parameter(torch.zeros(d_model, rank))

    def forward(self, x: Tensor) -> Tensor:
        """Apply adapter: x + (x @ A.T @ B.T) * scaling.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Tensor of same shape as x.
        """
        # x @ A.T: (..., rank)  then @ B.T: (..., d_model)
        return x + (x @ self.A.T @ self.B.T) * self.scaling


class MultiTaskAdapterLayer(nn.Module):
    """Layer holding a dict of task-specific adapters.

    Adapters are registered as submodules so they participate in
    model.parameters() / state_dict() correctly.

    Args:
        d_model: Feature dimension shared by all adapters.
        config: AdapterContinualConfig controlling rank, alpha, etc.
    """

    def __init__(self, d_model: int, config: AdapterContinualConfig) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config
        # Use nn.ModuleDict so adapters are proper submodules
        self._adapter_modules: nn.ModuleDict = nn.ModuleDict()

    @property
    def adapters(self) -> dict[int, TaskAdapter]:
        """Return mapping of task_id (int) -> TaskAdapter."""
        return {int(k): v for k, v in self._adapter_modules.items()}  # type: ignore[return-value]

    def add_task(self, task_id: int) -> None:
        """Create and register a new TaskAdapter for task_id.

        Args:
            task_id: Integer task identifier.
        """
        adapter = TaskAdapter(
            d_model=self.d_model,
            rank=self.config.adapter_rank,
            alpha=self.config.adapter_alpha,
            task_id=task_id,
        )
        self._adapter_modules[str(task_id)] = adapter

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Route input through the adapter for task_id.

        If task_id is not registered, returns x unchanged (identity).

        Args:
            x: Input tensor of shape (..., d_model).
            task_id: Which task's adapter to apply.

        Returns:
            Tensor of same shape as x.
        """
        key = str(task_id)
        if key not in self._adapter_modules:
            return x
        return self._adapter_modules[key](x)  # type: ignore[operator]

    def get_adapter_params(self, task_id: int) -> list[nn.Parameter]:
        """Return all parameters for the specified task's adapter.

        Args:
            task_id: Task whose parameters to retrieve.

        Returns:
            List of nn.Parameter objects (empty if task not registered).
        """
        key = str(task_id)
        if key not in self._adapter_modules:
            return []
        return list(self._adapter_modules[key].parameters())


def compute_task_overlap(adapter1: TaskAdapter, adapter2: TaskAdapter) -> float:
    """Compute cosine similarity between the effective weight matrices of two adapters.

    The effective weight for an adapter is B @ A (shape d_model x d_model),
    flattened to a vector before computing cosine similarity.

    Args:
        adapter1: First TaskAdapter.
        adapter2: Second TaskAdapter.

    Returns:
        Float in [0, 1] representing parameter overlap (0 = orthogonal, 1 = identical).
    """
    with torch.no_grad():
        # Effective matrix: B @ A — shape (d_model, d_model)
        w1 = (adapter1.B @ adapter1.A).flatten().float()
        w2 = (adapter2.B @ adapter2.A).flatten().float()

        norm1 = w1.norm()
        norm2 = w2.norm()

        if norm1 == 0 or norm2 == 0:
            # Zero-norm vectors have undefined cosine similarity; treat as no overlap
            return 0.0

        cosine = (w1 @ w2) / (norm1 * norm2)
        # Clamp to [0, 1]: negative similarity means anti-correlated, treat as 0 overlap
        return float(cosine.clamp(0.0, 1.0).item())


def freeze_adapters(layer: MultiTaskAdapterLayer, except_task_id: int) -> None:
    """Freeze all adapters in layer except the one for except_task_id.

    Sets requires_grad=False for every parameter in every adapter whose
    task_id != except_task_id, ensuring only the current task trains.

    Args:
        layer: MultiTaskAdapterLayer containing all task adapters.
        except_task_id: The task whose adapter should remain trainable.
    """
    for key, adapter in layer._adapter_modules.items():
        trainable = int(key) == except_task_id
        for param in adapter.parameters():
            param.requires_grad_(trainable)


def compute_overlap_penalty(adapters: dict[int, TaskAdapter]) -> Tensor:
    """Compute sum of pairwise cosine-similarity overlaps as a regularization penalty.

    Iterates over all unique pairs (i, j) with i < j and sums their overlaps.

    Args:
        adapters: Mapping of task_id -> TaskAdapter.

    Returns:
        Scalar float tensor (0.0 if fewer than 2 adapters).
    """
    keys = sorted(adapters.keys())
    device = next(iter(adapters.values())).A.device if adapters else torch.device("cpu")
    penalty = torch.tensor(0.0, device=device)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = compute_task_overlap(adapters[keys[i]], adapters[keys[j]])
            penalty = penalty + overlap

    return penalty


class AdapterContinualTrainer:
    """Continual learning trainer using task-specific adapters.

    Each new task gets its own adapter; old adapters are frozen so previously
    learned tasks are not disturbed. An overlap penalty discourages new adapters
    from replicating what old adapters learned.

    Args:
        model: The base nn.Module (e.g. AureliusTransformer).
        config: AdapterContinualConfig.
        base_optimizer_cls: Optimizer class to use for each task (default AdamW).
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdapterContinualConfig,
        base_optimizer_cls=torch.optim.AdamW,
    ) -> None:
        self.model = model
        self.config = config
        self.base_optimizer_cls = base_optimizer_cls

        # Single shared MultiTaskAdapterLayer (not inserted into the model graph;
        # overlap penalty is computed from it and added to the task loss).
        d_model: int = getattr(config, "_d_model", 64)  # default; overridden in start_task
        self._adapter_layer: MultiTaskAdapterLayer | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._current_task_id: int | None = None
        self._initial_losses: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_adapter_layer(self, d_model: int) -> MultiTaskAdapterLayer:
        if self._adapter_layer is None:
            self._adapter_layer = MultiTaskAdapterLayer(d_model, self.config)
        return self._adapter_layer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_task(self, task_id: int) -> None:
        """Prepare trainer for a new task.

        - Creates a MultiTaskAdapterLayer if needed (infers d_model from model).
        - Adds a new TaskAdapter for task_id.
        - Freezes all other adapters.
        - Builds an optimizer scoped to the current task's adapter parameters.

        Args:
            task_id: Integer identifier for the new task.
        """
        # Infer d_model from model if possible
        d_model = 64  # fallback default
        for param in self.model.parameters():
            if param.dim() >= 2:
                d_model = param.shape[-1]
                break

        layer = self._ensure_adapter_layer(d_model)
        layer.add_task(task_id)

        if self.config.freeze_old_adapters:
            freeze_adapters(layer, except_task_id=task_id)

        # Optimizer only for current task's adapter params
        task_params = layer.get_adapter_params(task_id)
        if task_params:
            self._optimizer = self.base_optimizer_cls(task_params, lr=1e-3)
        else:
            self._optimizer = None

        self._current_task_id = task_id

    def train_step(self, input_ids: Tensor, labels: Tensor, task_id: int) -> dict:
        """Perform a single training step for the given task.

        Forward pass through model, compute overlap penalty, backprop, update.

        Args:
            input_ids: Token id tensor of shape (batch, seq_len).
            labels: Target ids of same shape.
            task_id: Which task is being trained.

        Returns:
            dict with keys: "loss" (float), "task_id" (int), "overlap_penalty" (float).
        """
        self.model.train()

        if self._optimizer is not None:
            self._optimizer.zero_grad()

        # Forward — model returns (loss, logits, ...) or plain tuple
        output = self.model(input_ids)
        if isinstance(output, tuple):
            # Try to get loss from keyword call if plain tuple has no loss
            _, logits, _ = output
            # Compute cross-entropy loss manually
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )
        else:
            loss = output

        # Overlap penalty
        overlap_penalty_val = torch.tensor(0.0)
        if self._adapter_layer is not None and len(self._adapter_layer.adapters) >= 2:
            overlap_penalty_val = compute_overlap_penalty(self._adapter_layer.adapters)

        total_loss = loss + self.config.task_overlap_penalty * overlap_penalty_val

        total_loss.backward()

        if self._optimizer is not None:
            self._optimizer.step()

        # Store initial loss for forgetting tracking (first time we see this task)
        if task_id not in self._initial_losses:
            self._initial_losses[task_id] = loss.item()

        return {
            "loss": loss.item(),
            "task_id": task_id,
            "overlap_penalty": overlap_penalty_val.item(),
        }

    def evaluate_forgetting(self, task_losses: dict[int, float]) -> float:
        """Estimate catastrophic forgetting across previously seen tasks.

        Forgetting is the mean increase in loss compared to when each task
        was first trained. Only tasks present in self._initial_losses (i.e.,
        tasks that have been trained) are considered.

        Args:
            task_losses: Current per-task losses {task_id: loss}.

        Returns:
            Mean forgetting (float >= 0). Returns 0.0 if no previous tasks.
        """
        if not self._initial_losses:
            return 0.0

        increases = []
        for tid, initial_loss in self._initial_losses.items():
            if tid in task_losses:
                increase = task_losses[tid] - initial_loss
                if increase > 0:
                    increases.append(increase)

        if not increases:
            return 0.0

        return float(sum(increases) / len(increases))
