"""Multi-task learning with task embeddings, uncertainty weighting, and GradNorm."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskConfig:
    """Configuration for multi-task training with task embeddings."""

    n_tasks: int = 4
    task_embed_dim: int = 16
    loss_weighting: str = "uniform"  # "uniform" | "uncertainty" | "gradnorm"
    lr: float = 1e-4


# ---------------------------------------------------------------------------
# Task Embedding
# ---------------------------------------------------------------------------

class TaskEmbedding(nn.Module):
    """Learnable embedding for each task.

    Produces a dense vector of shape (embed_dim,) for a given task_id.
    task_id can be a Python int or a 0-d / 1-d long tensor.
    """

    def __init__(self, n_tasks: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_tasks, embed_dim)

    def forward(self, task_id: int | Tensor) -> Tensor:
        """Return embedding vector for task_id.

        Args:
            task_id: integer task index or scalar/1-d long tensor.

        Returns:
            Tensor of shape (embed_dim,).
        """
        if isinstance(task_id, int):
            idx = torch.tensor(task_id, dtype=torch.long,
                               device=self.embedding.weight.device)
        else:
            idx = task_id.long()
            # Flatten to scalar if possible
            if idx.dim() > 0:
                idx = idx.view(-1)[0]

        return self.embedding(idx)  # shape: (embed_dim,)


# ---------------------------------------------------------------------------
# Uncertainty Weighting  (Kendall et al., 2018)
# ---------------------------------------------------------------------------

class UncertaintyWeighting(nn.Module):
    """Multi-task loss weighting via learnable task uncertainty (log-sigma).

    For each task i:
        weight_i = 1 / (2 * sigma_i^2)
        total    = sum_i( weight_i * loss_i + log(sigma_i) )

    log_sigma is stored as a learnable parameter; sigma = exp(log_sigma).
    """

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        # log_sigma initialised to 0 => sigma=1, weight=0.5
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[Tensor]) -> Tensor:
        """Combine per-task losses with uncertainty weighting.

        Args:
            losses: list of scalar loss tensors, one per task.

        Returns:
            Scalar weighted total loss.
        """
        total = torch.zeros(1, device=self.log_sigma.device).squeeze()
        for i, loss in enumerate(losses):
            sigma_sq = torch.exp(self.log_sigma[i]) ** 2          # sigma^2
            weight = 1.0 / (2.0 * sigma_sq)                       # 1/(2*sigma^2)
            total = total + weight * loss + self.log_sigma[i]      # + log(sigma)
        return total


# ---------------------------------------------------------------------------
# GradNorm Weighting
# ---------------------------------------------------------------------------

class GradNormWeighting:
    """GradNorm dynamic loss weighting (Chen et al., 2018).

    Tracks the ratio of each task's current loss to its initial loss,
    then adjusts weights so tasks with higher relative loss gain more weight.

    alpha controls the restoring force (higher -> stronger equalisation).
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5) -> None:
        self.n_tasks = n_tasks
        self.alpha = alpha
        # Start with uniform weights summing to n_tasks
        self._weights = torch.ones(n_tasks, dtype=torch.float32)

    def update(self, losses: List[Tensor], initial_losses: List[Tensor]) -> Tensor:
        """Recompute task weights based on relative loss decrease.

        Args:
            losses:         current per-task loss tensors (one per task).
            initial_losses: initial per-task loss tensors (one per task).

        Returns:
            Updated weight tensor of shape (n_tasks,) with positive values
            that sum approximately to n_tasks.
        """
        assert len(losses) == self.n_tasks
        assert len(initial_losses) == self.n_tasks

        with torch.no_grad():
            # Relative inverse training rate: loss_i / loss_i_0
            ratios = torch.stack([
                (l.detach() / (l0.detach() + 1e-8)).clamp(min=1e-8)
                for l, l0 in zip(losses, initial_losses)
            ])  # (n_tasks,)

            mean_ratio = ratios.mean()

            # Target rate: r_i / mean_r raised to alpha
            target = (ratios / (mean_ratio + 1e-8)).pow(self.alpha)

            # Renormalise so weights sum to n_tasks
            target = target / (target.sum() + 1e-8) * self.n_tasks

            self._weights = target.clamp(min=1e-4)

        return self._weights.clone()

    @property
    def weights(self) -> Tensor:
        return self._weights.clone()


# ---------------------------------------------------------------------------
# Multi-Task Trainer
# ---------------------------------------------------------------------------

class MultiTaskTrainer:
    """Trains a model across multiple tasks simultaneously.

    Samples one batch from each task loader per step, computes per-task losses
    using the model's plain-tuple API ``loss, logits, pkv = model(input_ids)``,
    combines them via the configured weighting scheme, and runs a backward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MultiTaskConfig,
        task_loaders: Dict[int, DataLoader],
    ) -> None:
        self.model = model
        self.config = config
        self.task_loaders = task_loaders
        self.n_tasks = len(task_loaders)

        # Build infinite iterators for each loader
        self._iters: Dict[int, Iterator] = {
            tid: iter(loader) for tid, loader in task_loaders.items()
        }

        # Set up weighting module / helper
        self._uncertainty: Optional[UncertaintyWeighting] = None
        self._gradnorm: Optional[GradNormWeighting] = None
        self._gradnorm_initial: Optional[List[Tensor]] = None

        if config.loss_weighting == "uncertainty":
            self._uncertainty = UncertaintyWeighting(self.n_tasks)
        elif config.loss_weighting == "gradnorm":
            self._gradnorm = GradNormWeighting(self.n_tasks)

        # Build optimizer (model params + any weighting params)
        params = list(model.parameters())
        if self._uncertainty is not None:
            params += list(self._uncertainty.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=config.lr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_batch(self, task_id: int) -> Tensor:
        """Fetch next batch input_ids for the given task, cycling the loader."""
        try:
            batch = next(self._iters[task_id])
        except StopIteration:
            self._iters[task_id] = iter(self.task_loaders[task_id])
            batch = next(self._iters[task_id])

        # Accept either a raw tensor or a (input_ids, ...) tuple/list
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch

    def _compute_loss(self, input_ids: Tensor) -> Tensor:
        """Run model and return the loss scalar via the plain-tuple API.

        If the model returns None for loss (no labels passed), we compute
        the next-token cross-entropy loss from the logits directly.
        """
        loss, logits, _pkv = self.model(input_ids)
        if loss is None:
            # logits: (B, S, V) — predict next token from all but last position
            shift_logits = logits[:, :-1, :].contiguous()       # (B, S-1, V)
            shift_labels = input_ids[:, 1:].contiguous()         # (B, S-1)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return loss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self) -> Dict[str, object]:
        """One training step over all tasks.

        Returns:
            dict with keys:
              - ``per_task_losses``: dict mapping task_id -> float
              - ``total_loss``: scalar float
        """
        self.model.train()
        self.optimizer.zero_grad()

        task_ids = list(self.task_loaders.keys())
        per_task_loss_tensors: Dict[int, Tensor] = {}

        for tid in task_ids:
            input_ids = self._next_batch(tid)
            loss = self._compute_loss(input_ids)
            per_task_loss_tensors[tid] = loss

        loss_list = [per_task_loss_tensors[tid] for tid in task_ids]

        # ------ Combine losses ------
        if self._uncertainty is not None:
            total = self._uncertainty(loss_list)

        elif self._gradnorm is not None:
            # Initialise initial losses on first step
            if self._gradnorm_initial is None:
                self._gradnorm_initial = [l.detach().clone() for l in loss_list]

            weights = self._gradnorm.update(loss_list, self._gradnorm_initial)
            device = loss_list[0].device
            weights = weights.to(device)
            total = sum(w * l for w, l in zip(weights, loss_list))

        else:
            # Uniform weighting
            total = sum(loss_list) / len(loss_list)

        total.backward()
        self.optimizer.step()

        per_task_losses = {tid: per_task_loss_tensors[tid].item() for tid in task_ids}

        return {
            "per_task_losses": per_task_losses,
            "total_loss": total.item(),
        }
