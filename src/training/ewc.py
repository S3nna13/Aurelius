"""Elastic Weight Consolidation (EWC) for continual learning.

Kirkpatrick et al., 2017 - arXiv:1612.00796

Computes the diagonal Fisher Information Matrix after training on task A,
then adds a quadratic penalty during task B training to prevent forgetting A.

Penalty: lambda/2 * sum_i F_i * (theta_i - theta*_i)^2

Where:
- F_i = Fisher information for parameter i (importance weight)
- theta*_i = optimal parameter value after task A
- theta_i = current parameter value
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EWCConfig:
    ewc_lambda: float = 5000.0    # penalty strength (higher = more conservative)
    n_fisher_samples: int = 200   # samples to estimate Fisher info
    fisher_batch_size: int = 4    # batch size during Fisher estimation


class EWC:
    """Elastic Weight Consolidation regularizer.

    Usage:
        # After training on task A:
        ewc = EWC(model, cfg)
        ewc.compute_fisher(task_a_dataloader)

        # During task B training:
        loss = task_b_loss + ewc.penalty(model)

    Args:
        model: The model (must remain the same Python object).
        cfg: EWC configuration.
    """

    def __init__(self, model: nn.Module, cfg: EWCConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or EWCConfig()
        # Stored after compute_fisher():
        self._fisher: dict[str, torch.Tensor] = {}   # param_name -> diagonal Fisher
        self._optimal: dict[str, torch.Tensor] = {}  # param_name -> theta* (frozen copy)

    def compute_fisher(self, dataloader) -> None:
        """Estimate diagonal Fisher information from task A data.

        Runs model forward on sampled batches, backpropagates the log-likelihood,
        and accumulates squared gradients as Fisher diagonal estimate.

        After this call, `self._fisher` and `self._optimal` are populated.

        Args:
            dataloader: DataLoader yielding {"input_ids": Tensor, "labels": Tensor}
                        or (input_ids, labels) tuples.
        """
        self.model.set_grad_checkpointing(False) if hasattr(self.model, 'set_grad_checkpointing') else None
        self.model.eval()

        # Store current parameters as theta*
        self._optimal = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Initialize Fisher accumulators
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        n_samples = 0
        for batch in dataloader:
            if n_samples >= self.cfg.n_fisher_samples:
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch.get("labels", batch["input_ids"])
            else:
                input_ids, labels = batch[0], batch[1]

            # Take a mini-batch
            ids_batch = input_ids[:self.cfg.fisher_batch_size]
            lbl_batch = labels[:self.cfg.fisher_batch_size]

            self.model.zero_grad()
            loss, _, _ = self.model(input_ids=ids_batch, labels=lbl_batch)
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            n_samples += ids_batch.shape[0]

        # Average and store
        n_batches = max(1, n_samples // self.cfg.fisher_batch_size)
        self._fisher = {name: f / n_batches for name, f in fisher.items()}

        logger.info(
            "Computed Fisher information over %d samples (%d batches)",
            n_samples, n_batches,
        )
        self.model.train()

    def penalty(self, model: nn.Module | None = None) -> torch.Tensor:
        """Compute EWC penalty term for current model parameters.

        Args:
            model: Model to compute penalty for (defaults to self.model).

        Returns:
            Scalar penalty tensor (to add to task loss before backward).

        Raises:
            RuntimeError: If compute_fisher() has not been called yet.
        """
        if not self._fisher:
            raise RuntimeError(
                "EWC.compute_fisher() must be called before EWC.penalty(). "
                "Run compute_fisher() after training on the previous task."
            )

        model = model or self.model
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name not in self._fisher:
                continue
            fisher = self._fisher[name].to(param.device)
            optimal = self._optimal[name].to(param.device)
            penalty = penalty + (fisher * (param - optimal).pow(2)).sum()

        return self.cfg.ewc_lambda / 2 * penalty

    def is_ready(self) -> bool:
        """Returns True if Fisher information has been computed."""
        return len(self._fisher) > 0

    def named_importances(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return top-n most important parameters by Fisher information magnitude.

        Useful for understanding which layers EWC is protecting.

        Returns:
            List of (param_name, mean_fisher) sorted descending.
        """
        scores = [
            (name, f.mean().item())
            for name, f in self._fisher.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
