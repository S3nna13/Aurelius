"""Continual learning orchestrator: manages training across a sequence of tasks.

Uses EWC regularization to prevent catastrophic forgetting. Tracks per-task
performance to detect and report forgetting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as _F
from torch.utils.data import DataLoader

from src.training.ewc import EWC, EWCConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    """Record of a completed training task."""
    task_id: str
    final_loss: float
    eval_loss: float | None
    n_steps: int
    ewc_lambda: float           # lambda used for this task's EWC
    fisher_computed: bool       # whether Fisher was computed after this task


@dataclass
class ContinualConfig:
    base_ewc_lambda: float = 5000.0    # starting EWC penalty strength
    lambda_growth: float = 1.0         # multiply lambda by this per task (1.0 = constant)
    n_fisher_samples: int = 200        # samples for Fisher estimation
    steps_per_task: int = 100          # gradient steps per task
    lr: float = 1e-4
    forgetting_threshold: float = 0.1  # flag if loss regresses > 10%


@dataclass
class ContinualReport:
    task_records: list[TaskRecord]
    forgetting_events: list[str]  # task_id strings where forgetting was detected

    @property
    def n_tasks_completed(self) -> int:
        return len(self.task_records)

    @property
    def any_forgetting(self) -> bool:
        return len(self.forgetting_events) > 0


class ContinualTrainer:
    """Multi-task continual learning trainer with EWC regularization.

    Usage:
        trainer = ContinualTrainer(model, cfg)
        trainer.train_task("task_1", train_loader_1, loader_1)
        trainer.train_task("task_2", train_loader_2, loader_2)
        report = trainer.get_report()
    """

    def __init__(self, model: nn.Module, cfg: ContinualConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or ContinualConfig()
        self._ewc_regularizers: list[EWC] = []
        self._task_records: list[TaskRecord] = []
        self._forgetting_events: list[str] = []
        self._optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: str,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
    ) -> TaskRecord:
        """Train on a new task with EWC regularization from previous tasks.

        Steps:
        1. Run steps_per_task gradient steps.
        2. At each step: task_loss + sum(ewc.penalty(model) for ewc in past_ewcs).
        3. After training: compute Fisher for this task (becomes new EWC regularizer).
        4. Check for forgetting using the provided validation_loader.

        Returns TaskRecord for this task.
        """
        task_index = len(self._task_records)
        lam = self.ewc_lambda_for_task(task_index)

        self.model.train()
        final_loss = 0.0
        step = 0
        loader_iter = iter(train_loader)

        while step < self.cfg.steps_per_task:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch.get("labels", batch["input_ids"])
            else:
                input_ids, labels = batch[0], batch[1]

            self._optimizer.zero_grad()
            _loss, logits, _ = self.model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = _F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

            # Add EWC penalties from all completed tasks
            total_loss = loss
            for ewc in self._ewc_regularizers:
                total_loss = total_loss + ewc.penalty(self.model)

            total_loss.backward()
            self._optimizer.step()

            final_loss = loss.item()
            step += 1

        logger.info("Task %s training done: final_loss=%.4f", task_id, final_loss)

        # Measure performance on the validation set
        measured_loss: float | None = None
        if validation_loader is not None:
            measured_loss = self.evaluate(validation_loader)
            logger.info("Task %s measured_loss=%.4f", task_id, measured_loss)

        # Forgetting detection: compare vs previous task's recorded loss
        if task_index > 0 and measured_loss is not None:
            prev_record = self._task_records[-1]
            if prev_record.eval_loss is not None:
                prev_eval = prev_record.eval_loss
                if measured_loss > prev_eval * (1.0 + self.cfg.forgetting_threshold):
                    logger.warning(
                        "Forgetting detected for task %s: prev=%.4f current=%.4f",
                        task_id, prev_eval, measured_loss,
                    )
                    self._forgetting_events.append(task_id)

        # Compute Fisher information for this task and register as EWC regularizer
        ewc_cfg = EWCConfig(
            ewc_lambda=lam,
            n_fisher_samples=self.cfg.n_fisher_samples,
        )
        ewc = EWC(self.model, ewc_cfg)
        ewc.compute_fisher(train_loader)
        self._ewc_regularizers.append(ewc)

        record = TaskRecord(
            task_id=task_id,
            final_loss=final_loss,
            eval_loss=measured_loss,
            n_steps=step,
            ewc_lambda=lam,
            fisher_computed=True,
        )
        self._task_records.append(record)
        return record

    def evaluate(self, data_loader: DataLoader) -> float:
        """Compute mean loss over data_loader. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    labels = batch.get("labels", batch["input_ids"])
                else:
                    input_ids, labels = batch[0], batch[1]

                _loss, logits, _ = self.model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = _F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                n_batches += 1

        self.model.train()
        return total_loss / max(1, n_batches)

    def get_report(self) -> ContinualReport:
        """Return summary of all tasks trained so far."""
        return ContinualReport(
            task_records=list(self._task_records),
            forgetting_events=list(self._forgetting_events),
        )

    def ewc_lambda_for_task(self, task_index: int) -> float:
        """Compute EWC lambda for task at given index.

        lambda = base_ewc_lambda * lambda_growth^task_index
        """
        return self.cfg.base_ewc_lambda * (self.cfg.lambda_growth ** task_index)
