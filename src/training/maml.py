"""MAML (Model-Agnostic Meta-Learning) for Aurelius LLM."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and data structures
# ---------------------------------------------------------------------------

@dataclass
class MAMLConfig:
    """Configuration for MAML meta-learning."""

    n_inner_steps: int = 5
    inner_lr: float = 0.01
    meta_lr: float = 1e-3
    n_tasks: int = 4
    first_order: bool = True      # FOMAML approximation
    task_batch_size: int = 8


@dataclass
class Task:
    """A single meta-learning task."""

    support_ids: Tensor   # few-shot examples  (B, T)
    query_ids: Tensor     # evaluation set     (B, T)
    task_id: str = ""


# ---------------------------------------------------------------------------
# Core functional utilities
# ---------------------------------------------------------------------------

def compute_task_loss(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Cross-entropy next-token prediction loss, shifted by 1.

    Args:
        model: Language model whose forward returns (loss, logits, past_kv).
        input_ids: (B, T) token indices.

    Returns:
        Scalar loss tensor.
    """
    _, logits, _ = model(input_ids)
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss


def inner_loop(
    model: nn.Module,
    task: Task,
    config: MAMLConfig,
) -> dict[str, Tensor]:
    """Perform n_inner_steps gradient updates on the support set.

    Uses torch.autograd.grad to compute gradients and manually applies SGD:
        adapted_param = param - inner_lr * grad

    Does NOT permanently modify the model — saves and restores weights.

    Args:
        model: The language model.
        task: Task containing support_ids.
        config: MAMLConfig with inner_lr and n_inner_steps.

    Returns:
        Dict mapping parameter name -> adapted parameter tensor.
    """
    # Snapshot original weights so we can restore after adaptation
    original_state: dict[str, Tensor] = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    if config.n_inner_steps == 0:
        return original_state

    for _ in range(config.n_inner_steps):
        # Zero any lingering gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        loss = compute_task_loss(model, task.support_ids)

        # Compute gradients wrt all trainable parameters
        params_list = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss,
            params_list,
            create_graph=not config.first_order,
            allow_unused=True,
        )

        # Apply manual SGD update in-place (no grad tracking)
        with torch.no_grad():
            for param, grad in zip(params_list, grads):
                if grad is not None:
                    param.data.sub_(config.inner_lr * grad)

    # Capture the adapted parameter values
    adapted: dict[str, Tensor] = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Restore model to original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])

    return adapted



def apply_adapted_params(model: nn.Module, adapted_params: dict[str, Tensor]) -> None:
    """Temporarily apply adapted params to the model (in-place, no grad)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in adapted_params:
                param.data.copy_(adapted_params[name].detach())


def restore_params(model: nn.Module, original_params: dict[str, Tensor]) -> None:
    """Restore original params to the model (in-place, no grad)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])


# ---------------------------------------------------------------------------
# MAMLTrainer
# ---------------------------------------------------------------------------

class MAMLTrainer:
    """Implements the MAML outer loop over a batch of tasks.

    Args:
        model: Language model compatible with (loss, logits, pkv) = model(input_ids).
        meta_optimizer: Outer-loop optimizer (operates on model.parameters()).
        config: MAMLConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_optimizer,
        config: MAMLConfig,
    ) -> None:
        self.model = model
        self.meta_optimizer = meta_optimizer
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def meta_train_step(self, tasks: list[Task]) -> dict:
        """MAML outer loop over a list of tasks.

        1. Save original params.
        2. For each task: run inner_loop on support → evaluate on query with
           adapted params → accumulate query loss.
        3. Restore original params.
        4. Meta backward on sum of query losses, meta_optimizer step.

        Args:
            tasks: List of Task objects.

        Returns:
            Dict with keys "meta_loss", "mean_task_loss", "n_tasks".
        """
        cfg = self.config

        # 1. Save original params
        original_params: dict[str, Tensor] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        task_losses: list[Tensor] = []

        for task in tasks:
            # 2a. Inner loop — get adapted params
            adapted = inner_loop(self.model, task, cfg)

            # Restore to original before applying adapted (inner_loop may leave
            # model in a partially modified state from the leaf-swap trick)
            restore_params(self.model, original_params)

            # 2b. Apply adapted params and evaluate on query set
            apply_adapted_params(self.model, adapted)
            query_loss = compute_task_loss(self.model, task.query_ids)
            task_losses.append(query_loss)

            # Restore after each task evaluation
            restore_params(self.model, original_params)

        # 3. Restore original params (already done in loop, but be explicit)
        restore_params(self.model, original_params)

        # 4. Meta backward on sum of query losses
        meta_loss = torch.stack(task_losses).sum()
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        mean_task_loss = meta_loss.item() / max(1, len(tasks))

        return {
            "meta_loss": meta_loss.item(),
            "mean_task_loss": mean_task_loss,
            "n_tasks": len(tasks),
        }

    def adapt(self, support_ids: Tensor, n_steps: int | None = None) -> dict[str, Tensor]:
        """Adapt to a new task using the support set.

        Args:
            support_ids: (B, T) support token ids.
            n_steps: Number of inner steps (defaults to config.n_inner_steps).

        Returns:
            Dict of adapted parameter tensors.
        """
        cfg = self.config
        task = Task(support_ids=support_ids, query_ids=support_ids)
        if n_steps is not None:
            # Temporarily override n_inner_steps
            import dataclasses
            cfg = dataclasses.replace(cfg, n_inner_steps=n_steps)
        adapted = inner_loop(self.model, task, cfg)
        # Restore model to original state (inner_loop may have modified it)
        original_params: dict[str, Tensor] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        restore_params(self.model, original_params)
        return adapted

    def evaluate_adapted(
        self,
        adapted_params: dict[str, Tensor],
        query_ids: Tensor,
    ) -> float:
        """Evaluate model with adapted params on query set.

        Args:
            adapted_params: Dict of adapted parameter tensors.
            query_ids: (B, T) query token ids.

        Returns:
            Scalar loss as Python float.
        """
        # Save original params
        original_params: dict[str, Tensor] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        apply_adapted_params(self.model, adapted_params)
        with torch.no_grad():
            loss = compute_task_loss(self.model, query_ids)
        restore_params(self.model, original_params)
        return loss.item()
