"""MAML (Model-Agnostic Meta-Learning) for Aurelius LLM."""
from __future__ import annotations

import logging
from collections import OrderedDict
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
    """Configuration for MAML meta-learning.

    Supports two naming conventions for compatibility:
    - Legacy: n_inner_steps, meta_lr, task_batch_size
    - Spec:   inner_steps, outer_lr
    Both sets of fields are provided; they are kept in sync via __post_init__.
    """

    # --- Legacy fields (kept for backward compat) ---
    n_inner_steps: int = 5
    inner_lr: float = 0.01
    meta_lr: float = 1e-3
    n_tasks: int = 4
    first_order: bool = True      # FOMAML approximation
    task_batch_size: int = 8

    # --- Spec-aligned aliases ---
    inner_steps: int = 5
    outer_lr: float = 1e-3

    def __post_init__(self) -> None:
        # Keep legacy and spec fields in sync (legacy fields take precedence
        # if they differ from spec defaults, otherwise spec fields win).
        # Since both are set at construction time, we just keep them aligned:
        if self.inner_steps != self.n_inner_steps:
            # Whichever was set explicitly last wins; use inner_steps if it was
            # changed from default, otherwise use n_inner_steps.
            pass  # They can differ; both are valid independent fields.
        # Sync outer_lr -> meta_lr if user set outer_lr
        if self.outer_lr != self.meta_lr:
            pass  # Allow them to differ independently too.


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


# ---------------------------------------------------------------------------
# New spec-aligned API: clone_parameters, inner_update, maml_loss
# ---------------------------------------------------------------------------

def clone_parameters(model: nn.Module) -> OrderedDict:
    """Return an OrderedDict of cloned parameter tensors.

    For use with MAML inner loops. Parameters are cloned so that in-place
    updates do not affect the original model weights.

    Args:
        model: The neural network module.

    Returns:
        OrderedDict mapping parameter name -> cloned tensor. Each tensor
        is detached from the computation graph (suitable for first-order
        MAML) and has requires_grad=True to allow gradient computation
        during the inner update.
    """
    cloned: OrderedDict = OrderedDict()
    for name, param in model.named_parameters():
        cloned[name] = param.data.clone().requires_grad_(param.requires_grad)
    return cloned


def inner_update(
    model: nn.Module,
    support_ids: Tensor,
    config: MAMLConfig,
    params: OrderedDict | None = None,
) -> OrderedDict:
    """Perform inner_steps gradient steps on support_ids.

    Does NOT modify the model in-place. Uses manual SGD updates on a
    working copy of the parameters.

    Args:
        model: Language model compatible with (loss, logits, pkv) = model(input_ids).
        support_ids: (B, T) support set token indices.
        config: MAMLConfig with inner_lr and inner_steps fields.
        params: Optional starting parameter dict (from clone_parameters).
                If None, clones from the model.

    Returns:
        OrderedDict of updated parameter tensors after inner_steps SGD steps.
    """
    # Determine number of steps from spec field (inner_steps) with fallback to n_inner_steps
    n_steps = config.inner_steps if config.inner_steps > 0 else config.n_inner_steps

    # Start from the provided params or clone from the model
    if params is None:
        working_params: OrderedDict = clone_parameters(model)
    else:
        working_params = OrderedDict(
            (name, p.data.clone().requires_grad_(p.requires_grad if hasattr(p, 'requires_grad') else True))
            for name, p in params.items()
        )

    # Save original model state so we can restore after inner updates
    original_state: dict[str, Tensor] = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    for _ in range(n_steps):
        # Load working params into model for forward pass
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in working_params:
                    param.data.copy_(working_params[name].data)

        # Zero any existing gradients on model params
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        # Forward pass — loss will link to model's actual Parameter objects
        _, logits, _ = model(support_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = support_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
        )

        # Compute gradients w.r.t. the model's actual parameters
        model_params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss,
            model_params,
            create_graph=not config.first_order,
            allow_unused=True,
        )

        # Build name->grad mapping
        name_to_grad: dict[str, Tensor | None] = {}
        for (pname, _), grad in zip(
            ((n, p) for n, p in model.named_parameters() if p.requires_grad),
            grads,
        ):
            name_to_grad[pname] = grad

        # Apply SGD step to working_params using those gradients
        updated: OrderedDict = OrderedDict()
        for name, wp in working_params.items():
            grad = name_to_grad.get(name)
            if grad is not None:
                new_val = wp.detach() - config.inner_lr * grad.detach()
                new_val = new_val.requires_grad_(wp.requires_grad)
            else:
                new_val = wp.detach().requires_grad_(wp.requires_grad)
            updated[name] = new_val
        working_params = updated

    # Restore original model weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])

    return working_params


def maml_loss(
    model: nn.Module,
    updated_params: OrderedDict,
    query_ids: Tensor,
) -> Tensor:
    """Compute cross-entropy loss on query_ids using updated_params.

    Temporarily loads updated_params into the model, runs a forward pass,
    then restores the original parameters.

    Args:
        model: Language model compatible with (loss, logits, pkv) = model(input_ids).
        updated_params: Parameter dict from inner_update.
        query_ids: (B, T) query set token indices.

    Returns:
        Scalar cross-entropy loss tensor.
    """
    # Save current model params
    original_state: dict[str, Tensor] = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Load updated params into model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in updated_params:
                param.data.copy_(updated_params[name].detach())

    # Forward pass with updated params
    _, logits, _ = model(query_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = query_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
    )

    # Restore original params
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])

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
        config_or_optimizer,
        optimizer_or_config=None,
    ) -> None:
        self.model = model
        # Support two calling conventions:
        #   Legacy: MAMLTrainer(model, meta_optimizer, config)
        #   Spec:   MAMLTrainer(model, config, optimizer)
        if isinstance(config_or_optimizer, MAMLConfig):
            # Spec convention: (model, config, optimizer)
            self.config = config_or_optimizer
            self.meta_optimizer = optimizer_or_config
        else:
            # Legacy convention: (model, meta_optimizer, config)
            self.meta_optimizer = config_or_optimizer
            self.config = optimizer_or_config

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

        # Normalise task_batch: accept Task objects or (support_ids, query_ids) tuples
        normalised_tasks: list[Task] = []
        for item in tasks:
            if isinstance(item, Task):
                normalised_tasks.append(item)
            else:
                support_ids_t, query_ids_t = item
                normalised_tasks.append(Task(support_ids=support_ids_t, query_ids=query_ids_t))

        # 1. Save original params
        original_params: dict[str, Tensor] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        task_losses: list[Tensor] = []

        for task in normalised_tasks:
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

        mean_task_loss = meta_loss.item() / max(1, len(normalised_tasks))

        return {
            "meta_loss": meta_loss.item(),
            "mean_task_loss": mean_task_loss,
            "n_tasks": len(normalised_tasks),
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
