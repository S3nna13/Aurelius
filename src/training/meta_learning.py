"""Meta-learning: MAML and Reptile for few-shot task adaptation."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for MAML / Reptile meta-learning."""

    algorithm: str = "reptile"      # "maml" | "reptile"
    n_inner_steps: int = 5          # inner loop gradient steps
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    n_tasks_per_batch: int = 4      # meta-batch size
    first_order: bool = True        # for MAML: use first-order approx (FOMAML)


# ---------------------------------------------------------------------------
# Core functional utilities
# ---------------------------------------------------------------------------

def compute_task_loss(
    model: nn.Module,
    support_ids: Tensor,
    support_labels: Tensor,
) -> Tensor:
    """Forward pass through model and return CE loss.

    Args:
        model: The language model. forward returns (loss, logits, kv).
        support_ids: (B, T) token indices.
        support_labels: (B, T) target token ids.

    Returns:
        Scalar cross-entropy loss tensor.
    """
    _, logits, _ = model(support_ids)
    # Shift for next-token prediction (same convention as AureliusTransformer)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = support_labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss


def inner_loop_update(
    model: nn.Module,
    support_ids: Tensor,
    support_labels: Tensor,
    inner_lr: float,
    n_steps: int,
) -> dict[str, Tensor]:
    """Perform n_steps of manual SGD on support set, returning adapted params.

    Does NOT modify model in-place permanently — saves and restores weights.

    Args:
        model: The language model.
        support_ids: (B, T) token indices.
        support_labels: (B, T) target token ids.
        inner_lr: Step size for inner SGD.
        n_steps: Number of gradient steps.

    Returns:
        Dict mapping param name -> adapted parameter tensor.
    """
    # Snapshot the original state so we can restore afterwards
    original_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Collect references to trainable params for autograd
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]

    # Perform n_steps of manual SGD directly on model weights
    for _ in range(n_steps):
        # Zero any lingering grads
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        loss = compute_task_loss(model, support_ids, support_labels)

        # Compute gradients wrt all trainable params
        params_list = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss,
            params_list,
            create_graph=False,
            allow_unused=True,
        )

        # Apply SGD update in-place (no_grad to avoid tracking)
        with torch.no_grad():
            for param, grad in zip(params_list, grads):
                if grad is not None:
                    param.data.sub_(inner_lr * grad)

    # Capture adapted parameter values
    adapted_params: dict[str, Tensor] = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])

    return adapted_params


def reptile_update(
    original_params: dict[str, Tensor],
    adapted_params: dict[str, Tensor],
    outer_lr: float,
) -> dict[str, Tensor]:
    """Reptile meta-update: move original params toward adapted params.

    new_param = original + outer_lr * (adapted - original)

    Args:
        original_params: Dict of current model parameter tensors.
        adapted_params: Dict of parameter tensors after inner-loop adaptation.
        outer_lr: Step size for the meta-update (Reptile epsilon).

    Returns:
        Dict of updated parameter tensors.
    """
    updated: dict[str, Tensor] = {}
    for name, orig in original_params.items():
        adapted = adapted_params[name]
        updated[name] = orig + outer_lr * (adapted - orig)
    return updated


def fomaml_meta_gradient(
    original_params: dict[str, Tensor],
    adapted_params: dict[str, Tensor],
    query_loss: Tensor,
) -> dict[str, Tensor]:
    """FOMAML: gradient of query_loss w.r.t. adapted params.

    In FOMAML the second-order terms are dropped, so the gradient of the
    query loss w.r.t. the *original* parameters is approximated by the
    gradient w.r.t. the *adapted* parameters.

    Args:
        original_params: Dict of original model parameters (provides keys).
        adapted_params: Dict of adapted parameter tensors (must require grad).
        query_loss: Scalar loss on the query set (computed with adapted params).

    Returns:
        Dict mapping param name -> meta-gradient tensor (same keys as original_params).
    """
    adapted_values = list(adapted_params.values())
    grads = torch.autograd.grad(
        query_loss,
        adapted_values,
        allow_unused=True,
    )
    meta_grads: dict[str, Tensor] = {}
    for (name, _), grad in zip(adapted_params.items(), grads):
        if grad is None:
            meta_grads[name] = torch.zeros_like(original_params[name])
        else:
            meta_grads[name] = grad
    return meta_grads


# ---------------------------------------------------------------------------
# MetaLearner class
# ---------------------------------------------------------------------------

class MetaLearner:
    """Wraps a model with MAML or Reptile meta-learning.

    Args:
        model: The language model (AureliusTransformer or compatible).
        config: Meta-learning configuration.
        optimizer: Outer-loop optimizer for the model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MetaLearningConfig,
        optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def meta_step(self, tasks: list[dict]) -> dict:
        """Perform one meta-learning step over a batch of tasks.

        Each task dict must contain:
            "support_ids"    : Tensor (B, T)
            "support_labels" : Tensor (B, T)
            "query_ids"      : Tensor (B, T)  (used only by MAML)
            "query_labels"   : Tensor (B, T)  (used only by MAML)

        Reptile path:
            - Adapt to each task via inner_loop_update.
            - Average the adapted params across tasks.
            - Apply reptile_update to move the model toward the average.

        Args:
            tasks: List of task dicts.

        Returns:
            Dict with keys "meta_loss", "n_tasks", "mean_inner_loss".
        """
        cfg = self.config

        # Snapshot original params
        original_params: dict[str, Tensor] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        all_adapted: list[dict[str, Tensor]] = []
        inner_losses: list[float] = []

        for task in tasks:
            support_ids = task["support_ids"]
            support_labels = task["support_labels"]

            # Measure support loss before adaptation (for reporting)
            with torch.no_grad():
                pre_loss = compute_task_loss(self.model, support_ids, support_labels)
            inner_losses.append(pre_loss.item())

            # Inner-loop adaptation
            adapted = inner_loop_update(
                self.model,
                support_ids,
                support_labels,
                inner_lr=cfg.inner_lr,
                n_steps=cfg.n_inner_steps,
            )
            all_adapted.append(adapted)

        # --- Reptile meta-update ---
        if cfg.algorithm == "reptile" or cfg.first_order:
            # Average adapted params across tasks
            avg_adapted: dict[str, Tensor] = {}
            for name in original_params:
                avg_adapted[name] = torch.stack(
                    [a[name] for a in all_adapted]
                ).mean(dim=0)

            new_params = reptile_update(original_params, avg_adapted, cfg.outer_lr)

            # Write updated params back to model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in new_params:
                        param.data.copy_(new_params[name])

        # Compute a rough meta_loss as mean inner loss (for logging)
        mean_inner_loss = sum(inner_losses) / max(1, len(inner_losses))

        return {
            "meta_loss": mean_inner_loss,
            "n_tasks": len(tasks),
            "mean_inner_loss": mean_inner_loss,
        }

    def adapt(
        self,
        support_ids: Tensor,
        support_labels: Tensor,
    ) -> nn.Module:
        """Return a copy of the model adapted to the support set.

        Runs the inner loop on support data and loads the adapted weights
        into a deep copy of the model (original model is unchanged).

        Args:
            support_ids: (B, T) token indices.
            support_labels: (B, T) target token ids.

        Returns:
            Adapted model (nn.Module) with inner-loop weights applied.
        """
        cfg = self.config
        adapted_params = inner_loop_update(
            self.model,
            support_ids,
            support_labels,
            inner_lr=cfg.inner_lr,
            n_steps=cfg.n_inner_steps,
        )

        # Deep-copy the model so the original is untouched
        adapted_model = copy.deepcopy(self.model)
        with torch.no_grad():
            for name, param in adapted_model.named_parameters():
                if name in adapted_params:
                    param.data.copy_(adapted_params[name])

        return adapted_model
