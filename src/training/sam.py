"""SAM (Sharpness-Aware Minimization) optimizer for Aurelius LLM.

Implements SAM as described in Foret et al. (2021) — arXiv:2010.01412.

SAM seeks parameters in flat minima by solving a min-max problem:
    min_w max_{||eps||<=rho} L(w + eps)

The two-step procedure:
    1. First step:  compute gradient, perturb weights to w_hat = w + rho * g/||g||
    2. Second step: compute gradient at w_hat, restore w, apply base optimizer step
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SAMConfig:
    """Configuration for SAM optimizer.

    Attributes:
        rho:      Perturbation neighbourhood radius (default 0.05).
        adaptive: Use Adaptive SAM (ASAM), scales eps by |w| element-wise.
        momentum: Momentum coefficient (informational; applies to base optimizer).
    """

    rho: float = 0.05
    adaptive: bool = False
    momentum: float = 0.9


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all parameter gradients.

    Args:
        model: The neural network module.

    Returns:
        L2 norm of all gradients as a Python float; 0.0 if no gradients exist.
    """
    total_sq = 0.0
    found_any = False
    for param in model.parameters():
        if param.grad is not None:
            total_sq += param.grad.detach().norm().item() ** 2
            found_any = True
    return float(total_sq**0.5) if found_any else 0.0


def perturb_weights(
    model: nn.Module,
    rho: float,
    adaptive: bool = False,
    eps: float = 1e-12,
) -> float:
    """Add SAM perturbation epsilon_hat to model weights in place.

    For each parameter with a gradient:
        epsilon_hat = rho * g / (||g|| + eps)     (standard SAM)
        epsilon_hat = rho * |w| * g / (||g|| + eps)  (adaptive SAM)

    Stores original data in param.data_backup before modifying.

    Args:
        model:    The neural network module (must have gradients already computed).
        rho:      Perturbation radius.
        adaptive: If True, use Adaptive SAM scaling (element-wise |w|).
        eps:      Numerical stability constant.

    Returns:
        The perturbation scale factor (rho / (grad_norm + eps)) as a float.
    """
    # Collect gradients
    grads = []
    params_with_grad = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.detach())
            params_with_grad.append(param)

    # Compute global grad norm
    grad_norm = (
        float(torch.sqrt(sum(g.norm() ** 2 for g in grads) + torch.tensor(0.0)).item())
        if grads
        else 0.0
    )

    scale = rho / (grad_norm + eps)

    # Apply perturbation and back up original data
    for param, grad in zip(params_with_grad, grads):
        # Store original weights
        param.data_backup = param.data.detach().clone()

        if adaptive:
            eps_hat = scale * param.data.abs() * grad
        else:
            eps_hat = scale * grad

        param.data.add_(eps_hat)

    return scale


def restore_weights(model: nn.Module) -> None:
    """Restore model parameter data from param.data_backup.

    Args:
        model: The neural network module (previously perturbed via perturb_weights).
    """
    for param in model.parameters():
        if hasattr(param, "data_backup"):
            param.data.copy_(param.data_backup)
            del param.data_backup


# ---------------------------------------------------------------------------
# Loss computation helper
# ---------------------------------------------------------------------------


def _compute_loss(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Compute cross-entropy next-token prediction loss.

    Uses the plain tuple API: loss, logits, pkv = model(input_ids).
    The returned loss from the model is used directly when available
    (requires model to use internal label-shifting logic), or we compute
    it from logits for next-token prediction.

    Args:
        model:     Language model returning (loss_or_none, logits, past_kv).
        input_ids: (B, T) token indices.

    Returns:
        Scalar loss tensor.
    """
    loss, logits, _ = model(input_ids)
    if loss is not None:
        return loss
    # Compute next-token prediction loss from logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
    )


# ---------------------------------------------------------------------------
# SAMOptimizer
# ---------------------------------------------------------------------------


class SAMOptimizer:
    """SAM two-step optimizer wrapper.

    Wraps any base PyTorch optimizer and implements the SAM update rule:
        1. first_step(loss):  backward + perturb weights + zero_grad
        2. second_step(loss): backward on perturbed weights + restore + base step

    Args:
        base_optimizer: Underlying optimizer (e.g. Adam, SGD).
        model:          The neural network module.
        config:         SAMConfig with rho, adaptive, momentum fields.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        model: nn.Module,
        config: SAMConfig,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.model = model
        self.config = config

    def first_step(self, loss: Tensor) -> Tensor:
        """Perform the first SAM step.

        Computes gradients, perturbs weights toward the worst-case neighbour,
        then zeros the gradients in preparation for the second forward pass.

        Args:
            loss: Scalar loss tensor at the current (unperturbed) weights.

        Returns:
            The detached loss value.
        """
        loss_val = loss.detach().clone()
        self.base_optimizer.zero_grad()
        loss.backward()
        perturb_weights(self.model, self.config.rho, adaptive=self.config.adaptive)
        self.base_optimizer.zero_grad()
        return loss_val

    def second_step(self, loss: Tensor) -> Tensor:
        """Perform the second SAM step.

        Computes gradients at the perturbed weights, restores original weights,
        then applies the base optimizer update using those perturbed-weight gradients.

        Args:
            loss: Scalar loss tensor computed at the perturbed weights.

        Returns:
            The detached loss value.
        """
        loss_val = loss.detach().clone()
        loss.backward()
        restore_weights(self.model)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        return loss_val


# ---------------------------------------------------------------------------
# SAMTrainer
# ---------------------------------------------------------------------------


class SAMTrainer:
    """High-level trainer that runs the full SAM two-step update per batch.

    Args:
        model:          The neural network module.
        config:         SAMConfig instance.
        base_optimizer: Underlying optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SAMConfig,
        base_optimizer: Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.sam = SAMOptimizer(base_optimizer, model, config)

    def train_step(self, input_ids: Tensor) -> dict[str, float]:
        """Run one full SAM training step.

        Args:
            input_ids: (B, T) integer token tensor.

        Returns:
            dict with keys:
                first_loss:  loss at original weights (float).
                second_loss: loss at perturbed weights (float).
                grad_norm:   L2 norm of gradients after first backward (float).
        """
        self.model.train()

        # --- First step ---
        self.sam.base_optimizer.zero_grad()
        first_loss_tensor = _compute_loss(self.model, input_ids)
        first_loss_tensor.backward()
        grad_norm = compute_grad_norm(self.model)
        first_loss_val = first_loss_tensor.item()

        perturb_weights(self.model, self.config.rho, adaptive=self.config.adaptive)
        self.sam.base_optimizer.zero_grad()

        # --- Second step ---
        second_loss_tensor = _compute_loss(self.model, input_ids)
        second_loss_val = second_loss_tensor.item()
        second_loss_tensor.backward()
        restore_weights(self.model)
        self.sam.base_optimizer.step()
        self.sam.base_optimizer.zero_grad()

        return {
            "first_loss": first_loss_val,
            "second_loss": second_loss_val,
            "grad_norm": grad_norm,
        }
