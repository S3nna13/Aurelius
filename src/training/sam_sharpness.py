"""Sharpness-Aware Minimization sharpness utilities.

Implements the local sharpness quantity from Foret et al. (2021):

    max_{||epsilon||_2 <= rho} L(w + epsilon)

using the paper's first-order approximation

    epsilon(w) = rho * g(w) / ||g(w)||_2

where g(w) = grad_w L(w).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


LossFn = Callable[[nn.Module], torch.Tensor]


@dataclass(frozen=True)
class SAMSharpnessConfig:
    """Configuration for the SAM sharpness approximation."""

    rho: float = 0.05
    stability_eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {self.rho}")
        if self.stability_eps <= 0.0:
            raise ValueError(f"stability_eps must be positive, got {self.stability_eps}")


@dataclass(frozen=True)
class SAMSharpnessResult:
    """Paper-faithful SAM quantities evaluated at w and w + epsilon(w)."""

    L_w: torch.Tensor
    L_w_plus_epsilon_w: torch.Tensor
    sharpness: torch.Tensor
    grad_norm_w: torch.Tensor
    epsilon_w: dict[str, torch.Tensor]


class _LossModule(nn.Module):
    """Wraps a model loss closure so functional_call can swap parameters."""

    def __init__(self, model: nn.Module, loss_fn: LossFn) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self) -> torch.Tensor:
        return self.loss_fn(self.model)


def _require_scalar_loss(loss: torch.Tensor, name: str) -> None:
    if loss.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(loss.shape)}")


def _named_trainable_parameters(model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def _l2_norm(
    tensors: tuple[torch.Tensor | None, ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    squared_norm = torch.zeros((), device=reference.device, dtype=reference.dtype)
    for tensor in tensors:
        if tensor is not None:
            squared_norm = squared_norm + tensor.detach().pow(2).sum()
    return squared_norm.sqrt()


def sam_sharpness(
    model: nn.Module,
    loss_fn: LossFn,
    config: SAMSharpnessConfig | None = None,
) -> SAMSharpnessResult:
    """Evaluate L(w), epsilon(w), and L(w + epsilon(w)).

    Args:
        model: Module whose parameters define w.
        loss_fn: Callable returning a scalar loss for the supplied model.
        config: SAM sharpness hyperparameters.

    Returns:
        SAMSharpnessResult containing the paper's key quantities.
    """
    config = config or SAMSharpnessConfig()

    named_trainable_parameters = _named_trainable_parameters(model)
    if not named_trainable_parameters:
        raise ValueError("model must have at least one trainable parameter")

    parameters_w = [parameter for _, parameter in named_trainable_parameters]
    L_w = loss_fn(model)
    _require_scalar_loss(L_w, "L_w")

    g_w = torch.autograd.grad(L_w, parameters_w, allow_unused=True)
    grad_norm_w = _l2_norm(g_w, L_w)
    scale = config.rho / (grad_norm_w + config.stability_eps)

    epsilon_w: OrderedDict[str, torch.Tensor] = OrderedDict()
    model_state = OrderedDict()
    for name, parameter in model.named_parameters():
        model_state[f"model.{name}"] = parameter
    for name, buffer in model.named_buffers():
        model_state[f"model.{name}"] = buffer

    for (name, parameter), grad_w in zip(named_trainable_parameters, g_w):
        if grad_w is None:
            epsilon_w[name] = torch.zeros_like(parameter)
        else:
            epsilon_w[name] = (
                scale.to(device=parameter.device, dtype=parameter.dtype) * grad_w.detach()
            )
        model_state[f"model.{name}"] = parameter + epsilon_w[name]

    loss_module = _LossModule(model, loss_fn)
    L_w_plus_epsilon_w = torch.func.functional_call(loss_module, model_state, ())
    _require_scalar_loss(L_w_plus_epsilon_w, "L_w_plus_epsilon_w")

    return SAMSharpnessResult(
        L_w=L_w,
        L_w_plus_epsilon_w=L_w_plus_epsilon_w,
        sharpness=L_w_plus_epsilon_w.detach() - L_w.detach(),
        grad_norm_w=grad_norm_w.detach(),
        epsilon_w=dict(epsilon_w),
    )


def sam_sharpness_loss(
    model: nn.Module,
    loss_fn: LossFn,
    config: SAMSharpnessConfig | None = None,
) -> torch.Tensor:
    """Return the SAM objective L(w + epsilon(w)).

    This is the differentiable surrogate used by SAM, with epsilon(w) treated as a
    stop-gradient perturbation built from the current gradient g(w).
    """
    return sam_sharpness(model, loss_fn, config=config).L_w_plus_epsilon_w
