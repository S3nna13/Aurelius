"""Activation steering and representation engineering: compute steering vectors, apply them at inference time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""

    layer_idx: int = 12
    """Which layer to steer."""

    coeff: float = 20.0
    """Steering coefficient."""

    mode: str = "add"
    """Steering mode: 'add' | 'subtract' | 'project_out'."""

    normalize: bool = True
    """L2-normalize steering vector before applying."""


@torch.no_grad()
def compute_mean_activation(
    model: nn.Module,
    input_ids_list: list[Tensor],
    layer_idx: int,
) -> Tensor:
    """Compute mean hidden-state activation at a given layer across a list of prompts.

    Runs each prompt through the model, collecting hidden states at ``layer_idx``
    via a forward hook on ``model.layers[layer_idx]``.  Mean-pools over the time
    dimension for each prompt, then returns the mean over all prompts.

    Args:
        model: The transformer model (must expose ``model.layers``).
        input_ids_list: List of ``(B, T)`` token-id tensors (one per prompt).
        layer_idx: Index into ``model.layers`` from which to capture output.

    Returns:
        Tensor of shape ``(d_model,)`` — mean activation across all prompts.
    """
    collected: list[Tensor] = []

    def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)):
            hidden = output[0]
        else:
            hidden = output
        # hidden: (B, T, D) — mean-pool over T
        collected.append(hidden.detach().mean(dim=1))  # (B, D)

    target_layer = model.layers[layer_idx]
    handle = target_layer.register_forward_hook(_hook)
    try:
        for input_ids in input_ids_list:
            model(input_ids)
    finally:
        handle.remove()

    # collected: list of (B, D) tensors; cat then mean over all prompts * batches
    all_acts = torch.cat(collected, dim=0)  # (N_total, D)
    return all_acts.mean(dim=0)  # (D,)


def compute_steering_vector(
    positive_acts: Tensor,
    negative_acts: Tensor,
) -> Tensor:
    """Compute an L2-normalized steering vector from positive and negative concept activations.

    Args:
        positive_acts: ``(N_pos, d_model)`` — mean activations for the positive concept.
        negative_acts: ``(N_neg, d_model)`` — mean activations for the negative concept.

    Returns:
        Tensor of shape ``(d_model,)`` — L2-normalized direction vector.
    """
    direction = positive_acts.mean(dim=0) - negative_acts.mean(dim=0)
    direction = F.normalize(direction, dim=0)
    return direction


class SteeringHook:
    """Context manager that patches hidden states during a model forward pass.

    The hook modifies the output of ``model.layers[config.layer_idx]`` according to
    the chosen mode:

    * ``"add"``: ``output = output + coeff * direction``
    * ``"subtract"``: ``output = output - coeff * direction``
    * ``"project_out"``: removes the component of ``output`` along ``direction``

    Args:
        steering_vector: ``(d_model,)`` direction vector (will be L2-normalised if
            ``config.normalize`` is ``True``).
        config: ``SteeringConfig`` specifying layer, coefficient, mode and normalization.
    """

    def __init__(self, steering_vector: Tensor, config: SteeringConfig) -> None:
        self.config = config
        if config.normalize:
            self.direction = F.normalize(steering_vector, dim=0)
        else:
            self.direction = steering_vector
        self._handle: Any = None
        self._model: nn.Module | None = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def register(self, model: nn.Module) -> "SteeringHook":
        """Register the forward hook on *model*.  Returns self for chaining."""
        self._model = model
        target_layer = model.layers[self.config.layer_idx]
        self._handle = target_layer.register_forward_hook(self._hook_fn)
        return self

    def remove(self) -> None:
        """Remove the registered forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "SteeringHook":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Internal hook
    # ------------------------------------------------------------------

    def _hook_fn(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        """Modify the layer output according to the configured mode."""
        is_tuple = isinstance(output, (tuple, list))
        if is_tuple:
            hidden = output[0]
        else:
            hidden = output

        direction = self.direction.to(hidden.device, hidden.dtype)
        coeff = self.config.coeff

        if self.config.mode == "add":
            hidden = hidden + coeff * direction
        elif self.config.mode == "subtract":
            hidden = hidden - coeff * direction
        elif self.config.mode == "project_out":
            # Remove the component along direction: h - (h · d) * d
            # direction: (D,), hidden: (B, T, D)
            proj = (hidden @ direction).unsqueeze(-1) * direction  # (B, T, D)
            hidden = hidden - proj
        else:
            raise ValueError(f"Unknown steering mode: {self.config.mode!r}")

        if is_tuple:
            return (hidden,) + tuple(output[1:])
        return hidden


def apply_steering(
    model: nn.Module,
    input_ids: Tensor,
    steering_vector: Tensor,
    config: SteeringConfig,
) -> Tensor:
    """Run model forward with activation steering applied, returning logits.

    Args:
        model: The transformer model.
        input_ids: ``(B, T)`` token-id tensor.
        steering_vector: ``(d_model,)`` direction to steer towards.
        config: Steering configuration.

    Returns:
        Logits tensor of shape ``(B, T, V)``.
    """
    hook = SteeringHook(steering_vector, config)
    hook.register(model)
    with hook:
        _loss, logits, _pkv = model(input_ids)
    return logits


def measure_steering_effect(
    model: nn.Module,
    input_ids: Tensor,
    steering_vector: Tensor,
    config: SteeringConfig,
) -> dict[str, float]:
    """Quantify the effect of activation steering by comparing steered vs. baseline logits.

    Runs the model twice — with and without the steering hook — and computes three
    scalar metrics over all ``B * T`` token positions:

    * ``"logit_kl"``  — KL divergence of ``softmax(logits_steered)`` vs ``softmax(logits_base)``, averaged over tokens.
    * ``"logit_mse"`` — Mean-squared error between the raw logit tensors.
    * ``"max_logit_diff"`` — Maximum absolute difference between the raw logit tensors.

    Args:
        model: The transformer model.
        input_ids: ``(B, T)`` token-id tensor.
        steering_vector: ``(d_model,)`` direction to steer towards.
        config: Steering configuration.

    Returns:
        Dictionary with keys ``"logit_kl"``, ``"logit_mse"``, ``"max_logit_diff"``.
    """
    with torch.no_grad():
        _loss_base, logits_base, _pkv = model(input_ids)
        logits_steered = apply_steering(model, input_ids, steering_vector, config)

    # Flatten B*T
    logits_base_flat = logits_base.view(-1, logits_base.shape[-1])       # (B*T, V)
    logits_steered_flat = logits_steered.view(-1, logits_steered.shape[-1])  # (B*T, V)

    probs_base = F.softmax(logits_base_flat, dim=-1)
    probs_steered = F.softmax(logits_steered_flat, dim=-1)

    # KL(steered || base) averaged over tokens
    # F.kl_div expects log-input, target; reduction='batchmean' averages over batch
    log_probs_steered = torch.log(probs_steered + 1e-10)
    kl = F.kl_div(log_probs_steered, probs_base, reduction="batchmean")

    mse = F.mse_loss(logits_steered_flat, logits_base_flat)
    max_diff = (logits_steered_flat - logits_base_flat).abs().max()

    return {
        "logit_kl": float(kl),
        "logit_mse": float(mse),
        "max_logit_diff": float(max_diff),
    }
