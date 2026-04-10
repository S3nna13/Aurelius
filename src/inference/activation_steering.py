"""Activation steering / representation engineering for controlling model behavior.

Implements steering vectors (also known as representation engineering) that add
a direction to hidden states at a specific layer to steer model outputs toward
desired behaviors.

References:
    Zou et al. 2023 "Representation Engineering: A Top-Down Approach to AI Transparency"
    Turner et al. 2023 "Activation Addition: Steering Language Models Without Optimization"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""
    layer_idx: int = 0
    coeff: float = 1.0
    normalize: bool = True


# ---------------------------------------------------------------------------
# compute_steering_vector
# ---------------------------------------------------------------------------

def compute_steering_vector(
    model: nn.Module,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute a steering direction as mean(positive) - mean(negative) hidden states.

    Hidden states are extracted at the output of the specified layer at the
    last token position.

    Args:
        model: AureliusTransformer instance.
        positive_ids: (B, S) token ids for "positive" examples.
        negative_ids: (B, S) token ids for "negative" examples.
        layer_idx: Which transformer layer to extract hidden states from.

    Returns:
        Steering vector of shape (d_model,).
    """
    pos_activations: List[torch.Tensor] = []
    neg_activations: List[torch.Tensor] = []

    def _make_hook(storage: List[torch.Tensor]):
        def hook(module, input, output):
            # TransformerBlock returns (hidden, kv) — take hidden
            hidden = output[0] if isinstance(output, tuple) else output
            # last token position, mean over batch
            storage.append(hidden[:, -1, :].detach())
        return hook

    with torch.no_grad():
        # --- positive pass ---
        h_pos = _make_hook(pos_activations)
        handle = model.layers[layer_idx].register_forward_hook(h_pos)
        try:
            model(positive_ids)
        finally:
            handle.remove()

        # --- negative pass ---
        h_neg = _make_hook(neg_activations)
        handle = model.layers[layer_idx].register_forward_hook(h_neg)
        try:
            model(negative_ids)
        finally:
            handle.remove()

    mean_pos = torch.cat(pos_activations, dim=0).mean(dim=0)  # (d_model,)
    mean_neg = torch.cat(neg_activations, dim=0).mean(dim=0)  # (d_model,)
    return mean_pos - mean_neg


# ---------------------------------------------------------------------------
# SteeringHook
# ---------------------------------------------------------------------------

class SteeringHook:
    """Forward hook that adds a steering vector to hidden states at a given layer."""

    def __init__(self, steering_vector: torch.Tensor, config: SteeringConfig) -> None:
        self.steering_vector = steering_vector  # (d_model,)
        self.config = config

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output,
    ):
        """Add scaled (optionally normalized) steering vector to hidden states."""
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        direction = self.steering_vector
        if self.config.normalize:
            norm = direction.norm()
            if norm > 1e-8:
                direction = direction / norm

        # Broadcast over batch and sequence dimensions
        hidden = hidden + self.config.coeff * direction.to(hidden.device)

        if is_tuple:
            return (hidden,) + output[1:]
        return hidden

    def register(self, model: nn.Module) -> RemovableHandle:
        """Attach this hook to model.layers[config.layer_idx].

        Returns:
            A RemovableHandle that can be used to remove the hook later.
        """
        return model.layers[self.config.layer_idx].register_forward_hook(self)


# ---------------------------------------------------------------------------
# ActivationSteerer
# ---------------------------------------------------------------------------

class ActivationSteerer:
    """Manages steering vectors and steered generation."""

    def __init__(self, model: nn.Module, config: SteeringConfig) -> None:
        self.model = model
        self.config = config
        self._steering_vector: torch.Tensor | None = None
        self._handles: List[RemovableHandle] = []

    def add_steering_vector(self, vector: torch.Tensor) -> None:
        """Store the steering vector (normalized if config.normalize).

        Args:
            vector: Shape (d_model,).
        """
        if self.config.normalize:
            norm = vector.norm()
            if norm > 1e-8:
                vector = vector / norm
        self._steering_vector = vector

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy generation with the steering hook active.

        Args:
            input_ids: (1, S) prompt token ids.
            max_new_tokens: Number of tokens to generate.

        Returns:
            Generated token ids of shape (1, max_new_tokens).
        """
        if self._steering_vector is None:
            raise ValueError("Call add_steering_vector() before generate().")

        # Build hook — pass normalize=False here because vector is already
        # normalized (or not) per user config; we just want raw coeff scaling.
        hook_cfg = SteeringConfig(
            layer_idx=self.config.layer_idx,
            coeff=self.config.coeff,
            normalize=False,  # already applied in add_steering_vector
        )
        hook = SteeringHook(self._steering_vector, hook_cfg)
        handle = hook.register(self.model)
        self._handles.append(handle)

        generated = []
        cur_ids = input_ids.clone()

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    _, logits, _ = self.model(cur_ids)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
                    generated.append(next_token)
                    cur_ids = torch.cat([cur_ids, next_token], dim=1)
        finally:
            handle.remove()
            # Remove the handle we just added from the list
            if handle in self._handles:
                self._handles.remove(handle)

        return torch.cat(generated, dim=1)  # (1, max_new_tokens)

    def remove_steering(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# contrastive_activation_addition
# ---------------------------------------------------------------------------

def contrastive_activation_addition(
    model: nn.Module,
    base_ids: torch.Tensor,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    layer_idx: int,
    coeff: float,
) -> torch.Tensor:
    """One-shot contrastive activation addition.

    Computes a steering vector from positive/negative examples, applies it
    during a forward pass of base_ids, and returns the output logits.

    Args:
        model: AureliusTransformer instance.
        base_ids: (1, T) prompt token ids to run with steering.
        positive_ids: (B, S) token ids for positive examples.
        negative_ids: (B, S) token ids for negative examples.
        layer_idx: Layer at which to apply the steering vector.
        coeff: Steering coefficient.

    Returns:
        Logits of shape (1, T, vocab_size).
    """
    steering_vector = compute_steering_vector(
        model, positive_ids, negative_ids, layer_idx
    )

    config = SteeringConfig(layer_idx=layer_idx, coeff=coeff, normalize=True)
    hook = SteeringHook(steering_vector, config)
    handle = hook.register(model)

    try:
        with torch.no_grad():
            _, logits, _ = model(base_ids)
    finally:
        handle.remove()

    return logits
