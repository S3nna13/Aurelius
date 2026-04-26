"""Progressive Layer Dropping: skip transformer layers during training for speedup and regularization."""  # noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LayerDropConfig:
    """Configuration for progressive layer dropping.

    Attributes:
        drop_rate: Maximum fraction of layers to drop (0.0-1.0).
        schedule: How drop rate changes over training — "linear", "constant", or "cosine".
        warmup_steps: Steps before any dropping begins (ramp from 0).
    """

    drop_rate: float = 0.2
    schedule: str = "linear"  # "linear" | "constant" | "cosine"
    warmup_steps: int = 100


def compute_drop_rate(step: int, total_steps: int, config: LayerDropConfig) -> float:
    """Compute the effective drop rate at a given training step.

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        config: LayerDropConfig with schedule and rate settings.

    Returns:
        Drop rate for this step, in [0, config.drop_rate].
    """
    if total_steps <= 0:
        return 0.0

    if step < config.warmup_steps:
        return 0.0

    # Progress after warmup, in [0, 1]
    remaining = total_steps - config.warmup_steps
    if remaining <= 0:
        return 0.0
    progress = min((step - config.warmup_steps) / remaining, 1.0)

    if config.schedule == "constant":
        return config.drop_rate
    elif config.schedule == "linear":
        return config.drop_rate * progress
    elif config.schedule == "cosine":
        # Cosine ramp from 0 to drop_rate
        return config.drop_rate * 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(
            f"Unknown schedule: {config.schedule!r}. Use 'linear', 'constant', or 'cosine'."
        )


def layer_drop_mask(n_layers: int, drop_rate: float, training: bool) -> list[bool]:
    """Generate a boolean mask indicating which layers to keep.

    Args:
        n_layers: Total number of layers.
        drop_rate: Probability of dropping each interior layer.
        training: Whether in training mode; no dropping during eval.

    Returns:
        List of booleans of length n_layers. True = keep, False = drop.
        First and last layers are always kept.
    """
    if n_layers <= 0:
        return []

    if not training or drop_rate <= 0.0:
        return [True] * n_layers

    mask = []
    for i in range(n_layers):
        # Always keep first and last layer
        if i == 0 or i == n_layers - 1:
            mask.append(True)
        else:
            keep = torch.rand(1).item() >= drop_rate
            mask.append(keep)
    return mask


def apply_layer_drop(model: nn.Module, mask: list[bool]) -> Tensor:
    """Run model forward, skipping layers where mask is False.

    Uses the Aurelius model API: model.embed, model.layers, model.norm, model.lm_head.
    Expects input_ids to be set on the model via a closure or passed separately.
    This function performs the embedding -> selective layers -> norm -> lm_head pipeline.

    Args:
        model: An AureliusTransformer instance.
        mask: Boolean mask of length len(model.layers). True = run layer, False = skip.

    Returns:
        logits: (batch, seq_len, vocab_size) tensor.
    """
    # We need input_ids — stored on model._layer_drop_input_ids by the trainer
    input_ids = model._layer_drop_input_ids

    B, S = input_ids.shape
    x = model.embed(input_ids)

    # Get RoPE frequencies
    freqs_cis = model.freqs_cis[:S]

    for i, layer in enumerate(model.layers):
        if mask[i]:
            x, _kv = layer(x, freqs_cis, None, None)

    x = model.norm(x)
    logits = model.lm_head(x)
    return logits


class LayerDropTrainer:
    """Trainer that applies progressive layer dropping during training.

    Args:
        model: AureliusTransformer model.
        config: LayerDropConfig.
        optimizer: torch optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        config: LayerDropConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.step_count = 0
        self.total_steps = 1000  # default, can be overridden

    def train_step(self, input_ids: Tensor) -> dict:
        """Run one training step with layer dropping.

        Args:
            input_ids: (batch, seq_len) token ids.

        Returns:
            Dict with keys: loss, n_active_layers, drop_rate.
        """
        self.model.train()

        # Compute current drop rate
        current_drop_rate = compute_drop_rate(self.step_count, self.total_steps, self.config)

        # Generate layer mask
        n_layers = len(self.model.layers)
        mask = layer_drop_mask(n_layers, current_drop_rate, training=True)
        n_active = sum(mask)

        # Store input_ids on model for apply_layer_drop
        self.model._layer_drop_input_ids = input_ids

        # Forward with layer dropping
        logits = apply_layer_drop(self.model, mask)

        # Compute loss: shift logits and labels (input_ids as labels for LM)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Backward + step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1

        # Clean up
        if hasattr(self.model, "_layer_drop_input_ids"):
            del self.model._layer_drop_input_ids

        return {
            "loss": loss.item(),
            "n_active_layers": n_active,
            "drop_rate": current_drop_rate,
        }


def estimate_speedup(n_layers: int, drop_rate: float) -> float:
    """Estimate training speedup from layer dropping.

    Args:
        n_layers: Total number of layers.
        drop_rate: Fraction of layers dropped.

    Returns:
        Estimated speedup factor: n_layers / (n_layers * (1 - drop_rate)).
    """
    active = n_layers * (1.0 - drop_rate)
    if active <= 0:
        return float("inf")
    return n_layers / active
