"""Stochastic depth (layer dropout): randomly drop entire layers during training for regularization."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class StochasticDepthConfig:
    """Configuration for stochastic depth (layer dropout)."""

    drop_rate: float = 0.1
    schedule: str = "linear"  # "linear" | "uniform" | "reverse_linear"
    min_drop_rate: float = 0.0
    max_drop_rate: float = 0.5


def get_layer_drop_rates(n_layers: int, config: StochasticDepthConfig) -> list[float]:
    """Compute per-layer drop rate based on schedule.

    Args:
        n_layers: Number of transformer layers.
        config: StochasticDepthConfig with schedule and rate settings.

    Returns:
        List of n_layers floats, each clamped to [min_drop_rate, max_drop_rate].
    """
    if n_layers == 0:
        return []

    rates: list[float] = []

    if config.schedule == "linear":
        # linearly from 0 to drop_rate (deeper = higher drop rate)
        for i in range(n_layers):
            if n_layers == 1:
                rate = config.drop_rate
            else:
                rate = config.drop_rate * i / (n_layers - 1)
            rates.append(rate)

    elif config.schedule == "uniform":
        # all layers get the same drop_rate
        rates = [config.drop_rate] * n_layers

    elif config.schedule == "reverse_linear":
        # linearly from drop_rate to 0 (early layers higher)
        for i in range(n_layers):
            if n_layers == 1:
                rate = config.drop_rate
            else:
                rate = config.drop_rate * (1.0 - i / (n_layers - 1))
            rates.append(rate)

    else:
        raise ValueError(
            f"Unknown schedule: {config.schedule!r}. "
            "Use 'linear', 'uniform', or 'reverse_linear'."
        )

    # Clamp to [min_drop_rate, max_drop_rate]
    rates = [
        max(config.min_drop_rate, min(config.max_drop_rate, r))
        for r in rates
    ]

    return rates


def stochastic_depth_forward(
    x: Tensor,
    layer: nn.Module,
    drop_rate: float,
    training: bool,
) -> Tensor:
    """Apply layer with stochastic depth (layer dropout).

    During training: drop entire layer with probability drop_rate (return x unchanged).
    Surviving output is scaled by 1/(1-drop_rate) to maintain expected value.
    During eval: always apply the layer.

    Note: This function calls layer(x) with only the tensor argument.
    Use StochasticDepthLayer for layers that require additional arguments.

    Args:
        x: Input tensor.
        layer: The layer to (possibly) apply (must accept a single tensor).
        drop_rate: Probability of dropping this layer.
        training: Whether we are in training mode.

    Returns:
        Output tensor (same shape as x).
    """
    if not training or drop_rate == 0.0:
        return layer(x)

    if drop_rate >= 1.0:
        # Always drop — return input unchanged
        return x

    # Sample a single Bernoulli to decide whether to drop the whole layer
    keep_prob = 1.0 - drop_rate
    if torch.rand(1).item() < drop_rate:
        # Drop: return input unchanged
        return x
    else:
        # Keep: apply layer and scale up to maintain expected value
        out = layer(x)
        return out / keep_prob


class StochasticDepthLayer(nn.Module):
    """Wraps a layer with stochastic depth (layer dropout).

    Passes all positional and keyword arguments through to the underlying
    layer, so it works with layers that require extra arguments (e.g.,
    TransformerBlock which takes freqs_cis, mask, and past_kv).
    """

    def __init__(self, layer: nn.Module, drop_rate: float) -> None:
        super().__init__()
        self.layer = layer
        self.drop_rate = drop_rate

    def forward(self, x: Tensor, *args, **kwargs):
        """Forward pass with stochastic depth.

        Args:
            x: Input tensor (first positional argument).
            *args: Additional positional args passed to the wrapped layer.
            **kwargs: Additional keyword args passed to the wrapped layer.

        Returns:
            Layer output (or x unchanged if layer is dropped).
        """
        if not self.training or self.drop_rate == 0.0:
            return self.layer(x, *args, **kwargs)

        if self.drop_rate >= 1.0:
            # Always drop
            if args or kwargs:
                # Need to return same structure as layer would
                # For transformer blocks returning (tensor, kv), fabricate a dummy kv
                # by running the layer in no_grad then zeroing — use eval trick
                with torch.no_grad():
                    result = self.layer(x, *args, **kwargs)
                if isinstance(result, tuple):
                    # Return x with dummy second element matching shape
                    dummy = tuple(torch.zeros_like(t) for t in result[1:])
                    return (x,) + dummy
                return x
            return x

        keep_prob = 1.0 - self.drop_rate
        if torch.rand(1).item() < self.drop_rate:
            # Drop
            if args or kwargs:
                with torch.no_grad():
                    result = self.layer(x, *args, **kwargs)
                if isinstance(result, tuple):
                    dummy = tuple(torch.zeros_like(t) for t in result[1:])
                    return (x,) + dummy
            return x
        else:
            # Keep and scale
            result = self.layer(x, *args, **kwargs)
            if isinstance(result, tuple):
                # Scale only the first element (the hidden state)
                scaled = (result[0] / keep_prob,) + result[1:]
                return scaled
            return result / keep_prob


def wrap_with_stochastic_depth(
    layers: nn.ModuleList,
    config: StochasticDepthConfig,
) -> nn.ModuleList:
    """Wrap each layer in a ModuleList with StochasticDepthLayer.

    Args:
        layers: Original nn.ModuleList of transformer layers.
        config: StochasticDepthConfig controlling per-layer drop rates.

    Returns:
        New nn.ModuleList where each layer is wrapped with StochasticDepthLayer.
    """
    n_layers = len(layers)
    drop_rates = get_layer_drop_rates(n_layers, config)
    wrapped = nn.ModuleList([
        StochasticDepthLayer(layer, rate)
        for layer, rate in zip(layers, drop_rates)
    ])
    return wrapped


def compute_expected_depth(n_layers: int, drop_rates: list[float]) -> float:
    """Compute expected number of active layers.

    Args:
        n_layers: Total number of layers.
        drop_rates: Per-layer drop rates (length must equal n_layers).

    Returns:
        Expected number of active (non-dropped) layers.
    """
    return sum(1.0 - r for r in drop_rates[:n_layers])


class StochasticDepthTrainer:
    """Wraps a model's layers with stochastic depth for regularized training."""

    def __init__(self, model: nn.Module, config: StochasticDepthConfig) -> None:
        self.model = model
        self.config = config

        # Replace model.layers with wrapped version
        wrapped = wrap_with_stochastic_depth(model.layers, config)
        model.layers = wrapped

        # Collect drop rates for expected depth computation
        n_layers = len(model.layers)
        self._drop_rates = get_layer_drop_rates(n_layers, config)

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict:
        """Perform one forward + backward training step.

        Args:
            input_ids: (B, T) input token ids.
            labels: (B, T) target token ids.

        Returns:
            Dict with keys:
                'loss': float scalar cross-entropy loss.
                'expected_depth': float expected number of active layers.
        """
        self.model.train()

        _, logits, _ = self.model(input_ids)

        # Shift for next-token prediction
        B, T, V = logits.shape
        shift_logits = logits[:, :-1, :].reshape(-1, V)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels)

        loss.backward()

        expected_depth = compute_expected_depth(len(self._drop_rates), self._drop_rates)

        return {
            "loss": loss.item(),
            "expected_depth": expected_depth,
        }
