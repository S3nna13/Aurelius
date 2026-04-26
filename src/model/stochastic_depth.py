"""Stochastic Depth (layer drop) for the Aurelius LLM project.

Randomly drops entire transformer layers during training, keeping only the
residual connection.  This regularises deep networks while adding minimal
overhead at inference time.

Components:
  - StochasticDepthConfig      : dataclass for drop_rate and mode
  - stochastic_depth           : functional API wrapping a layer with drop logic
  - StochasticDepthLayer       : nn.Module wrapper for a single layer
  - LinearStochasticDepth      : utility for computing/applying linearly-spaced
                                  drop rates across a stack of layers
  - StochasticDepthTransformer : nn.Module that chains wrapped layers sequentially
  - get_expected_depth         : returns the expected number of active layers
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StochasticDepthConfig:
    """Configuration for stochastic depth.

    Attributes:
        drop_rate: Probability that a given layer is dropped during a forward
                   pass (replaced by the identity residual). Range [0, 1).
        mode:      Sampling granularity.
                     "row"   — sample independently per batch element.
                     "batch" — sample a single Bernoulli for the whole batch.
    """

    drop_rate: float = 0.1
    mode: str = "row"


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def stochastic_depth(
    x: torch.Tensor,
    layer_fn: Callable[[torch.Tensor], torch.Tensor],
    drop_prob: float,
    training: bool,
) -> torch.Tensor:
    """Apply stochastic depth (layer drop) to a residual layer.

    During training a Bernoulli gate decides whether the layer output is
    added to the residual.  When the layer *is* kept its output is scaled by
    ``1 / (1 - drop_prob)`` so that the expected value of the layer output
    matches the full-pass value (i.e. an unbiased estimate).

    During evaluation the layer is always applied without scaling.

    Args:
        x:          Input tensor of arbitrary shape (B, ...).
        layer_fn:   Callable that maps x -> layer_output, same shape as x.
        drop_prob:  Probability of dropping the layer. 0 = never drop,
                    1 = always drop.
        training:   Whether the model is in training mode.

    Returns:
        Tensor of the same shape as x:
            - eval:  x + layer_fn(x)
            - train, kept:   x + layer_fn(x) / (1 - drop_prob)
            - train, dropped: x  (residual only)
    """
    if not training or drop_prob == 0.0:
        return x + layer_fn(x)

    # drop_prob == 1.0 means always drop — return residual only
    if drop_prob >= 1.0:
        return x

    survival_prob = 1.0 - drop_prob
    # Bernoulli sample: 1 = keep, 0 = drop
    noise = torch.empty(1, device=x.device, dtype=x.dtype).bernoulli_(survival_prob)
    if noise.item() == 0.0:
        return x  # layer dropped
    # Layer kept — scale for unbiased estimate
    layer_out = layer_fn(x)
    return x + layer_out / survival_prob


# ---------------------------------------------------------------------------
# StochasticDepthLayer
# ---------------------------------------------------------------------------


class StochasticDepthLayer(nn.Module):
    """nn.Module wrapper that applies stochastic depth to a single layer.

    Args:
        layer:     The underlying nn.Module to wrap (e.g. a transformer block).
        drop_prob: Probability of dropping this layer during training.

    Properties:
        survival_prob: 1 - drop_prob.
    """

    def __init__(self, layer: nn.Module, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.layer = layer
        self.drop_prob = drop_prob

    @property
    def survival_prob(self) -> float:
        """Probability that this layer is executed (1 - drop_prob)."""
        return 1.0 - self.drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic depth.

        Args:
            x: Input tensor (B, ...).

        Returns:
            Tensor of the same shape as x.
        """
        return stochastic_depth(x, self.layer, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# LinearStochasticDepth
# ---------------------------------------------------------------------------


class LinearStochasticDepth:
    """Utility for linearly-spaced stochastic depth across a layer stack.

    Drop rates increase linearly from 0 (first layer) to ``max_drop_rate``
    (last layer), following the schedule introduced in "Deep Networks with
    Stochastic Depth" (Huang et al., 2016).
    """

    @staticmethod
    def get_drop_rates(n_layers: int, max_drop_rate: float) -> list[float]:
        """Compute per-layer drop rates linearly spaced from 0 to max_drop_rate.

        Args:
            n_layers:      Number of transformer layers.
            max_drop_rate: Drop rate for the last layer.

        Returns:
            List of n_layers floats.  First element is 0.0, last element is
            max_drop_rate.  For n_layers == 1 the single rate is 0.0.
        """
        if n_layers <= 1:
            return [0.0] * n_layers
        return [max_drop_rate * i / (n_layers - 1) for i in range(n_layers)]

    @staticmethod
    def wrap_layers(
        layers: list[nn.Module],
        max_drop_rate: float,
    ) -> list[StochasticDepthLayer]:
        """Wrap each layer with its linearly-computed stochastic depth rate.

        Args:
            layers:        List of nn.Module layers.
            max_drop_rate: Maximum drop rate (applied to the last layer).

        Returns:
            List of StochasticDepthLayer wrappers, one per input layer.
        """
        drop_rates = LinearStochasticDepth.get_drop_rates(len(layers), max_drop_rate)
        return [
            StochasticDepthLayer(layer, drop_prob=rate) for layer, rate in zip(layers, drop_rates)
        ]


# ---------------------------------------------------------------------------
# StochasticDepthTransformer
# ---------------------------------------------------------------------------


class StochasticDepthTransformer(nn.Module):
    """Sequential transformer that applies stochastic depth to every layer.

    Args:
        layers: List of nn.Module layers (e.g. transformer blocks).
        config: StochasticDepthConfig controlling drop_rate and mode.

    Note:
        Each layer is wrapped with a *uniform* drop_rate taken from
        ``config.drop_rate``.  For linearly-varying drop rates use
        ``LinearStochasticDepth.wrap_layers`` and pass the result directly.
    """

    def __init__(
        self,
        layers: list[nn.Module],
        config: StochasticDepthConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.wrapped_layers = nn.ModuleList(
            StochasticDepthLayer(layer, drop_prob=config.drop_rate) for layer in layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass x through all wrapped layers sequentially.

        Args:
            x: Input tensor (B, ...).

        Returns:
            Output tensor of the same shape as x.
        """
        for layer in self.wrapped_layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_expected_depth(n_layers: int, drop_rates: list[float]) -> float:
    """Compute the expected number of active (non-dropped) layers.

    Expected depth = sum_i (1 - drop_rate_i).

    Args:
        n_layers:   Total number of layers.
        drop_rates: Per-layer drop rates (length must equal n_layers).

    Returns:
        Float — expected number of layers that are executed.
    """
    return sum(1.0 - r for r in drop_rates)
