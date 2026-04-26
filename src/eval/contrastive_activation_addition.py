"""Contrastive Activation Addition (CAA): extract steering vectors via contrastive
activation differences (Rimsky et al. 2023), then steer at inference time.

Reference: https://arxiv.org/abs/2312.06681
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CAAConfig:
    """Configuration for Contrastive Activation Addition."""

    layer_idx: int = 8
    """Which layer index to use when extracting the steering vector."""

    token_position: int = -1
    """Which token position to use (-1 = last token)."""

    normalize: bool = True
    """Whether to L2-normalise the extracted steering vector."""

    alpha: float = 20.0
    """Steering strength (scale applied to the steering vector)."""


# ---------------------------------------------------------------------------
# Activation Collector
# ---------------------------------------------------------------------------


class ActivationCollector:
    """Collects hidden-state activations from a list of model layers via hooks.

    Parameters
    ----------
    layers:
        An ordered list of ``nn.Module`` objects (e.g. transformer blocks).
        A forward hook is registered on each layer during collection.
    """

    def __init__(self, layers: list[nn.Module]) -> None:
        self._layers = layers
        self._activations: list[Tensor | None] = [None] * len(layers)
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Internal hook factory
    # ------------------------------------------------------------------

    def _make_hook(self, idx: int) -> Callable:
        def _hook(module: nn.Module, inputs: tuple, output) -> None:
            if isinstance(output, (tuple, list)):
                hidden = output[0]
            else:
                hidden = output
            self._activations[idx] = hidden.detach()

        return _hook

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ActivationCollector:
        self._install_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._remove_hooks()

    def _install_hooks(self) -> None:
        self._handles = [
            layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(self._layers)
        ]

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        model_fn: Callable,
        input_ids: Tensor,
    ) -> list[Tensor]:
        """Run ``model_fn(input_ids)`` with hooks attached, return activations.

        Parameters
        ----------
        model_fn:
            Callable that accepts ``input_ids`` (LongTensor of shape (B, T))
            and returns the model output (any form; only the hooked layer
            outputs are used).
        input_ids:
            LongTensor of shape (B, T).

        Returns
        -------
        List[Tensor]
            One ``(B, T, d)`` tensor per layer in ``self._layers``.
        """
        # Reset stored activations
        self._activations = [None] * len(self._layers)

        with self:
            model_fn(input_ids)

        return list(self._activations)  # type: ignore[return-value]

    def get_activations(self) -> list[Tensor]:
        """Return the activations captured during the last :meth:`collect` call."""
        return list(self._activations)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CAA Extractor
# ---------------------------------------------------------------------------


class CAAExtractor:
    """Extract steering vectors using contrastive activation differences.

    Parameters
    ----------
    config:
        :class:`CAAConfig` controlling which layer/token and whether to
        normalise.
    layers:
        Ordered list of ``nn.Module`` objects (transformer blocks) that the
        :class:`ActivationCollector` will hook into.
    """

    def __init__(self, config: CAAConfig, layers: list[nn.Module]) -> None:
        self.config = config
        self.layers = layers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect(self, model_fn: Callable, input_ids: Tensor) -> list[Tensor]:
        collector = ActivationCollector(self.layers)
        return collector.collect(model_fn, input_ids)

    @staticmethod
    def _pick_token(acts: Tensor, token_position: int) -> Tensor:
        """Return ``acts[:, token_position, :]`` — shape (B, d)."""
        return acts[:, token_position, :]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        model_fn: Callable,
        pos_input_ids: Tensor,
        neg_input_ids: Tensor,
    ) -> Tensor:
        """Extract a single steering vector for ``config.layer_idx``.

        Parameters
        ----------
        model_fn:
            Callable ``(input_ids: LongTensor(B, T)) -> Any``.
        pos_input_ids:
            Positive-contrast input ids, shape (B, T).
        neg_input_ids:
            Negative-contrast input ids, shape (B, T).

        Returns
        -------
        Tensor
            Steering vector of shape ``(d,)``.
        """
        pos_acts_all = self._collect(model_fn, pos_input_ids)
        neg_acts_all = self._collect(model_fn, neg_input_ids)

        layer_idx = self.config.layer_idx
        token_pos = self.config.token_position

        pos_layer = pos_acts_all[layer_idx]  # (B, T, d)
        neg_layer = neg_acts_all[layer_idx]  # (B, T, d)

        pos_vec = self._pick_token(pos_layer, token_pos).mean(dim=0)  # (d,)
        neg_vec = self._pick_token(neg_layer, token_pos).mean(dim=0)  # (d,)

        steering_vector = pos_vec - neg_vec

        if self.config.normalize:
            norm = steering_vector.norm()
            if norm > 0:
                steering_vector = steering_vector / norm

        return steering_vector

    def extract_multi_layer(
        self,
        model_fn: Callable,
        pos_input_ids: Tensor,
        neg_input_ids: Tensor,
    ) -> dict[int, Tensor]:
        """Extract steering vectors for every layer.

        Parameters
        ----------
        model_fn, pos_input_ids, neg_input_ids:
            Same as :meth:`extract`.

        Returns
        -------
        Dict[int, Tensor]
            Mapping ``{layer_idx: steering_vector (d,)}`` for all layers.
        """
        pos_acts_all = self._collect(model_fn, pos_input_ids)
        neg_acts_all = self._collect(model_fn, neg_input_ids)

        token_pos = self.config.token_position

        result: dict[int, Tensor] = {}
        for idx in range(len(self.layers)):
            pos_vec = self._pick_token(pos_acts_all[idx], token_pos).mean(dim=0)
            neg_vec = self._pick_token(neg_acts_all[idx], token_pos).mean(dim=0)
            sv = pos_vec - neg_vec
            if self.config.normalize:
                norm = sv.norm()
                if norm > 0:
                    sv = sv / norm
            result[idx] = sv

        return result


# ---------------------------------------------------------------------------
# CAA Steering Hook
# ---------------------------------------------------------------------------


class CAASteeringHook:
    """Forward hook that adds ``alpha * steering_vector`` at a token position.

    Parameters
    ----------
    steering_vector:
        Shape ``(d,)``.
    alpha:
        Scalar multiplier for the steering vector.
    token_position:
        Token index to steer at (-1 = last).
    """

    def __init__(
        self,
        steering_vector: Tensor,
        alpha: float,
        token_position: int = -1,
    ) -> None:
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.token_position = token_position
        self._handle: torch.utils.hooks.RemovableHook | None = None

    # ------------------------------------------------------------------
    # Hook callable
    # ------------------------------------------------------------------

    def __call__(self, module: nn.Module, input: tuple, output) -> object:
        """Add ``alpha * steering_vector`` to output at ``token_position``."""
        if isinstance(output, (tuple, list)):
            hidden = output[0]
            is_tuple = isinstance(output, tuple)
            rest = output[1:]
        else:
            hidden = output
            is_tuple = False
            rest = None

        # hidden: (B, T, d)
        delta = (self.alpha * self.steering_vector).to(hidden.device, hidden.dtype)
        hidden = hidden.clone()
        hidden[:, self.token_position, :] = hidden[:, self.token_position, :] + delta

        if rest is not None:
            if is_tuple:
                return (hidden,) + tuple(rest)
            else:
                return [hidden] + list(rest)
        return hidden

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, layer: nn.Module) -> None:
        """Register this hook as a forward hook on ``layer``."""
        self._handle = layer.register_forward_hook(self)

    def detach(self) -> None:
        """Remove the hook from the layer it was attached to."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def attached_to(self, layer: nn.Module):
        """Context manager: attach to ``layer``, yield, then detach."""
        self.attach(layer)
        try:
            yield self
        finally:
            self.detach()


# ---------------------------------------------------------------------------
# CAA Evaluator
# ---------------------------------------------------------------------------


class CAAEvaluator:
    """Run baseline and steered inference and return activation comparisons.

    Parameters
    ----------
    extractor:
        :class:`CAAExtractor` used to collect activations.
    config:
        :class:`CAAConfig` with steering parameters.
    """

    def __init__(self, extractor: CAAExtractor, config: CAAConfig) -> None:
        self.extractor = extractor
        self.config = config

    def steer_and_compare(
        self,
        model_fn: Callable,
        baseline_input_ids: Tensor,
        steered_layer: nn.Module,
        steering_vector: Tensor,
        alpha: float,
    ) -> dict[str, list[Tensor]]:
        """Run baseline and steered forward passes, return activations.

        Parameters
        ----------
        model_fn:
            Callable ``(input_ids: LongTensor(B, T)) -> Any``.
        baseline_input_ids:
            Input ids for both passes, shape (B, T).
        steered_layer:
            The ``nn.Module`` layer on which to attach the steering hook.
        steering_vector:
            Shape ``(d,)`` — the direction to steer in.
        alpha:
            Steering magnitude.

        Returns
        -------
        Dict with keys:
            - ``"baseline_activations"``: List[Tensor] — one per layer.
            - ``"steered_activations"``: List[Tensor] — one per layer.
        """
        # Baseline run
        collector = ActivationCollector(self.extractor.layers)
        baseline_acts = collector.collect(model_fn, baseline_input_ids)

        # Steered run — steering hook must be registered BEFORE the collector
        # so that PyTorch fires it first and the collector captures the
        # already-steered hidden states.
        hook = CAASteeringHook(
            steering_vector=steering_vector,
            alpha=alpha,
            token_position=self.config.token_position,
        )
        collector2 = ActivationCollector(self.extractor.layers)

        with hook.attached_to(steered_layer):
            with collector2:
                model_fn(baseline_input_ids)

        steered_acts = collector2.get_activations()

        return {
            "baseline_activations": baseline_acts,
            "steered_activations": steered_acts,
        }
