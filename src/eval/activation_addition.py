"""Activation Addition / Representation Engineering (Zou et al. 2023).

Steers model behaviour by adding or subtracting a steering vector to hidden
states at a specified layer during inference.

References:
    Zou et al. (2023) "Representation Engineering: A Top-Down Approach to
    AI Transparency" — https://arxiv.org/abs/2310.01405
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# SteeringVector
# ---------------------------------------------------------------------------


@dataclass
class SteeringVector:
    """A direction in activation space associated with a specific layer.

    Attributes:
        direction: Unit-norm vector of shape ``(d_model,)``.
        layer_idx: Index of the transformer layer this vector applies to.
    """

    direction: Tensor
    layer_idx: int


# ---------------------------------------------------------------------------
# ActivationAddition
# ---------------------------------------------------------------------------


class ActivationAddition:
    """Adds steering vectors to layer outputs via forward hooks.

    Usage::

        aa = ActivationAddition(model.layers)
        aa.add_vector(sv, alpha=10.0)
        with aa:
            logits = model(input_ids)

    Args:
        layers: List of ``nn.Module`` objects corresponding to the transformer
            layers that can be steered (e.g. ``list(model.layers)``).
    """

    def __init__(self, layers: List[nn.Module]) -> None:
        self._layers: List[nn.Module] = layers
        # Maps layer_idx -> (SteeringVector, alpha)
        self._vectors: Dict[int, tuple[SteeringVector, float]] = {}
        # Maps layer_idx -> hook handle (populated inside the context manager)
        self._handles: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_vector(self, sv: SteeringVector, alpha: float = 1.0) -> None:
        """Register a steering vector for the given layer.

        Args:
            sv: :class:`SteeringVector` specifying direction and layer index.
            alpha: Scalar multiplier applied to the direction before adding.
        """
        self._vectors[sv.layer_idx] = (sv, alpha)

    def remove_vector(self, layer_idx: int) -> None:
        """Remove the steering vector registered for *layer_idx*.

        Args:
            layer_idx: Index of the layer whose steering vector should be
                removed.

        Raises:
            KeyError: If no vector is registered for *layer_idx*.
        """
        del self._vectors[layer_idx]

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationAddition":
        for layer_idx, (sv, alpha) in self._vectors.items():
            layer = self._layers[layer_idx]
            direction = sv.direction  # captured in closure

            def _make_hook(d: Tensor, a: float) -> Callable:
                def _hook(module: nn.Module, inputs: Any, output: Any) -> Any:
                    is_tuple = isinstance(output, (tuple, list))
                    hidden = output[0] if is_tuple else output
                    d_cast = d.to(device=hidden.device, dtype=hidden.dtype)
                    hidden = hidden + a * d_cast
                    if is_tuple:
                        return (hidden,) + tuple(output[1:])
                    return hidden
                return _hook

            handle = layer.register_forward_hook(_make_hook(direction, alpha))
            self._handles[layer_idx] = handle
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# SteeringVectorExtractor
# ---------------------------------------------------------------------------


class SteeringVectorExtractor:
    """Extracts a steering vector from contrastive forward passes.

    Given a *model_fn* that returns hidden states at every layer and two sets
    of inputs (positive and negative), the extractor computes::

        direction = mean(pos_hiddens[layer_idx]) - mean(neg_hiddens[layer_idx])
        direction = F.normalize(direction, dim=0)

    where the mean is taken over both the batch (B) and sequence (T) dimensions.
    """

    def extract(
        self,
        model_fn: Callable[[Tensor], List[Tensor]],
        pos_inputs: Tensor,
        neg_inputs: Tensor,
        layer_idx: int,
    ) -> SteeringVector:
        """Extract and return a :class:`SteeringVector`.

        Args:
            model_fn: Callable that accepts ``input_ids`` of shape ``(B, T)``
                and returns a list of hidden-state tensors, one per layer,
                each of shape ``(B, T, d_model)``.
            pos_inputs: ``(B, T)`` token-id tensor for positive examples.
            neg_inputs: ``(B, T)`` token-id tensor for negative examples.
            layer_idx: Which layer's hidden states to use.

        Returns:
            :class:`SteeringVector` with an L2-normalised direction.
        """
        with torch.no_grad():
            pos_hiddens: List[Tensor] = model_fn(pos_inputs)
            neg_hiddens: List[Tensor] = model_fn(neg_inputs)

        # Each hidden: (B, T, d_model) — mean over B and T
        pos_mean: Tensor = pos_hiddens[layer_idx].mean(dim=(0, 1))  # (d_model,)
        neg_mean: Tensor = neg_hiddens[layer_idx].mean(dim=(0, 1))  # (d_model,)

        direction = pos_mean - neg_mean
        direction = F.normalize(direction, dim=0)

        return SteeringVector(direction=direction, layer_idx=layer_idx)


# ---------------------------------------------------------------------------
# RepresentationDatabase
# ---------------------------------------------------------------------------


@dataclass
class RepresentationDatabase:
    """Named store of :class:`SteeringVector` objects.

    Example::

        db = RepresentationDatabase()
        db.add("honesty", sv)
        sv = db.get("honesty")
        db.remove("honesty")
    """

    _store: Dict[str, SteeringVector] = field(default_factory=dict, init=False, repr=False)

    def add(self, name: str, sv: SteeringVector) -> None:
        """Store a steering vector under *name*.

        Args:
            name: Unique identifier for this concept direction.
            sv: :class:`SteeringVector` to store.
        """
        self._store[name] = sv

    def get(self, name: str) -> SteeringVector:
        """Retrieve the steering vector registered as *name*.

        Args:
            name: Identifier used when the vector was added.

        Returns:
            The corresponding :class:`SteeringVector`.

        Raises:
            KeyError: If no vector is registered under *name*.
        """
        return self._store[name]

    def list_names(self) -> List[str]:
        """Return the names of all stored steering vectors."""
        return list(self._store.keys())

    def remove(self, name: str) -> None:
        """Delete the steering vector registered as *name*.

        Args:
            name: Identifier used when the vector was added.

        Raises:
            KeyError: If no vector is registered under *name*.
        """
        del self._store[name]
