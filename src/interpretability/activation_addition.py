"""
src/interpretability/activation_addition.py

Activation Addition (ActAdd) steering for transformer models.

Reference: Turner et al., 2023 — "Activation Addition: Steering Language
Models Without Optimization".

The key idea: find a "steering vector" as the difference in residual-stream
activations between two contrastive prompts (e.g. "happy" vs "sad"), then
add a scaled version of that vector to the residual stream at a chosen layer
during generation.  This directly steers model behaviour without any
fine-tuning.

Pure PyTorch — no HuggingFace.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# SteeringVector
# ---------------------------------------------------------------------------

@dataclass
class SteeringVector:
    """A normalised direction in residual-stream space for a specific layer.

    Attributes
    ----------
    direction    : (d_model,) unit-norm steering direction.
    layer_idx    : Index of the transformer layer to apply this vector at.
    coefficient  : Default scaling factor applied during generation.
    label        : Human-readable description (e.g. "happy - sad").
    """
    direction: Tensor
    layer_idx: int
    coefficient: float = 1.0
    label: str = ""


# ---------------------------------------------------------------------------
# ActivationAddition
# ---------------------------------------------------------------------------

class ActivationAddition:
    """Implements Activation Addition steering for an AureliusTransformer.

    The class attaches lightweight forward hooks to the residual stream
    (i.e. the *output* of each TransformerBlock) so that activations can be
    captured and/or modified on the fly.

    Parameters
    ----------
    model : nn.Module
        An AureliusTransformer-like model that exposes a ``layers``
        ModuleList of TransformerBlock objects.
    layers_to_hook : list[int] or None
        Layer indices to attach capture hooks to.  If None, all layers are
        hooked.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_to_hook: Optional[List[int]] = None,
    ) -> None:
        self.model = model
        self.layers: nn.ModuleList = model.layers  # type: ignore[assignment]

        if layers_to_hook is None:
            layers_to_hook = list(range(len(self.layers)))
        self.layers_to_hook = layers_to_hook

        # Active steering hooks (added/removed by apply_steering context manager)
        self._steering_hooks: List[torch.utils.hooks.RemovableHook] = []

        # Placeholder so remove_hooks() has something to clear even before any
        # capture hooks are registered.
        self._capture_hooks: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_activations(self, input_ids: Tensor, layer_idx: int) -> Tensor:
        """Return residual-stream activations at *layer_idx*.

        A temporary capture hook is registered *after* any already-active
        steering hooks so that the captured tensor reflects the steered
        residual stream when called inside an ``apply_steering`` context.

        Parameters
        ----------
        input_ids : (batch, seq_len)
        layer_idx : which layer's output to return

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        if layer_idx not in self.layers_to_hook:
            raise ValueError(
                f"layer_idx {layer_idx} was not requested at construction time.  "
                f"Hooked layers: {self.layers_to_hook}"
            )

        captured: Dict[int, Tensor] = {}
        layer = self.layers[layer_idx]

        def _capture_hook(module: nn.Module, inputs, output) -> None:
            act = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = act.detach()

        # Register *after* any steering hooks already on this layer so that
        # what we capture is the already-steered activation.
        h = layer.register_forward_hook(_capture_hook)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            h.remove()

        if layer_idx not in captured:
            raise RuntimeError(f"Hook did not fire for layer {layer_idx}")
        return captured[layer_idx]

    def extract_steering_vector(
        self,
        positive_ids: Tensor,
        negative_ids: Tensor,
        layer_idx: int,
        label: str = "",
        coefficient: float = 1.0,
    ) -> SteeringVector:
        """Compute a unit-norm steering vector from two contrastive prompts.

        steering_direction = mean(positive_activations) - mean(negative_activations)

        Both means are taken over the *sequence* dimension so each prompt
        contributes a single (d_model,) representation per batch item; we
        then average over the batch dimension to get a single direction.

        Parameters
        ----------
        positive_ids : (batch, seq_len) token ids for the positive concept.
        negative_ids : (batch, seq_len) token ids for the negative concept.
        layer_idx    : which layer to extract activations from.
        label        : human-readable description stored in the returned vector.
        coefficient  : default scaling factor stored in the returned vector.

        Returns
        -------
        SteeringVector with unit-norm direction.
        """
        pos_acts = self.get_activations(positive_ids, layer_idx)  # (B, S, D)
        neg_acts = self.get_activations(negative_ids, layer_idx)  # (B, S, D)

        # Mean over sequence then over batch -> (D,)
        pos_mean = pos_acts.mean(dim=1).mean(dim=0)
        neg_mean = neg_acts.mean(dim=1).mean(dim=0)

        direction = pos_mean - neg_mean
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm

        if not label:
            label = f"layer_{layer_idx}_steering"

        return SteeringVector(
            direction=direction,
            layer_idx=layer_idx,
            coefficient=coefficient,
            label=label,
        )

    @contextlib.contextmanager
    def apply_steering(
        self,
        steering_vector: SteeringVector,
        coefficient: Optional[float] = None,
    ):
        """Context manager that injects a steering vector during forward passes.

        Inside the ``with`` block every call to ``model(...)`` will have
        ``coeff * direction`` added to the residual stream at
        ``steering_vector.layer_idx``.

        Parameters
        ----------
        steering_vector : the vector to apply.
        coefficient     : override the vector's own coefficient if provided.
        """
        coeff = coefficient if coefficient is not None else steering_vector.coefficient
        direction = steering_vector.direction  # (D,)
        layer_idx = steering_vector.layer_idx
        layer = self.layers[layer_idx]

        def _steering_hook(module: nn.Module, inputs, output):
            is_tuple = isinstance(output, tuple)
            act = output[0] if is_tuple else output
            # act: (batch, seq, d_model) — broadcast direction over batch/seq
            steered = act + coeff * direction.to(act.device)
            return (steered,) + output[1:] if is_tuple else steered

        handle = layer.register_forward_hook(_steering_hook)
        self._steering_hooks.append(handle)
        try:
            yield
        finally:
            handle.remove()
            if handle in self._steering_hooks:
                self._steering_hooks.remove(handle)

    def generate_steered(
        self,
        input_ids: Tensor,
        steering_vectors: List[SteeringVector],
        max_new_tokens: int = 20,
    ) -> Tensor:
        """Autoregressively generate tokens with steering vectors applied.

        Parameters
        ----------
        input_ids       : (batch, prompt_len) prompt token ids.
        steering_vectors: list of SteeringVector objects to apply simultaneously.
        max_new_tokens  : number of new tokens to generate.

        Returns
        -------
        (batch, prompt_len + max_new_tokens) token ids.
        """
        # Stack all context managers
        @contextlib.contextmanager
        def _all_steered():
            with contextlib.ExitStack() as stack:
                for sv in steering_vectors:
                    stack.enter_context(self.apply_steering(sv))
                yield

        with _all_steered():
            # Simple greedy generation (no KV-cache to keep hooks simple)
            generated = input_ids.clone()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    _, logits, _ = self.model(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

        return generated

    def remove_hooks(self) -> None:
        """Remove all registered capture and steering hooks."""
        for h in self._capture_hooks:
            h.remove()
        self._capture_hooks.clear()

        for h in self._steering_hooks:
            h.remove()
        self._steering_hooks.clear()


# ---------------------------------------------------------------------------
# SteeringVectorBank
# ---------------------------------------------------------------------------

class SteeringVectorBank:
    """A named store for SteeringVector objects with composition support.

    Usage
    -----
    >>> bank = SteeringVectorBank()
    >>> bank.add(happy_vector, "happy")
    >>> bank.add(angry_vector, "angry")
    >>> composed = bank.compose(["happy", "angry"], weights=[0.8, 0.2])
    """

    def __init__(self) -> None:
        self._store: Dict[str, SteeringVector] = {}

    def add(self, vector: SteeringVector, name: str) -> None:
        """Store *vector* under *name*."""
        self._store[name] = vector

    def get(self, name: str) -> SteeringVector:
        """Retrieve the vector stored under *name*."""
        if name not in self._store:
            raise KeyError(f"No steering vector named '{name}'.  Available: {list(self._store)}")
        return self._store[name]

    def compose(
        self,
        names: List[str],
        weights: Optional[List[float]] = None,
    ) -> SteeringVector:
        """Return a new SteeringVector as the (optionally weighted) average.

        All named vectors must share the same layer_idx and d_model.

        Parameters
        ----------
        names   : list of vector names to combine.
        weights : per-vector weights (must sum to > 0).  If None, uniform.

        Returns
        -------
        New SteeringVector whose direction is the weighted average of the
        individual directions (not re-normalised, to preserve magnitude
        information).  layer_idx is taken from the first vector; coefficient
        is the mean of the individual coefficients.
        """
        if not names:
            raise ValueError("names must not be empty")

        vectors = [self.get(n) for n in names]

        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        else:
            if len(weights) != len(vectors):
                raise ValueError("len(weights) must equal len(names)")
            total = sum(weights)
            if total <= 0:
                raise ValueError("weights must have positive sum")
            weights = [w / total for w in weights]

        # Stack directions: (n_vectors, d_model)
        stacked = torch.stack([v.direction for v in vectors], dim=0)  # (N, D)
        weight_t = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        composed_dir = (weight_t.unsqueeze(1) * stacked).sum(dim=0)  # (D,)

        mean_coeff = sum(v.coefficient for v in vectors) / len(vectors)
        label = " + ".join(names)

        return SteeringVector(
            direction=composed_dir,
            layer_idx=vectors[0].layer_idx,
            coefficient=mean_coeff,
            label=label,
        )
