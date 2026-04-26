"""Latent Space Interpolation & Steering for Aurelius transformer.

Provides tools to extract hidden states from intermediate layers, compute
steering vectors as mean-difference directions, and apply them during
inference via forward hooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SteeringConfig:
    """Configuration for latent steering.

    Attributes:
        alpha: Scaling factor applied to the steering vector.
        layer_idx: Index of the transformer layer to hook.
            Negative values count from the end (e.g. -1 = last layer).
        method: How to apply the steering vector.
            "add"     — add alpha * vector to hidden states.
            "project" — add the projection of hidden states onto the vector,
                        scaled by alpha.
            "lerp"    — linear interpolation via lerp_hidden().
    """

    alpha: float = 1.0
    layer_idx: int = -1
    method: str = "add"  # "add" | "project" | "lerp"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_layer_idx(model: nn.Module, layer_idx: int) -> int:
    """Resolve a possibly-negative layer index to a non-negative one."""
    n = len(model.layers)
    if layer_idx < 0:
        layer_idx = n + layer_idx
    if not (0 <= layer_idx < n):
        raise IndexError(f"layer_idx {layer_idx} is out of range for model with {n} layers")
    return layer_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Run a forward pass and capture hidden states at the given layer.

    Registers a temporary hook on ``model.layers[layer_idx]`` that records
    the *output* hidden-state tensor (the first element returned by the layer,
    which has shape (B, T, d_model)).

    Args:
        model: AureliusTransformer instance.
        input_ids: (B, T) token-id tensor.
        layer_idx: Which transformer block to capture from. Negative values
            count from the end.

    Returns:
        Tensor of shape (B, T, d_model) — the hidden states at that layer.
    """
    resolved = _resolve_layer_idx(model, layer_idx)

    captured: list[torch.Tensor] = []

    def hook_fn(module, input, output):
        # TransformerBlock returns (hidden_states, kv); grab hidden_states.
        h = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(h.detach())

    handle = model.layers[resolved].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return captured[0]


def compute_steering_vector(
    model: nn.Module,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute a steering direction as the mean-difference in hidden space.

    Extracts hidden states for both sets of inputs, averages over the batch
    and sequence dimensions, and returns (mean_positive - mean_negative).

    Args:
        model: AureliusTransformer instance.
        positive_ids: (B_pos, T) token ids for the "positive" concept.
        negative_ids: (B_neg, T) token ids for the "negative" concept.
        layer_idx: Layer to extract hidden states from.

    Returns:
        Tensor of shape (d_model,) — the steering direction vector.
    """
    pos_h = extract_hidden_states(model, positive_ids, layer_idx)  # (B, T, d)
    neg_h = extract_hidden_states(model, negative_ids, layer_idx)  # (B, T, d)

    # Average over batch and sequence dimensions
    pos_mean = pos_h.mean(dim=(0, 1))  # (d,)
    neg_mean = neg_h.mean(dim=(0, 1))  # (d,)

    return pos_mean - neg_mean


def lerp_hidden(
    h: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Linear interpolation of hidden states toward a direction.

    Applies: h_new = h + alpha * (direction / ||direction||)

    When alpha=0 the output is identical to the input.

    Args:
        h: Hidden states of shape (B, T, d_model).
        direction: Direction vector of shape (d_model,).
        alpha: Interpolation strength.

    Returns:
        Tensor of the same shape as ``h``.
    """
    norm = direction.norm()
    if norm == 0:
        return h
    unit = direction / norm
    return h + alpha * unit


def apply_steering_hook(
    model: nn.Module,
    steering_vector: torch.Tensor,
    alpha: float,
    method: str,
    layer_idx: int,
) -> torch.utils.hooks.RemovableHook:
    """Register a persistent forward hook that modifies hidden states.

    The hook is applied *after* ``model.layers[layer_idx]`` and modifies
    the hidden-state tensor in-place before returning it to the next layer.

    Args:
        model: AureliusTransformer instance.
        steering_vector: (d_model,) direction tensor.
        alpha: Scaling factor.
        method: One of "add", "project", or "lerp".
        layer_idx: Which layer to attach the hook to.

    Returns:
        A ``RemovableHook`` handle with a ``.remove()`` method.
    """
    resolved = _resolve_layer_idx(model, layer_idx)
    vec = steering_vector  # (d_model,)

    def hook_fn(module, input, output):
        h, kv = output  # TransformerBlock returns (hidden_states, kv)

        if method == "add":
            h = h + alpha * vec.to(h.device)

        elif method == "project":
            # Project h onto vec, scale by alpha, and add
            v = vec.to(h.device)
            v_norm_sq = (v * v).sum()
            if v_norm_sq > 0:
                # projection of h onto v: (h·v / ||v||²) * v
                proj_coeff = (h * v).sum(dim=-1, keepdim=True) / v_norm_sq
                h = h + alpha * proj_coeff * v

        elif method == "lerp":
            h = lerp_hidden(h, vec.to(h.device), alpha)

        else:
            raise ValueError(f"Unknown steering method: {method!r}")

        return h, kv

    handle = model.layers[resolved].register_forward_hook(hook_fn)
    return handle


# ---------------------------------------------------------------------------
# LatentSteerer class
# ---------------------------------------------------------------------------


class LatentSteerer:
    """High-level interface for latent-space steering during generation.

    Example::

        config = SteeringConfig(alpha=2.0, layer_idx=-1, method="add")
        steerer = LatentSteerer(model, config)
        vec = compute_steering_vector(model, pos_ids, neg_ids, config.layer_idx)
        steerer.set_steering_vector(vec)
        steerer.enable()
        tokens = steerer.steer_generate(prompt_ids, max_new_tokens=50)
        steerer.disable()
    """

    def __init__(self, model: nn.Module, config: SteeringConfig) -> None:
        self.model = model
        self.config = config
        self._vector: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHook | None = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def set_steering_vector(self, vector: torch.Tensor) -> None:
        """Store the steering vector to use when the hook is active.

        Args:
            vector: (d_model,) direction tensor.
        """
        self._vector = vector

    def enable(self) -> None:
        """Register the steering hook on the target layer.

        If a hook is already registered, it is first removed.
        Raises ``RuntimeError`` if no steering vector has been set.
        """
        if self._vector is None:
            raise RuntimeError("No steering vector set. Call set_steering_vector() first.")
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

        self._handle = apply_steering_hook(
            self.model,
            self._vector,
            self.config.alpha,
            self.config.method,
            self.config.layer_idx,
        )

    def disable(self) -> None:
        """Remove the steering hook if one is registered."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def steer_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        """Generate tokens with the steering hook active.

        Enables the hook, runs generation, then disables the hook.
        If the hook was already active before the call it will remain active
        after (same semantics as calling enable/generate/disable manually).

        Args:
            input_ids: (B, T) prompt token ids.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            (B, T + generated_len) token ids.
        """
        was_active = self._handle is not None
        if not was_active:
            self.enable()
        try:
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        finally:
            if not was_active:
                self.disable()
        return output
