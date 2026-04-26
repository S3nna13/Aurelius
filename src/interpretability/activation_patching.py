"""
src/interpretability/activation_patching.py

Activation patching utilities for causal tracing in transformer models.
Replaces activations at specific layer/position with activations from a
"clean" run to trace causal influence.

Pure PyTorch — no HuggingFace, no scipy, no sklearn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# PatchConfig
# ---------------------------------------------------------------------------


@dataclass
class PatchConfig:
    """Configuration for an activation patching experiment."""

    patch_layers: list[int] = field(default_factory=list)
    patch_positions: list[int] | None = None
    patch_heads: list[int] | None = None
    normalize_effect: bool = True


# ---------------------------------------------------------------------------
# ActivationStore
# ---------------------------------------------------------------------------


class ActivationStore:
    """Captures and stores activations via forward hooks."""

    def __init__(self) -> None:
        self.store: dict[str, Tensor] = {}

    def hook_fn(self, name: str) -> Callable:
        """Return a forward hook that saves the module output to self.store[name]."""

        def _hook(module: nn.Module, input, output) -> None:  # noqa: ANN001
            # output may be a tuple (e.g. attention layers) — store the first element
            if isinstance(output, tuple):
                self.store[name] = output[0].detach()
            else:
                self.store[name] = output.detach()

        return _hook

    def get(self, name: str) -> Tensor:
        """Retrieve a stored activation by name."""
        return self.store[name]

    def clear(self) -> None:
        """Remove all stored activations."""
        self.store.clear()


# ---------------------------------------------------------------------------
# patch_activations
# ---------------------------------------------------------------------------


def patch_activations(
    target: Tensor,
    source: Tensor,
    positions: list[int] | None = None,
) -> Tensor:
    """
    Replace activations in *target* with those from *source*.

    Parameters
    ----------
    target : Tensor  — tensor to patch (any shape, position dim is -2)
    source : Tensor  — same shape as target
    positions : list of ints or None
        Indices along the sequence (second-to-last) dimension to replace.
        If None, all positions are replaced.

    Returns
    -------
    Tensor with the same shape as *target* but with selected positions
    replaced by the corresponding values from *source*.
    """
    patched = target.clone()
    if positions is None:
        patched = source.clone()
    else:
        patched[..., positions, :] = source[..., positions, :]
    return patched


# ---------------------------------------------------------------------------
# compute_patching_effect
# ---------------------------------------------------------------------------


def compute_patching_effect(
    original_logits: Tensor,
    patched_logits: Tensor,
    clean_logits: Tensor,
) -> Tensor:
    """
    Normalised patching effect.

    effect = (patched - original) / (clean - original + 1e-8)

    A value near 1 means patching fully restored the clean behaviour;
    a value near 0 means patching had no effect.

    All three tensors must have the same shape.  The returned tensor has
    the same shape as the inputs.
    """
    return (patched_logits - original_logits) / (clean_logits - original_logits + 1e-8)


# ---------------------------------------------------------------------------
# PatchingExperiment
# ---------------------------------------------------------------------------


class PatchingExperiment:
    """High-level API for running activation patching experiments."""

    def __init__(self, model: nn.Module, config: PatchConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # capture_activations
    # ------------------------------------------------------------------

    def capture_activations(
        self,
        x: Tensor,
        hook_points: list[str],
    ) -> dict[str, Tensor]:
        """
        Run a forward pass and return the activations at each named module.

        Parameters
        ----------
        x           : input tensor
        hook_points : list of module names (as in model.named_modules())

        Returns
        -------
        dict mapping hook_point name → captured Tensor
        """
        store = ActivationStore()
        hooks = []
        module_dict = dict(self.model.named_modules())

        for name in hook_points:
            if name in module_dict:
                h = module_dict[name].register_forward_hook(store.hook_fn(name))
                hooks.append(h)

        try:
            with torch.no_grad():
                self.model(x)
        finally:
            for h in hooks:
                h.remove()

        return dict(store.store)

    # ------------------------------------------------------------------
    # run_patched_forward
    # ------------------------------------------------------------------

    def run_patched_forward(
        self,
        x: Tensor,
        patch_dict: dict[str, Tensor],
        hook_points: list[str],
    ) -> Tensor:
        """
        Run a forward pass where certain intermediate activations are replaced
        by pre-computed tensors from *patch_dict*.

        Parameters
        ----------
        x           : input tensor
        patch_dict  : mapping from hook_point name → replacement Tensor
        hook_points : list of hook_point names to patch (subset of patch_dict keys)

        Returns
        -------
        output logits Tensor
        """
        hooks = []
        module_dict = dict(self.model.named_modules())

        for name in hook_points:
            if name in module_dict and name in patch_dict:
                replacement = patch_dict[name]
                positions = self.config.patch_positions

                def _make_patch_hook(rep: Tensor, pos: list[int] | None) -> Callable:
                    def _hook(module, input, output):  # noqa: ANN001
                        is_tuple = isinstance(output, tuple)
                        act = output[0] if is_tuple else output
                        patched = patch_activations(act, rep, positions)
                        return (patched,) + output[1:] if is_tuple else patched

                    return _hook

                h = module_dict[name].register_forward_hook(
                    _make_patch_hook(replacement, positions)
                )
                hooks.append(h)

        try:
            with torch.no_grad():
                output = self.model(x)
        finally:
            for h in hooks:
                h.remove()

        return output

    # ------------------------------------------------------------------
    # compute_layer_importance
    # ------------------------------------------------------------------

    def compute_layer_importance(
        self,
        clean_x: Tensor,
        corrupted_x: Tensor,
        hook_points: list[str],
    ) -> dict[str, float]:
        """
        For each hook point, patch the corrupted run with clean activations
        and measure mean absolute difference of output logits vs. the
        clean-only forward pass.

        Returns
        -------
        dict mapping hook_point name → importance score (float, ≥ 0)
        """
        # Baseline: clean forward pass logits
        with torch.no_grad():
            clean_logits = self.model(clean_x)

        # Capture clean activations at every hook point
        clean_acts = self.capture_activations(clean_x, hook_points)

        importance: dict[str, float] = {}
        for name in hook_points:
            if name not in clean_acts:
                importance[name] = 0.0
                continue

            patch_dict = {name: clean_acts[name]}
            patched_logits = self.run_patched_forward(corrupted_x, patch_dict, [name])
            score = (patched_logits - clean_logits).abs().mean().item()
            importance[name] = score

        return importance


# ---------------------------------------------------------------------------
# attention_pattern_similarity
# ---------------------------------------------------------------------------


def attention_pattern_similarity(attn_a: Tensor, attn_b: Tensor) -> Tensor:
    """
    Cosine similarity between flattened attention patterns per batch item and head.

    Parameters
    ----------
    attn_a : Tensor  shape (B, H, T, T)
    attn_b : Tensor  shape (B, H, T, T)

    Returns
    -------
    Tensor shape (B, H) containing cosine similarities in [-1, 1].
    """
    # Flatten the (T, T) dimensions
    a_flat = attn_a.reshape(attn_a.shape[0], attn_a.shape[1], -1)  # (B, H, T*T)
    b_flat = attn_b.reshape(attn_b.shape[0], attn_b.shape[1], -1)  # (B, H, T*T)

    dot = (a_flat * b_flat).sum(dim=-1)  # (B, H)
    norm_a = a_flat.norm(dim=-1).clamp(min=1e-8)  # (B, H)
    norm_b = b_flat.norm(dim=-1).clamp(min=1e-8)  # (B, H)

    return dot / (norm_a * norm_b)
