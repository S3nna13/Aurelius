"""Activation Patching for Interpretability.

Provides tools to patch activations at specific layers and positions,
enabling causal analysis of which hidden states drive model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class PatchConfig:
    """Configuration for activation patching.

    Attributes:
        layer_idx: Which transformer layer to patch (0-indexed).
        position_idx: Which sequence position to patch (-1 = last).
        patch_type: How to apply the patch — "replace", "add", or "zero".
    """

    layer_idx: int = 0
    position_idx: int = -1
    patch_type: str = "replace"

    def __post_init__(self) -> None:
        if self.patch_type not in ("replace", "add", "zero"):
            raise ValueError(
                f"patch_type must be 'replace', 'add', or 'zero', got '{self.patch_type}'"
            )


def capture_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Run a forward pass and capture hidden states at a specific layer.

    Uses a forward hook on ``model.layers[layer_idx]`` to grab the output.

    Args:
        model: An ``AureliusTransformer`` instance.
        input_ids: (B, T) token ids.
        layer_idx: Index of the transformer layer to capture.

    Returns:
        (B, T, d_model) tensor of hidden states at that layer.
    """
    captured: list[torch.Tensor] = []

    def hook_fn(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        captured.append(x.detach())

    handle = model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return captured[0]


def create_patch_hook(
    source_activation: torch.Tensor,
    config: PatchConfig,
) -> Callable:
    """Create a forward hook that patches hidden states at a specific position.

    Args:
        source_activation: (B, T, d_model) activation tensor to patch from.
        config: Patch configuration controlling position and patch type.

    Returns:
        A hook function suitable for ``register_forward_hook``.
    """
    pos = config.position_idx

    def hook_fn(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        is_tuple = isinstance(output, tuple)

        # Clone to avoid in-place modification of the computation graph
        x = x.clone()

        if config.patch_type == "replace":
            x[:, pos, :] = source_activation[:, pos, :]
        elif config.patch_type == "add":
            x[:, pos, :] = x[:, pos, :] + source_activation[:, pos, :]
        elif config.patch_type == "zero":
            x[:, pos, :] = 0.0

        if is_tuple:
            return (x,) + output[1:]
        return x

    return hook_fn


def run_with_patch(
    model: nn.Module,
    input_ids: torch.Tensor,
    patch_hook: Callable,
    layer_idx: int,
) -> torch.Tensor:
    """Run a forward pass with a patch hook applied to a specific layer.

    Args:
        model: An ``AureliusTransformer`` instance.
        input_ids: (B, T) token ids.
        patch_hook: Hook function created by ``create_patch_hook``.
        layer_idx: Index of the transformer layer to patch.

    Returns:
        (B, T, V) logits from the patched forward pass.
    """
    handle = model.layers[layer_idx].register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            _loss, logits, _pkv = model(input_ids)
    finally:
        handle.remove()

    return logits


def compute_logit_diff(
    clean_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    token_a: int,
    token_b: int,
) -> float:
    """Compute the change in logit difference between two tokens.

    For the last sequence position, computes:
        clean_diff   = clean_logits[..., -1, token_a] - clean_logits[..., -1, token_b]
        patched_diff = patched_logits[..., -1, token_a] - patched_logits[..., -1, token_b]
    Returns the mean over the batch of (patched_diff - clean_diff).

    Args:
        clean_logits: (B, T, V) logits from the clean run.
        patched_logits: (B, T, V) logits from the patched run.
        token_a: First token index.
        token_b: Second token index.

    Returns:
        Scalar float of the mean logit difference change.
    """
    clean_diff = clean_logits[:, -1, token_a] - clean_logits[:, -1, token_b]
    patched_diff = patched_logits[:, -1, token_a] - patched_logits[:, -1, token_b]
    return (patched_diff - clean_diff).mean().item()


class ActivationPatcher:
    """High-level activation patching interface.

    Args:
        model: An ``AureliusTransformer`` instance.
        config: ``PatchConfig`` controlling which layer/position/type.
    """

    def __init__(self, model: nn.Module, config: PatchConfig) -> None:
        self.model = model
        self.config = config

    def patch_and_compare(
        self,
        clean_ids: torch.Tensor,
        source_ids: torch.Tensor,
        token_a: int,
        token_b: int,
    ) -> dict:
        """Run clean and patched forward passes and compare logit differences.

        Args:
            clean_ids: (B, T) token ids for the clean (baseline) input.
            source_ids: (B, T) token ids for the source (donor) input.
            token_a: First token index for logit diff.
            token_b: Second token index for logit diff.

        Returns:
            Dictionary with keys:
                - ``clean_diff``: mean logit(a) - logit(b) on clean input.
                - ``patched_diff``: mean logit(a) - logit(b) after patching.
                - ``effect_size``: patched_diff - clean_diff.
        """
        self.model.eval()

        # Get clean logits
        with torch.no_grad():
            _loss, clean_logits, _pkv = self.model(clean_ids)

        # Capture source activations
        source_act = capture_activations(
            self.model, source_ids, self.config.layer_idx
        )

        # Create patch hook and run patched forward
        patch_hook = create_patch_hook(source_act, self.config)
        patched_logits = run_with_patch(
            self.model, clean_ids, patch_hook, self.config.layer_idx
        )

        # Compute logit diffs at the last position
        clean_diff = (
            clean_logits[:, -1, token_a] - clean_logits[:, -1, token_b]
        ).mean().item()
        patched_diff = (
            patched_logits[:, -1, token_a] - patched_logits[:, -1, token_b]
        ).mean().item()

        return {
            "clean_diff": clean_diff,
            "patched_diff": patched_diff,
            "effect_size": patched_diff - clean_diff,
        }
