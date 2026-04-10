"""Causal intervention and counterfactual generation for language models.

Implements activation patching (causal tracing) to identify which hidden
states causally affect model outputs.

Reference: Meng et al. 2022 "Locating and Editing Factual Associations in GPT"
           (arXiv:2202.05262)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class InterventionConfig:
    """Configuration for a single activation intervention."""
    intervention_layer: int = 0    # which transformer layer to intervene on
    intervention_dim: int = 0      # which hidden dimension to patch
    patch_value: float = 0.0       # value to write into that dimension


class ActivationPatcher:
    """Patches a specific hidden dimension at a specific layer using forward hooks.

    Sets hidden_state[:, :, intervention_dim] = patch_value during the forward
    pass of the specified layer.
    """

    def __init__(self, model: nn.Module, config: InterventionConfig) -> None:
        self.model = model
        self.config = config
        self._hooks: list = []

    def patch(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward with patched activations.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            logits: (B, T, V) logits from the patched forward pass.
        """
        layer = self.model.layers[self.config.intervention_layer]
        dim = self.config.intervention_dim
        val = self.config.patch_value

        def hook(module, input, output):
            # TransformerBlock returns (hidden_state, kv_cache)
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, :, dim] = val
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        handle = layer.register_forward_hook(hook)
        self._hooks.append(handle)
        try:
            with torch.no_grad():
                loss, logits, pkv = self.model(input_ids)
        finally:
            handle.remove()
            self._hooks.remove(handle)

        return logits

    def restore(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def counterfactual_logits(
    model: nn.Module,
    original_ids: torch.Tensor,
    counterfactual_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Run original forward to capture hidden states at layer_idx, then patch
    those into a counterfactual forward of counterfactual_ids.

    Args:
        model: The transformer model.
        original_ids: (1, T) token ids for the "clean" / original input.
        counterfactual_ids: (1, T) token ids for the counterfactual input.
        layer_idx: Which layer's hidden states to transplant.

    Returns:
        logits: (1, T, V) logits from the counterfactual forward with patched
                hidden states.
    """
    # Step 1: Capture hidden states from original forward at layer_idx
    captured_hs: list[torch.Tensor] = []

    def capture_hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured_hs.append(hs.detach().clone())

    handle = model.layers[layer_idx].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(original_ids)
    finally:
        handle.remove()

    original_hs = captured_hs[0]  # (1, T, D)

    # Step 2: Run counterfactual forward, patching in original hidden states
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            return (original_hs,) + output[1:]
        return original_hs

    handle2 = model.layers[layer_idx].register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            loss, logits, pkv = model(counterfactual_ids)
    finally:
        handle2.remove()

    return logits


class CausalTracer:
    """Traces which (layer, position) pairs causally affect the output at a
    target position.

    Uses the indirect effect method: for each (layer, position), patch the
    clean activation at that location into a corrupted forward pass and measure
    how much it restores the clean output distribution.
    """

    def __init__(self, model: nn.Module, n_layers: int) -> None:
        self.model = model
        self.n_layers = n_layers

    def trace(self, input_ids: torch.Tensor, target_position: int) -> torch.Tensor:
        """Compute indirect effect for every (layer, position) pair.

        Args:
            input_ids: (1, T) clean token ids.
            target_position: The output position whose logits we track.

        Returns:
            effects: (n_layers, T) tensor of effect magnitudes (non-negative).
        """
        with torch.no_grad():
            B, T = input_ids.shape
            device = input_ids.device

            # ----------------------------------------------------------------
            # Step 1: Capture ALL layer hidden states from the clean forward
            # ----------------------------------------------------------------
            clean_hs: list[torch.Tensor] = [None] * self.n_layers  # type: ignore

            capture_handles = []
            for layer_idx in range(self.n_layers):
                def make_capture(idx):
                    def hook(module, input, output):
                        hs = output[0] if isinstance(output, tuple) else output
                        clean_hs[idx] = hs.detach().clone()
                    return hook
                h = self.model.layers[layer_idx].register_forward_hook(make_capture(layer_idx))
                capture_handles.append(h)

            loss, clean_logits, pkv = self.model(input_ids)
            for h in capture_handles:
                h.remove()

            clean_probs = clean_logits[0, target_position].softmax(dim=-1)  # (V,)

            # ----------------------------------------------------------------
            # Step 2: Create corrupted input (add noise to embedding)
            # We use a shuffled version of the ids as the corruption.
            # ----------------------------------------------------------------
            perm = torch.randperm(T, device=device)
            corrupted_ids = input_ids[:, perm]  # (1, T) — shuffled token ids

            # Get corrupted baseline logits
            loss_c, corrupted_logits, _ = self.model(corrupted_ids)
            corrupted_probs = corrupted_logits[0, target_position].softmax(dim=-1)  # (V,)

            # ----------------------------------------------------------------
            # Step 3: For each (layer, position), patch clean hs and measure
            # the indirect effect = KL(clean || patched_corrupted)
            # ----------------------------------------------------------------
            effects = torch.zeros(self.n_layers, T, device=device)

            for layer_idx in range(self.n_layers):
                for pos in range(T):
                    # Build a hook that patches only position `pos` at `layer_idx`
                    clean_slice = clean_hs[layer_idx][:, pos, :]  # (1, D)

                    def make_patch_hook(c_slice, p):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h, rest = output[0], output[1:]
                                h = h.clone()
                                h[:, p, :] = c_slice
                                return (h,) + rest
                            else:
                                out = output.clone()
                                out[:, p, :] = c_slice
                                return out
                        return hook

                    handle = self.model.layers[layer_idx].register_forward_hook(
                        make_patch_hook(clean_slice, pos)
                    )
                    loss_p, patched_logits, _ = self.model(corrupted_ids)
                    handle.remove()

                    patched_probs = patched_logits[0, target_position].softmax(dim=-1)  # (V,)

                    # Indirect effect = L1 distance between patched and corrupted distributions
                    # (measures how much patching restored the clean output)
                    effect = (patched_probs - corrupted_probs).abs().sum()
                    effects[layer_idx, pos] = effect

        return effects
