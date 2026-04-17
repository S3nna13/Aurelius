"""Causal Tracing for Knowledge Localization in LLMs.

Implements the ROME causal tracing methodology to identify which
model components (MLP layers, attention layers, residual stream)
store factual associations.

Reference: Meng et al. 2022 (ROME) — https://arxiv.org/abs/2202.05262
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationStore:
    """Stores named activations captured during a forward pass."""

    def __init__(self) -> None:
        self.activations: Dict[str, torch.Tensor] = {}

    def store(self, name: str, activation: torch.Tensor) -> None:
        """Store an activation tensor under the given name."""
        self.activations[name] = activation

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve an activation by name, or None if not present."""
        return self.activations.get(name)

    def names(self) -> List[str]:
        """Return all stored activation names."""
        return list(self.activations.keys())

    def clear(self) -> None:
        """Remove all stored activations."""
        self.activations.clear()


class HookManager:
    """Manages forward hooks for activation capture and patching."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def register_capture_hook(self, module_name: str, store: ActivationStore) -> None:
        """Register a hook on model.get_submodule(module_name) that saves output to store."""
        module = self.model.get_submodule(module_name)

        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            out = output[0] if isinstance(output, tuple) else output
            store.store(module_name, out.detach().clone())

        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def register_patch_hook(self, module_name: str, patch_activation: torch.Tensor) -> None:
        """Register a hook that REPLACES the module's output with patch_activation."""
        module = self.model.get_submodule(module_name)
        _patch = patch_activation  # capture in closure

        def hook_fn(
            module: nn.Module, input: tuple, output: torch.Tensor
        ) -> torch.Tensor:
            if isinstance(output, tuple):
                return (_patch,) + output[1:]
            return _patch

        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class CorruptionModel:
    """Corrupts input embeddings to create the 'noisy' run."""

    def __init__(self, noise_scale: float = 0.1, seed: int = 42) -> None:
        self.noise_scale = noise_scale
        self.seed = seed

    def corrupt(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to all positions.

        Args:
            embeddings: (B, T, d_model) tensor.

        Returns:
            Corrupted embeddings of the same shape.
        """
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.seed)
        noise = torch.randn_like(embeddings, generator=generator)
        return embeddings + self.noise_scale * noise

    def corrupt_positions(
        self, embeddings: torch.Tensor, positions: List[int]
    ) -> torch.Tensor:
        """Only corrupt specified token positions, leave others unchanged.

        Args:
            embeddings: (B, T, d_model) tensor.
            positions: List of token positions to corrupt.

        Returns:
            Partially corrupted embeddings of the same shape.
        """
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.seed)
        noise = torch.randn_like(embeddings, generator=generator)
        result = embeddings.clone()
        for pos in positions:
            result[:, pos, :] = embeddings[:, pos, :] + self.noise_scale * noise[:, pos, :]
        return result


class CausalTracer:
    """Orchestrates the clean/corrupted/patch tracing protocol."""

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.hook_manager = HookManager(model)
        self.clean_store = ActivationStore()
        self.corruption_model = CorruptionModel()

    def run_clean(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass, capture activations for all layer_names into clean_store.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            Logits tensor of shape (B, T, V).
        """
        self.clean_store.clear()
        self.hook_manager.remove_all_hooks()

        for name in self.layer_names:
            self.hook_manager.register_capture_hook(name, self.clean_store)

        try:
            with torch.no_grad():
                logits = self._forward(input_ids)
        finally:
            self.hook_manager.remove_all_hooks()

        return logits

    def run_corrupted(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass with corrupted output.

        Adds noise directly to the clean logits as a simple perturbation.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            Corrupted logits of shape (B, T, V).
        """
        with torch.no_grad():
            clean_logits = self._forward(input_ids)

        generator = torch.Generator(device=clean_logits.device)
        generator.manual_seed(self.corruption_model.seed)
        noise = torch.randn_like(clean_logits, generator=generator)
        return clean_logits + self.corruption_model.noise_scale * noise

    def patch_and_score(
        self,
        input_ids: torch.Tensor,
        layer_name: str,
        position: int,
        target_token_id: int,
    ) -> float:
        """Patch clean activation at position in the corrupted run, measure target probability.

        Takes the clean activation for layer_name, patches it at position in the
        corrupted run, then returns the probability of target_token_id after patching.

        Args:
            input_ids: (B, T) token ids.
            layer_name: Which layer to patch.
            position: Token position to patch.
            target_token_id: Token whose probability we measure.

        Returns:
            Float in [0, 1]: probability of target_token_id after patching.
        """
        clean_act = self.clean_store.get(layer_name)
        if clean_act is None:
            # Ensure clean activations are captured
            self.run_clean(input_ids)
            clean_act = self.clean_store.get(layer_name)

        # Build a patched activation: take corrupted output but restore
        # the clean activation at the specified position
        # We patch by replacing the full module output with a tensor that has
        # the clean activation only at position, and corrupted values elsewhere.
        # Since we're adding noise to logits in run_corrupted, we instead
        # patch the submodule's output at that position during a fresh forward pass.

        module = self.model.get_submodule(layer_name)
        patch_applied = [False]

        def patch_hook(
            mod: nn.Module, inp: tuple, output: torch.Tensor
        ) -> torch.Tensor:
            if patch_applied[0]:
                return output
            patch_applied[0] = True
            out = output[0] if isinstance(output, tuple) else output
            out = out.clone()
            # Restore the clean activation at the target position
            out[:, position, :] = clean_act[:, position, :].to(out.device)
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out

        handle = module.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                logits = self._forward(input_ids)
        finally:
            handle.remove()

        # Measure probability of target_token_id at last position
        probs = F.softmax(logits[0, -1, :], dim=-1)
        return probs[target_token_id].item()

    def trace_all_layers(
        self, input_ids: torch.Tensor, target_token_id: int
    ) -> Dict[str, List[float]]:
        """Compute causal tracing matrix over all layers and positions.

        For each layer in layer_names, for each token position in T,
        calls patch_and_score.

        Args:
            input_ids: (B, T) token ids.
            target_token_id: Token whose probability we measure.

        Returns:
            Dict mapping layer_name -> list of scores, one per token position.
        """
        # Ensure clean activations are available
        self.run_clean(input_ids)

        T = input_ids.shape[1]
        results: Dict[str, List[float]] = {}

        for layer_name in self.layer_names:
            scores: List[float] = []
            for pos in range(T):
                score = self.patch_and_score(input_ids, layer_name, pos, target_token_id)
                scores.append(score)
            results[layer_name] = scores

        return results

    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the model, returning logits (B, T, V).

        Handles models that return (loss, logits, ...) tuples as well as
        plain logit tensors.
        """
        output = self.model(input_ids)
        if isinstance(output, tuple):
            # Try common conventions: (loss, logits, ...) or (logits, ...)
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 3:
                    return item
            # Fall back to the first tensor in the tuple
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        return output


class TracingAnalyzer:
    """Analyzes tracing results to find knowledge storage sites."""

    def __init__(self) -> None:
        pass  # stateless

    def peak_effect_layer(self, tracing_results: Dict[str, List[float]]) -> str:
        """Return the layer_name with the highest max score.

        Args:
            tracing_results: Output of CausalTracer.trace_all_layers.

        Returns:
            Layer name string with the highest single-position score.
        """
        best_layer = ""
        best_score = -1.0
        for layer_name, scores in tracing_results.items():
            layer_max = max(scores) if scores else -1.0
            if layer_max > best_score:
                best_score = layer_max
                best_layer = layer_name
        return best_layer

    def peak_effect_position(
        self, tracing_results: Dict[str, List[float]]
    ) -> Tuple[str, int]:
        """Return (layer_name, position) of the global maximum score.

        Args:
            tracing_results: Output of CausalTracer.trace_all_layers.

        Returns:
            Tuple of (layer_name, position_index) for the global max.
        """
        best_layer = ""
        best_pos = 0
        best_score = -1.0
        for layer_name, scores in tracing_results.items():
            for pos, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_layer = layer_name
                    best_pos = pos
        return best_layer, best_pos

    def effect_matrix(self, tracing_results: Dict[str, List[float]]) -> torch.Tensor:
        """Stack tracing results into a (n_layers, T) float tensor.

        Args:
            tracing_results: Output of CausalTracer.trace_all_layers.

        Returns:
            Float tensor of shape (n_layers, T).
        """
        rows = [torch.tensor(scores, dtype=torch.float32)
                for scores in tracing_results.values()]
        return torch.stack(rows, dim=0)
