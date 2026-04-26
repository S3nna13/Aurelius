"""Activation patching (causal tracing) for mechanistic interpretability.

Identifies which layer+position computations causally mediate model outputs.

Algorithm:
1. Run clean forward pass, cache hidden states at each layer
2. Run corrupted forward pass, patch in clean hidden states at one layer+position
3. Measure change in output probability at the target token
4. Repeat for all (layer, position) combinations → causal importance grid

Reference: Meng et al. 2022 "Locating and Editing Factual Associations in GPT"
           (ROME, arXiv:2202.05262)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchingResult:
    """Result of a single activation patching experiment."""

    layer: int
    position: int
    clean_prob: float  # P(target_token | clean input)
    corrupted_prob: float  # P(target_token | corrupted input, no patching)
    patched_prob: float  # P(target_token | corrupted + patch at this layer/pos)

    @property
    def restoration_score(self) -> float:
        """How much the patch restored the clean probability.

        0 = no restoration, 1 = full restoration.
        Clamped to [0, 1].
        """
        denom = self.clean_prob - self.corrupted_prob
        if abs(denom) < 1e-8:
            return 0.0
        return max(0.0, min(1.0, (self.patched_prob - self.corrupted_prob) / denom))


def _get_prob(model: nn.Module, input_ids: torch.Tensor, target_token_id: int) -> float:
    """Run a forward pass and return P(target_token_id) at the last position."""
    with torch.no_grad():
        _, logits, _ = model(input_ids)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs[target_token_id].item()


def get_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,  # (1, S)
) -> dict[int, torch.Tensor]:
    """Run model forward, return hidden states at each layer.

    Returns dict: layer_idx → (1, S, D) hidden state tensor.
    Uses forward hooks on model.layers[i].
    """
    hidden_states: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # TransformerBlock returns (hidden_state, kv_cache)
            hs = output[0] if isinstance(output, tuple) else output
            hidden_states[layer_idx] = hs.detach().clone()

        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return hidden_states


def patch_and_forward(
    model: nn.Module,
    corrupted_ids: torch.Tensor,  # (1, S)
    patch_hidden: torch.Tensor,  # (1, S, D) — clean hidden state to inject
    layer_idx: int,
    position: int,  # token position to patch
    target_token_id: int,  # which token's probability to measure
) -> float:
    """Run corrupted forward pass with one activation patched.

    At layer_idx, replaces hidden_state[:, position, :] with patch_hidden[:, position, :].
    Returns P(target_token_id) at the LAST position (next-token prediction).
    """
    called = [False]

    def patching_hook(module, input, output):
        if called[0]:
            return output
        called[0] = True
        hs = output[0] if isinstance(output, tuple) else output
        # Clone to avoid in-place modification issues
        hs = hs.clone()
        hs[:, position, :] = patch_hidden[:, position, :].to(hs.device)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    hook = model.layers[layer_idx].register_forward_hook(patching_hook)
    try:
        with torch.no_grad():
            _, logits, _ = model(corrupted_ids)
    finally:
        hook.remove()

    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs[target_token_id].item()


def causal_trace(
    model: nn.Module,
    clean_ids: torch.Tensor,  # (1, S)
    corrupted_ids: torch.Tensor,  # (1, S) — same structure, different tokens
    target_token_id: int,  # token we're measuring probability of
    layers: list[int] | None = None,  # which layers to probe (None = all)
    positions: list[int] | None = None,  # which positions to probe (None = all)
) -> list[PatchingResult]:
    """Run full causal tracing experiment.

    Returns list of PatchingResult for each (layer, position) probed.
    """
    n_layers = len(model.layers)
    seq_len = clean_ids.shape[1]

    if layers is None:
        layers = list(range(n_layers))
    if positions is None:
        positions = list(range(seq_len))

    # Step 1: Cache clean hidden states
    clean_hs = get_hidden_states(model, clean_ids)

    # Step 2: Measure clean probability
    clean_prob = _get_prob(model, clean_ids, target_token_id)

    # Step 3: Measure corrupted probability (no patching)
    corrupted_prob = _get_prob(model, corrupted_ids, target_token_id)

    # Step 4: For each (layer, position), patch and measure
    results: list[PatchingResult] = []
    for layer_idx in layers:
        for pos in positions:
            patched_prob = patch_and_forward(
                model=model,
                corrupted_ids=corrupted_ids,
                patch_hidden=clean_hs[layer_idx],
                layer_idx=layer_idx,
                position=pos,
                target_token_id=target_token_id,
            )
            results.append(
                PatchingResult(
                    layer=layer_idx,
                    position=pos,
                    clean_prob=clean_prob,
                    corrupted_prob=corrupted_prob,
                    patched_prob=patched_prob,
                )
            )

    return results
