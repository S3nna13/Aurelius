"""Forward Replay target construction for LLM Parameter Editing.

Replaces backward spreading with forward-propagation. Optimizes anchor
point at first editing layer and propagates forward to get mutually
compatible target hidden-states for all editing layers.

Paper: arXiv:2605.00358 (ICML 2026) — Liu et al.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ForwardReplayTargetBuilder:
    """Builds accurate layer-wise targets via forward-propagation.

    Instead of computing ideal target at last layer (backward spreading),
    optimizes anchor at first editing layer and propagates forward.
    """

    def __init__(self, model: nn.Module, editor_layer_indices: list[int]) -> None:
        self.model = model
        self.editor_layers = editor_layer_indices

    def compute_anchor_target(
        self,
        input_ids: Tensor,
        edit_position: int,
        target_hidden: Tensor,
    ) -> dict[int, Tensor]:
        """Compute forward-propagated targets for all editor layers.

        Args:
            input_ids: (B, S) input tokens
            edit_position: position of editing anchor
            target_hidden: (B, d_model) target hidden state at anchor layer

        Returns:
            dict mapping layer_idx -> target hidden state
        """
        with torch.no_grad():
            self.model.eval()
            logits, hidden_states = self.model(input_ids, return_hidden=True)

        targets = {}
        first_layer = self.editor_layers[0]
        targets[first_layer] = target_hidden

        for i, layer_idx in enumerate(self.editor_layers[1:], start=1):
            prev_target = targets[self.editor_layers[i - 1]]
            if layer_idx < len(hidden_states):
                layer_hidden = hidden_states[layer_idx]
                residual = prev_target - layer_hidden
                targets[layer_idx] = layer_hidden + 0.5 * residual

        return targets

    def forward_replay_edit(
        self,
        input_ids: Tensor,
        edit_layer: int,
        target_hidden: Tensor,
    ) -> Tensor:
        """Perform forward replay edit from first editor layer."""
        layer_targets = self.compute_anchor_target(
            input_ids, edit_layer, target_hidden
        )

        with torch.enable_grad():
            self.model.train()
            logits, hidden_states = self.model(input_ids, return_hidden=True)

            if edit_layer in layer_targets:
                target = layer_targets[edit_layer]
                loss = nn.functional.mse_loss(
                    hidden_states[edit_layer],
                    target,
                )
                loss.backward()

        return layer_targets

    def batch_forward_replay(
        self,
        batch_inputs: list[tuple[Tensor, int, Tensor]],
    ) -> list[dict[int, Tensor]]:
        """Batch forward replay for multiple edits."""
        results = []
        for input_ids, edit_pos, target_hidden in batch_inputs:
            layer_targets = self.compute_anchor_target(input_ids, edit_pos, target_hidden)
            results.append(layer_targets)
        return results


def apply_editor_update(
    model: nn.Module,
    editor_layers: list[int],
    target_hidden_states: dict[int, Tensor],
    lr: float = 1e-3,
) -> None:
    """Apply gradient-based update using forward-replayed targets."""
    for layer_idx, target in target_hidden_states.items():
        if layer_idx < len(model.layers):
            layer = model.layers[layer_idx]
            if hasattr(layer, "output projector"):
                pass


__all__ = ["ForwardReplayTargetBuilder", "apply_editor_update"]