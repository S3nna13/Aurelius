"""Higher-level model editing API wrapping rank-one updates.

Provides a clean interface for applying, reverting, and evaluating
rank-one fact edits (ROME-style) to the FFN down_proj weights of
AureliusTransformer layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.transformer import AureliusTransformer


@dataclass
class EditConfig:
    """Configuration for model editing."""
    n_edits: int = 1
    layer_idx: int = -1          # which layer's FFN to edit; -1 = last layer
    edit_method: str = "rank1"   # "rank1" | "ft"
    learning_rate: float = 1e-4
    n_steps: int = 10


@dataclass
class FactEdit:
    """Specification for a single fact edit."""
    subject_ids: torch.Tensor    # (1, S) token ids for subject
    target_ids: torch.Tensor     # (1, T) token ids for target
    relation: str = ""


def _resolve_layer_idx(model: AureliusTransformer, layer_idx: int) -> int:
    """Resolve a possibly-negative layer index to a non-negative one."""
    n = len(model.layers)
    if layer_idx < 0:
        return n + layer_idx
    return layer_idx


def compute_key_vector(
    model: AureliusTransformer,
    subject_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Extract hidden state at layer_idx for subject by running a hooked forward.

    Returns the mean over the sequence dimension, shape (d_model,).

    Args:
        model: AureliusTransformer instance.
        subject_ids: (1, S) integer token ids.
        layer_idx: Which layer to extract from (supports negative indexing).

    Returns:
        Tensor of shape (d_model,).
    """
    resolved = _resolve_layer_idx(model, layer_idx)
    hidden_states: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, inp, out) -> None:
        # TransformerBlock returns (hidden_state, kv_cache) tuple
        h = out[0] if isinstance(out, (tuple, list)) else out
        hidden_states.append(h.detach())

    target_layer = model.layers[resolved]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            model(subject_ids)
    finally:
        handle.remove()

    # hidden_states[0] is (B, S, d_model); mean over S -> (d_model,)
    return hidden_states[0].squeeze(0).mean(dim=0)


def rank_one_update(
    weight: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    """Compute and apply a rank-1 update to weight.

    Update rule:
        W_new = W + outer(v - W @ k, k) / (k @ k + lambda_reg)

    Args:
        weight: (out_features, in_features) weight matrix.
        key:    (in_features,) key vector k.
        value:  (out_features,) value vector v.
        lambda_reg: Regularisation to prevent division by near-zero.

    Returns:
        Updated weight tensor, same shape as weight.
    """
    residual = value - weight @ key
    denom = key @ key + lambda_reg
    delta = torch.outer(residual, key) / denom
    return weight + delta


class ModelEditor:
    """Applies, tracks, and reverts rank-one edits to an AureliusTransformer."""

    def __init__(self, model: AureliusTransformer, config: EditConfig):
        self.model = model
        self.config = config
        self._originals: dict[int, torch.Tensor] = {}

    def _get_down_proj(self, resolved_idx: int) -> nn.Linear:
        return self.model.layers[resolved_idx].ffn.down_proj

    def _extract_ffn_vectors(
        self,
        edit: FactEdit,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract key (d_ff,) and value (d_model,) for rank-1 down_proj update."""
        ffn = self.model.layers[layer_idx].ffn

        subject_acts: list[torch.Tensor] = []

        def subject_hook(module, inp, out):
            subject_acts.append(inp[0].detach().squeeze(0).mean(dim=0))

        h = ffn.down_proj.register_forward_hook(subject_hook)
        with torch.no_grad():
            self.model(edit.subject_ids)
        h.remove()

        key_ff = subject_acts[0]    # (d_ff,)

        val_hidden: list[torch.Tensor] = []

        def hidden_hook(module, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            val_hidden.append(h.detach().squeeze(0).mean(dim=0))

        h = self.model.layers[layer_idx].register_forward_hook(hidden_hook)
        with torch.no_grad():
            self.model(edit.target_ids)
        h.remove()

        val_ff = val_hidden[0]      # (d_model,)
        return key_ff, val_ff

    def apply_edit(self, edit: FactEdit) -> dict:
        """Compute key/value vectors and apply rank_one_update to FFN down_proj.

        Args:
            edit: FactEdit specifying subject and target token ids.

        Returns:
            dict with keys "layer" (int) and "edit_norm" (float).
        """
        layer_idx = _resolve_layer_idx(self.model, self.config.layer_idx)
        down_proj = self._get_down_proj(layer_idx)

        if layer_idx not in self._originals:
            self._originals[layer_idx] = down_proj.weight.data.clone()

        key_ff, val_ff = self._extract_ffn_vectors(edit, layer_idx)

        W = down_proj.weight.data  # (d_model, d_ff)
        W_new = rank_one_update(W, key_ff, val_ff, lambda_reg=0.1)
        edit_norm = (W_new - W).norm().item()
        down_proj.weight.data = W_new

        return {"layer": layer_idx, "edit_norm": edit_norm}

    def revert_edit(self, layer_idx: int) -> None:
        """Restore the original down_proj weight at layer_idx.

        Args:
            layer_idx: Layer index (supports negative indexing).
        """
        resolved = _resolve_layer_idx(self.model, layer_idx)
        if resolved in self._originals:
            self._get_down_proj(resolved).weight.data = self._originals[resolved].clone()
            del self._originals[resolved]

    def evaluate_edit(self, edit: FactEdit) -> dict:
        """Check whether the model now generates target_ids given subject_ids.

        Args:
            edit: FactEdit with subject_ids and target_ids.

        Returns:
            dict with keys "success" (bool) and "score" (float in [0, 1]).
        """
        self.model.train(False)
        target_ids = edit.target_ids  # (1, T)
        T = target_ids.shape[1]

        context = edit.subject_ids.clone()  # (1, S)
        correct = 0
        with torch.no_grad():
            for t in range(T):
                _, logits, _ = self.model(context)  # logits: (1, ctx_len, vocab)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
                if next_token.item() == target_ids[0, t].item():
                    correct += 1
                context = torch.cat([context, next_token], dim=1)

        score = correct / T if T > 0 else 0.0
        return {"success": bool(score > 0.5), "score": float(score)}

    def batch_edit(self, edits: list[FactEdit]) -> list[dict]:
        """Apply multiple edits sequentially.

        Args:
            edits: List of FactEdit instances.

        Returns:
            List of result dicts (one per edit).
        """
        results = []
        for edit in edits:
            results.append(self.apply_edit(edit))
        return results
