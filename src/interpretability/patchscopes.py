"""
Patchscopes interpretability tool for the Aurelius LLM project.

Implements the Patchscopes framework from:
    "Patchscopes: A Unifying Framework for Inspecting Hidden Representations"
    (arXiv:2401.06102)

A Patchscope extracts a hidden state from a source model at a chosen
(layer, position) and injects it into a target model at another chosen
(layer, position) to probe what information the hidden state encodes.

Both source and target can be the same model instance (the typical single-model
interpretability setting).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PatchscopeConfig:
    """Configuration specifying where to extract and inject hidden states.

    Negative indices follow Python convention (-1 = last layer/token).
    Actual index resolution is deferred to runtime so the config is
    model-agnostic.
    """

    source_layer: int = -1  # layer from which to extract hidden state
    source_position: int = -1  # token position to extract (-1 = last token)
    target_layer: int = 0  # layer at which to inject the hidden state
    target_position: int = -1  # token position at which to inject (-1 = last)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_index(idx: int, length: int) -> int:
    """Resolve a possibly-negative index into [0, length)."""
    if idx < 0:
        idx = length + idx
    if not (0 <= idx < length):
        raise IndexError(f"Index {idx} is out of range for dimension of size {length}.")
    return idx


# ---------------------------------------------------------------------------
# Patchscope
# ---------------------------------------------------------------------------


class Patchscope:
    """Patchscope: extract and re-inject hidden states via forward hooks.

    Parameters
    ----------
    model:
        An AureliusTransformer (or any nn.Module with a ``layers`` ModuleList
        of TransformerBlock-like objects whose forward returns ``(hidden, kv)``).
    config:
        PatchscopeConfig describing where to extract and inject.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PatchscopeConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PatchscopeConfig()

        # Number of layers derived from the model at construction time.
        self._n_layers: int = len(model.layers)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Private: resolved indices
    # ------------------------------------------------------------------

    def _src_layer(self) -> int:
        return _resolve_index(self.config.source_layer, self._n_layers)

    def _tgt_layer(self) -> int:
        return _resolve_index(self.config.target_layer, self._n_layers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(self, input_ids: Tensor) -> Tensor:
        """Run the model on *input_ids* and return the hidden state at the
        configured (source_layer, source_position).

        Parameters
        ----------
        input_ids:
            (B, T) token-id tensor.

        Returns
        -------
        Tensor of shape (B, d_model) — the extracted hidden state.
        """
        B, T = input_ids.shape
        src_layer_idx = self._src_layer()
        src_pos_idx = _resolve_index(self.config.source_position, T)

        # Container mutated by the hook closure.
        captured: list[Tensor] = []

        def _extraction_hook(
            module: nn.Module,
            inputs: tuple,
            output: tuple,
        ) -> None:
            # TransformerBlock.forward returns (hidden, kv).
            hidden = output[0]  # (B, T, d_model)
            captured.append(hidden[:, src_pos_idx, :].detach().clone())

        handle = self.model.layers[src_layer_idx].register_forward_hook(  # type: ignore[index]
            _extraction_hook
        )
        try:
            self.model(input_ids)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Extraction hook was never called — forward pass failed.")

        return captured[0]  # (B, d_model)

    @torch.no_grad()
    def inject_and_decode(
        self,
        target_ids: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        """Run the model on *target_ids*, injecting *hidden* at the configured
        (target_layer, target_position), and return the full output logits.

        Parameters
        ----------
        target_ids:
            (B, T) token-id tensor for the target context.
        hidden:
            (B, d_model) hidden state to inject (e.g. from :meth:`extract`).

        Returns
        -------
        Tensor of shape (B, T, vocab_size) — output logits.
        """
        B, T = target_ids.shape
        tgt_layer_idx = self._tgt_layer()
        tgt_pos_idx = _resolve_index(self.config.target_position, T)

        def _injection_hook(
            module: nn.Module,
            inputs: tuple,
            output: tuple,
        ) -> tuple:
            # output is (hidden_state, kv, aux_loss) from TransformerBlock.
            h, kv, aux = output[0], output[1], output[2]  # h: (B, T, d_model)
            h = h.clone()
            h[:, tgt_pos_idx, :] = hidden
            return (h, kv, aux)

        handle = self.model.layers[tgt_layer_idx].register_forward_hook(  # type: ignore[index]
            _injection_hook
        )
        try:
            _, logits, _ = self.model(target_ids)
        finally:
            handle.remove()

        return logits  # (B, T, vocab_size)

    @torch.no_grad()
    def run(
        self,
        source_ids: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        """Full Patchscope pipeline: extract then inject-and-decode.

        Parameters
        ----------
        source_ids:
            (B, T_src) token ids for the source context.
        target_ids:
            (B, T_tgt) token ids for the target context.

        Returns
        -------
        Tensor of shape (B, T_tgt, vocab_size) — output logits after injection.
        """
        hidden = self.extract(source_ids)  # (B, d_model)
        return self.inject_and_decode(target_ids, hidden)  # (B, T_tgt, vocab_size)
