"""Logit Lens: project intermediate hidden states to vocabulary space.

Reference: nostalgebraist (2020) — "interpreting GPT: the logit lens"
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/

Each transformer layer produces a hidden state.  By projecting that hidden
state through the final unembedding matrix (optionally preceded by a layer
norm) we can read off what token the model "predicts" at every intermediate
layer, giving insight into how predictions form through the network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LogitLensConfig:
    """Configuration for the logit lens analysis.

    Attributes:
        n_layers:       Number of transformer layers in the model.
        d_model:        Hidden dimension of the model.
        vocab_size:     Vocabulary size (number of token types).
        use_layer_norm: Whether to apply layer norm before projecting hidden
                        states through the unembedding matrix.
    """

    n_layers: int
    d_model: int
    vocab_size: int
    use_layer_norm: bool = True


# ---------------------------------------------------------------------------
# Core lens
# ---------------------------------------------------------------------------


class LogitLens:
    """Projects a single layer's hidden state to vocabulary logits.

    Args:
        unembed:    An ``nn.Linear`` whose weight matrix has shape *(V, d)*.
                    The projection computes ``h @ unembed.weight.T`` (plus
                    bias if present), matching the standard LM head.
        layer_norm: Optional normalisation module applied to the hidden state
                    before projection.  Typically the model's final layer norm
                    (e.g. ``nn.LayerNorm``).
    """

    def __init__(
        self,
        unembed: nn.Linear,
        layer_norm: Optional[nn.Module] = None,
    ) -> None:
        self.unembed = unembed
        self.layer_norm = layer_norm

    # ------------------------------------------------------------------
    # Primary operations
    # ------------------------------------------------------------------

    def project(self, hidden: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden: Float tensor of shape *(B, T, d)*.

        Returns:
            Float tensor of shape *(B, T, V)* containing unnormalised logits.
        """
        h = hidden
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        # unembed.weight: (V, d)  →  h @ W^T: (B, T, V)
        logits = F.linear(h, self.unembed.weight, self.unembed.bias)
        return logits

    def top_k_tokens(self, hidden: Tensor, k: int = 5) -> torch.LongTensor:
        """Return the indices of the top-*k* predicted tokens.

        Args:
            hidden: Float tensor of shape *(B, T, d)*.
            k:      Number of top tokens to return per position.

        Returns:
            Long tensor of shape *(B, T, k)* with token indices.
        """
        logits = self.project(hidden)  # (B, T, V)
        _, top_indices = torch.topk(logits, k, dim=-1)  # (B, T, k)
        return top_indices

    def entropy(self, hidden: Tensor) -> Tensor:
        """Compute the Shannon entropy of the token distribution.

        Entropy is computed from the softmax of the projected logits.  A
        uniform distribution has maximum entropy; a one-hot (peaked)
        distribution has zero entropy.

        Args:
            hidden: Float tensor of shape *(B, T, d)*.

        Returns:
            Float tensor of shape *(B, T)* with entropy in nats (≥ 0).
        """
        logits = self.project(hidden)  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # numerically stable
        probs = log_probs.exp()
        # H = -sum_v p_v * log(p_v)
        ent = -(probs * log_probs).sum(dim=-1)  # (B, T)
        return ent


# ---------------------------------------------------------------------------
# Multi-layer analyser
# ---------------------------------------------------------------------------


class LogitLensAnalyzer:
    """Runs the logit lens over all collected hidden states.

    Args:
        lenses: One :class:`LogitLens` per layer.  The *i*-th lens is applied
                to the hidden state produced by layer *i*.
    """

    def __init__(self, lenses: List[LogitLens]) -> None:
        self.lenses = lenses

    # ------------------------------------------------------------------

    def analyze(
        self,
        layer_hiddens: List[Tensor],
        layer_idx: Optional[List[int]] = None,
    ) -> Dict:
        """Project every layer's hidden state and collect results.

        Args:
            layer_hiddens: List of *(B, T, d)* tensors, one per layer.
            layer_idx:     Optional list of layer indices to process.  If
                           ``None`` all layers are processed in order.

        Returns:
            Dictionary with keys:

            * ``"logits"``  – ``List[Tensor(B, T, V)]``
            * ``"top1"``    – ``List[LongTensor(B, T)]``
            * ``"entropy"`` – ``List[Tensor(B, T)]``
        """
        if layer_idx is None:
            layer_idx = list(range(len(layer_hiddens)))

        logits_list: List[Tensor] = []
        top1_list: List[torch.LongTensor] = []
        entropy_list: List[Tensor] = []

        for i, li in enumerate(layer_idx):
            lens = self.lenses[li] if li < len(self.lenses) else self.lenses[i]
            hidden = layer_hiddens[i]

            logits = lens.project(hidden)          # (B, T, V)
            top1 = logits.argmax(dim=-1)           # (B, T)
            ent = lens.entropy(hidden)             # (B, T)

            logits_list.append(logits)
            top1_list.append(top1)
            entropy_list.append(ent)

        return {
            "logits": logits_list,
            "top1": top1_list,
            "entropy": entropy_list,
        }

    # ------------------------------------------------------------------

    def rank_of_true_token(
        self,
        layer_hiddens: List[Tensor],
        true_tokens: torch.LongTensor,
    ) -> List[Tensor]:
        """Compute the rank of the true token in each layer's projection.

        Rank 0 means the true token is the top-predicted token; rank *V-1*
        means it is the least likely token according to the projected logits.

        Args:
            layer_hiddens: List of *(B, T, d)* tensors, one per layer.
            true_tokens:   Long tensor of shape *(B, T)* with ground-truth
                           token indices.

        Returns:
            List of long tensors of shape *(B, T)*, one per layer.
        """
        ranks_per_layer: List[Tensor] = []

        for i, (hidden, lens) in enumerate(zip(layer_hiddens, self.lenses)):
            logits = lens.project(hidden)  # (B, T, V)

            # argsort descending: position 0 has the highest logit token
            sorted_indices = logits.argsort(dim=-1, descending=True)  # (B, T, V)

            # For each (b, t) position find where true_tokens[b, t] appears
            # in sorted_indices[b, t, :].
            # ranks[b, t] = (sorted_indices[b, t, :] == true_tokens[b, t]).nonzero()[0]
            B, T, V = sorted_indices.shape
            true_expanded = true_tokens.unsqueeze(-1).expand(B, T, V)  # (B, T, V)
            matches = (sorted_indices == true_expanded)  # (B, T, V) bool
            # argmax along last dim gives the first True index = rank
            rank = matches.long().argmax(dim=-1)  # (B, T)

            ranks_per_layer.append(rank)

        return ranks_per_layer


# ---------------------------------------------------------------------------
# Tracker (context-manager hook collector)
# ---------------------------------------------------------------------------


class LogitLensTracker:
    """Attaches forward hooks to model layers to collect hidden states.

    Designed for use as a context manager::

        tracker = LogitLensTracker(model.layers)
        with tracker:
            output = model(input_ids)
        hiddens = tracker.get_hiddens()  # List[Tensor], one per layer

    Args:
        layers: List of ``nn.Module`` objects to hook (one per layer).
    """

    def __init__(self, layers: List[nn.Module]) -> None:
        self.layers = layers
        self._hiddens: List[Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "LogitLensTracker":
        self._install_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._remove_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        """Register a forward hook on every tracked layer."""
        for layer in self.layers:
            def _make_hook():
                def _hook(module: nn.Module, inputs, output):
                    # Some layers return tuples (hidden, cache, …); grab first element.
                    if isinstance(output, (tuple, list)):
                        h = output[0]
                    else:
                        h = output
                    self._hiddens.append(h.detach())

                return _hook

            handle = layer.register_forward_hook(_make_hook())
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_hiddens(self) -> List[Tensor]:
        """Return collected hidden states (one tensor per layer per call).

        Returns:
            List of tensors in the order the hooks fired.
        """
        return list(self._hiddens)

    def clear(self) -> None:
        """Discard all collected hidden states."""
        self._hiddens.clear()
