"""Logit Lens Analysis — project intermediate hidden states to vocabulary space.

Provides tools to inspect what a transformer "believes" at each layer by
projecting intermediate hidden states through the final norm + lm_head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LogitLensConfig:
    """Configuration for logit lens analysis.

    Attributes:
        layers: Which layer indices to analyze.  ``None`` means all layers.
        normalize: Whether to apply the model's final RMSNorm before projection.
        top_k: Number of top tokens to return per position.
    """

    layers: Optional[list[int]] = None
    normalize: bool = True
    top_k: int = 5


def extract_layer_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run a forward pass and capture hidden states after every transformer layer.

    Args:
        model: An ``AureliusTransformer`` instance.
        input_ids: (B, T) token ids.

    Returns:
        List of tensors, one per layer, each of shape (B, T, d_model).
    """
    hidden_states: list[torch.Tensor] = []
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(storage: list[torch.Tensor]):
        def hook_fn(module, input, output):
            # TransformerBlock.forward returns (x, kv)
            x = output[0] if isinstance(output, tuple) else output
            storage.append(x.detach())
        return hook_fn

    for layer in model.layers:
        h = layer.register_forward_hook(_make_hook(hidden_states))
        hooks.append(h)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return hidden_states


def project_to_vocab(
    hidden: torch.Tensor,
    model: nn.Module,
) -> torch.Tensor:
    """Project hidden states to vocabulary logits via final norm + lm_head.

    Args:
        hidden: (B, T, d_model) hidden states from an intermediate layer.
        model: An ``AureliusTransformer`` (must have ``.norm`` and ``.lm_head``).

    Returns:
        (B, T, V) logits over the vocabulary.
    """
    normed = model.norm(hidden)
    return model.lm_head(normed)


def get_top_tokens(
    logits: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-k token indices and their probabilities.

    Args:
        logits: (B, T, V) raw logits.
        k: Number of top tokens.

    Returns:
        Tuple of (indices, probs) each of shape (B, T, k).
    """
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
    return top_indices, top_probs


def compute_layer_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-position entropy of the softmax distribution.

    Args:
        logits: (B, T, V) raw logits.

    Returns:
        (B, T) entropy values (non-negative).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


class LogitLens:
    """High-level logit lens analyser.

    Args:
        model: An ``AureliusTransformer`` instance.
        config: Optional ``LogitLensConfig`` controlling which layers, top-k, etc.
    """

    def __init__(self, model: nn.Module, config: LogitLensConfig | None = None) -> None:
        self.model = model
        self.config = config or LogitLensConfig()

    def analyze(self, input_ids: torch.Tensor) -> dict:
        """Run logit lens analysis on the given input.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            Dictionary with keys:
                - ``layer_logits``: list of (B, T, V) tensors per selected layer.
                - ``layer_entropies``: list of (B, T) entropy tensors.
                - ``layer_top_tokens``: list of (indices, probs) tuples each (B, T, k).
                - ``convergence``: (num_layers-1,) cosine similarity between
                  consecutive layer logit distributions, averaged over batch and
                  sequence positions.
        """
        all_hidden = extract_layer_hidden_states(self.model, input_ids)

        # Determine which layers to analyse
        if self.config.layers is not None:
            selected_indices = [
                i for i in self.config.layers if 0 <= i < len(all_hidden)
            ]
        else:
            selected_indices = list(range(len(all_hidden)))

        selected_hidden = [all_hidden[i] for i in selected_indices]

        layer_logits: list[torch.Tensor] = []
        layer_entropies: list[torch.Tensor] = []
        layer_top_tokens: list[tuple[torch.Tensor, torch.Tensor]] = []

        for h in selected_hidden:
            if self.config.normalize:
                logits = project_to_vocab(h, self.model)
            else:
                logits = self.model.lm_head(h)
            layer_logits.append(logits)
            layer_entropies.append(compute_layer_entropy(logits))
            layer_top_tokens.append(get_top_tokens(logits, k=self.config.top_k))

        # Convergence: cosine similarity between consecutive layer softmax dists
        convergence = self._compute_convergence(layer_logits)

        return {
            "layer_logits": layer_logits,
            "layer_entropies": layer_entropies,
            "layer_top_tokens": layer_top_tokens,
            "convergence": convergence,
        }

    @staticmethod
    def _compute_convergence(layer_logits: list[torch.Tensor]) -> torch.Tensor:
        """Cosine similarity between consecutive layer probability distributions.

        Returns:
            1-D tensor of length ``len(layer_logits) - 1``.  Empty tensor if
            fewer than 2 layers.
        """
        if len(layer_logits) < 2:
            return torch.tensor([])

        similarities: list[torch.Tensor] = []
        for i in range(len(layer_logits) - 1):
            p1 = F.softmax(layer_logits[i], dim=-1)
            p2 = F.softmax(layer_logits[i + 1], dim=-1)
            # Flatten batch and seq dims
            p1_flat = p1.reshape(-1, p1.shape[-1])
            p2_flat = p2.reshape(-1, p2.shape[-1])
            cos_sim = F.cosine_similarity(p1_flat, p2_flat, dim=-1).mean()
            similarities.append(cos_sim)

        return torch.stack(similarities)
