"""Logit lens: project intermediate hidden states to vocab space to trace prediction formation across layers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LogitLensConfig:
    """Configuration for the logit lens analyzer."""

    n_top_tokens: int = 10
    """Top tokens to show per layer."""

    normalize_hidden: bool = True
    """Apply layer norm before projecting."""

    track_entropy: bool = True
    """Track prediction entropy per layer."""

    track_rank: bool = True
    """Track rank of final answer at each layer."""


class HiddenStateCollector:
    """Hooks into model layers to collect hidden states."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.hiddens: list[Tensor] = []
        self._hooks: list = []

    def register(self, layers: list[nn.Module]) -> None:
        """Register forward hooks on each layer.

        The hook captures the first output tensor (handles tuple returns).
        """
        for layer in layers:
            def _make_hook():
                def hook(module: nn.Module, inputs, output):
                    # Handle tuple outputs (e.g. (hidden, kv_cache))
                    if isinstance(output, (tuple, list)):
                        hidden = output[0]
                    else:
                        hidden = output
                    self.hiddens.append(hidden.detach())
                return hook

            handle = layer.register_forward_hook(_make_hook())
            self._hooks.append(handle)

    def clear(self) -> None:
        """Clear the collected hidden states list."""
        self.hiddens.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __enter__(self) -> "HiddenStateCollector":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()


def project_hidden_to_logits(
    hidden: Tensor,
    unembed_matrix: Tensor,
    layer_norm: nn.Module | None = None,
) -> Tensor:
    """Project hidden state (B, T, D) to vocab logits (B, T, V).

    Args:
        hidden: Hidden states of shape (B, T, D).
        unembed_matrix: Unembedding matrix of shape (V, D).
        layer_norm: Optional layer norm to apply before projecting.

    Returns:
        Logits of shape (B, T, V).
    """
    if layer_norm is not None:
        hidden = layer_norm(hidden)
    # unembed_matrix is (V, D), so hidden @ unembed_matrix.T gives (B, T, V)
    return hidden @ unembed_matrix.T


def get_top_tokens(logits: Tensor, n_top: int = 10) -> tuple[Tensor, Tensor]:
    """Get top-n tokens by logit value for the last position.

    Args:
        logits: Logits of shape (B, T, V).
        n_top: Number of top tokens to return.

    Returns:
        Tuple of (top_ids, top_logits), each of shape (B, n_top).
    """
    last_pos_logits = logits[:, -1, :]  # (B, V)
    top_logits, top_ids = torch.topk(last_pos_logits, n_top, dim=-1)
    return top_ids, top_logits


def compute_layer_entropy(logits: Tensor) -> Tensor:
    """Compute entropy of the softmax distribution at each position.

    Args:
        logits: Logits of shape (B, T, V).

    Returns:
        Entropy values of shape (B, T).
    """
    probs = F.softmax(logits, dim=-1)  # (B, T, V)
    log_probs = torch.log(probs + 1e-10)
    # H = -sum(p * log(p)) over vocab dim
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, T)
    return entropy


def compute_answer_rank(logits: Tensor, answer_token_id: int) -> Tensor:
    """Compute rank of answer_token_id in sorted logits at each layer.

    Lower rank = more confident prediction (rank 0 = top prediction).

    Args:
        logits: Logits of shape (B, T, V) — uses position -1.
        answer_token_id: The token whose rank to compute.

    Returns:
        Rank tensor of shape (B,).
    """
    last_pos_logits = logits[:, -1, :]  # (B, V)
    # Sort descending and find where answer_token_id falls
    sorted_ids = torch.argsort(last_pos_logits, dim=-1, descending=True)  # (B, V)
    # Find rank for each batch element
    answer = torch.tensor(answer_token_id, device=logits.device)
    # rank = position in sorted_ids where value == answer_token_id
    ranks = (sorted_ids == answer).nonzero(as_tuple=False)  # may be (B, 2)
    # ranks[:, 0] is batch index, ranks[:, 1] is rank position
    B = logits.shape[0]
    rank_tensor = torch.zeros(B, dtype=torch.long, device=logits.device)
    for row in ranks:
        batch_idx, rank_pos = int(row[0]), int(row[1])
        rank_tensor[batch_idx] = rank_pos
    return rank_tensor


class LogitLens:
    """Main logit lens analyzer.

    Projects intermediate layer hidden states through the unembedding matrix
    to reveal how predictions evolve across layers.
    """

    def __init__(self, model: nn.Module, config: LogitLensConfig) -> None:
        self.model = model
        self.config = config

        # Locate unembedding matrix — try common attribute names
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            self.unembed_matrix: Tensor = model.lm_head.weight  # (V, D)
        elif hasattr(model, "head") and hasattr(model.head, "weight"):
            self.unembed_matrix = model.head.weight
        elif hasattr(model, "embed") and hasattr(model.embed, "weight"):
            self.unembed_matrix = model.embed.weight
        else:
            raise AttributeError(
                "Cannot locate unembedding matrix. Expected model.lm_head.weight, "
                "model.head.weight, or model.embed.weight."
            )

        # Locate final layer norm if available
        self.layer_norm: nn.Module | None = None
        if config.normalize_hidden:
            for attr in ("norm", "final_norm", "ln_f"):
                if hasattr(model, attr):
                    self.layer_norm = getattr(model, attr)
                    break

        self.collector = HiddenStateCollector(model)

    def _get_layers(self) -> list[nn.Module]:
        """Return the list of transformer layer modules."""
        for attr in ("layers", "blocks", "transformer_blocks"):
            if hasattr(self.model, attr):
                layers = getattr(self.model, attr)
                return list(layers)
        raise AttributeError("Cannot find transformer layers on model.")

    @torch.no_grad()
    def analyze(
        self,
        input_ids: Tensor,
        answer_token_id: int | None = None,
    ) -> dict[str, Any]:
        """Run the logit lens analysis.

        Args:
            input_ids: Token ids of shape (B, T).
            answer_token_id: Optional token id to track rank across layers.

        Returns:
            Dictionary with keys:
                n_layers: int
                layer_entropies: list[float] — mean entropy per layer
                layer_top_tokens: list[list[int]] — top token ids per layer
                answer_ranks: list[int] | None
        """
        layers = self._get_layers()

        self.collector.clear()
        self.collector.register(layers)
        try:
            self.model(input_ids)
        finally:
            self.collector.remove_hooks()

        hiddens = self.collector.hiddens  # list of (B, T, D) tensors

        layer_entropies: list[float] = []
        layer_top_tokens: list[list[int]] = []
        answer_ranks: list[int] | None = [] if answer_token_id is not None else None

        for hidden in hiddens:
            logits = project_hidden_to_logits(
                hidden, self.unembed_matrix, self.layer_norm
            )  # (B, T, V)

            # Top tokens (use last position)
            top_ids, _ = get_top_tokens(logits, n_top=self.config.n_top_tokens)
            # top_ids: (B, n_top) — take batch 0
            layer_top_tokens.append(top_ids[0].tolist())

            # Entropy
            if self.config.track_entropy:
                entropy = compute_layer_entropy(logits)  # (B, T)
                layer_entropies.append(float(entropy.mean()))

            # Answer rank
            if self.config.track_rank and answer_token_id is not None:
                ranks = compute_answer_rank(logits, answer_token_id)  # (B,)
                answer_ranks.append(int(ranks[0]))

        return {
            "n_layers": len(hiddens),
            "layer_entropies": layer_entropies,
            "layer_top_tokens": layer_top_tokens,
            "answer_ranks": answer_ranks,
        }

    def layer_prediction_agreement(self, results: dict) -> float:
        """Fraction of layers where top-1 prediction matches the final layer's top-1.

        Args:
            results: Output from analyze().

        Returns:
            Float in [0, 1].
        """
        layer_top_tokens = results["layer_top_tokens"]
        if not layer_top_tokens:
            return 0.0

        final_top1 = layer_top_tokens[-1][0]
        n_agree = sum(1 for layer_tokens in layer_top_tokens if layer_tokens[0] == final_top1)
        return n_agree / len(layer_top_tokens)


def plot_logit_lens_text(
    results: dict,
    tokenizer_decode: Callable,
    n_layers_show: int = 4,
) -> str:
    """Render a simple text representation of logit lens results.

    Shows top-3 tokens at the first, middle, and last layers.

    Args:
        results: Output from LogitLens.analyze().
        tokenizer_decode: Callable that maps token id (int) -> str.
        n_layers_show: How many layers to show (not used beyond capping).

    Returns:
        Multi-line string representation.
    """
    layer_top_tokens = results["layer_top_tokens"]
    layer_entropies = results["layer_entropies"]
    n_layers = results["n_layers"]

    if n_layers == 0:
        return "(no layers captured)"

    # Pick indices: first, middle, last
    indices = sorted(
        set([0, n_layers // 2, n_layers - 1])
    )
    # If there are more layers than n_layers_show, trim to evenly spaced
    if len(indices) > n_layers_show:
        indices = indices[:n_layers_show]

    lines = [f"Logit Lens — {n_layers} layers"]
    lines.append("-" * 40)

    for idx in indices:
        entropy_str = ""
        if idx < len(layer_entropies):
            entropy_str = f"  entropy={layer_entropies[idx]:.3f}"
        top3 = layer_top_tokens[idx][:3]
        decoded = [tokenizer_decode(tid) for tid in top3]
        lines.append(f"Layer {idx:>3}{entropy_str}")
        lines.append(f"  top-3: {decoded}")

    lines.append("-" * 40)
    return "\n".join(lines)
