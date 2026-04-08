"""Logit Lens interpretability tool for Aurelius.

Decodes intermediate layer representations through the final unembedding matrix
to reveal how predictions evolve through the network layers.

Reference: nostalgebraist (2020) — "interpreting GPT: the logit lens"
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.model.transformer import AureliusTransformer, TransformerBlock


class LogitLens:
    """Logit lens for probing intermediate representations.

    Takes the hidden state at each TransformerBlock output, applies the final
    layer norm and unembedding matrix (lm_head), and decodes the resulting
    distribution. This reveals how predictions evolve through the network.

    Args:
        model: AureliusTransformer — must have .norm (final RMSNorm) and .lm_head.
        tokenizer: optional — if provided, enables decode_top_tokens().
    """

    def __init__(self, model: AureliusTransformer, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: list = []
        self._layer_hiddens: dict[int, torch.Tensor] = {}
        self._layer_logits: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        """Register forward hooks on each TransformerBlock to capture hidden states."""
        self._layer_hiddens.clear()
        self._layer_logits.clear()

        for layer_idx, layer in enumerate(self.model.layers):
            def make_hook(idx: int):
                def hook(module: TransformerBlock, inputs, output):
                    # TransformerBlock returns (hidden, kv_cache); grab hidden
                    hidden = output[0]  # (B, S, d_model)
                    self._layer_hiddens[idx] = hidden.detach()
                return hook

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run forward pass and return logits at each layer.

        Args:
            input_ids: (batch, seq_len) token indices.

        Returns:
            layer_logits: dict mapping layer_idx -> (batch, seq_len, vocab_size)
                          logits computed by applying final norm + lm_head to each
                          layer's output hidden state.
        """
        self._install_hooks()
        try:
            self.model(input_ids)
        finally:
            self._remove_hooks()

        # Apply final norm + lm_head to each captured hidden state
        self._layer_logits.clear()
        for layer_idx, hidden in self._layer_hiddens.items():
            normed = self.model.norm(hidden)
            logits = self.model.lm_head(normed)  # (B, S, vocab_size)
            self._layer_logits[layer_idx] = logits

        return dict(self._layer_logits)

    # ------------------------------------------------------------------
    # Analysis methods (must be called after run())
    # ------------------------------------------------------------------

    def top_tokens_at_layer(
        self,
        layer_idx: int,
        position: int,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Get top-k (token_id, probability) at a given layer and position.

        Must be called after run().

        Args:
            layer_idx: Which layer's logits to query.
            position: Token position in the sequence.
            k: Number of top tokens to return.

        Returns:
            List of (token_id, probability) tuples, sorted by probability descending.
        """
        if layer_idx not in self._layer_logits:
            raise RuntimeError("Call run() before top_tokens_at_layer().")

        logits = self._layer_logits[layer_idx]  # (B, S, vocab_size)
        # Use batch index 0
        logits_at_pos = logits[0, position, :]  # (vocab_size,)
        probs = F.softmax(logits_at_pos, dim=-1)

        topk_probs, topk_ids = torch.topk(probs, k)
        return [(int(tid), float(prob)) for tid, prob in zip(topk_ids, topk_probs)]

    def probability_of_token(
        self,
        target_token_id: int,
        position: int,
    ) -> list[float]:
        """Return probability of target_token_id at each layer at a given position.

        Must be called after run().

        Args:
            target_token_id: The vocabulary token ID to track.
            position: Token position in the sequence.

        Returns:
            List of floats of length n_layers, one probability per layer.
        """
        if not self._layer_logits:
            raise RuntimeError("Call run() before probability_of_token().")

        probs_per_layer: list[float] = []
        n_layers = len(self._layer_logits)
        for layer_idx in range(n_layers):
            logits = self._layer_logits[layer_idx]  # (B, S, vocab_size)
            logits_at_pos = logits[0, position, :]
            probs = F.softmax(logits_at_pos, dim=-1)
            probs_per_layer.append(float(probs[target_token_id]))

        return probs_per_layer

    def entropy_profile(self, position: int) -> list[float]:
        """Return entropy of the predictive distribution at each layer for a position.

        High early entropy indicates model uncertainty; decreasing entropy
        indicates the model is becoming more confident.

        Args:
            position: Token position in the sequence.

        Returns:
            List of floats of length n_layers.
        """
        if not self._layer_logits:
            raise RuntimeError("Call run() before entropy_profile().")

        entropies: list[float] = []
        n_layers = len(self._layer_logits)
        for layer_idx in range(n_layers):
            logits = self._layer_logits[layer_idx]  # (B, S, vocab_size)
            logits_at_pos = logits[0, position, :]
            probs = F.softmax(logits_at_pos, dim=-1)
            # Shannon entropy: H = -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = float(-torch.sum(probs * log_probs))
            entropies.append(entropy)

        return entropies

    def plot_data(self, position: int, top_k: int = 5) -> dict:
        """Return structured data for visualization.

        Args:
            position: Token position in the sequence.
            top_k: Number of top tokens to return per layer.

        Returns:
            {
                'layers': [0, 1, ..., n_layers-1],
                'top_tokens_per_layer': [[{token_id, prob}, ...], ...],
                'entropy_per_layer': [float, ...],
            }
        """
        if not self._layer_logits:
            raise RuntimeError("Call run() before plot_data().")

        n_layers = len(self._layer_logits)
        layers = list(range(n_layers))

        top_tokens_per_layer = []
        for layer_idx in layers:
            top_tokens = self.top_tokens_at_layer(layer_idx, position, k=top_k)
            top_tokens_per_layer.append(
                [{"token_id": tid, "prob": prob} for tid, prob in top_tokens]
            )

        entropy_per_layer = self.entropy_profile(position)

        return {
            "layers": layers,
            "top_tokens_per_layer": top_tokens_per_layer,
            "entropy_per_layer": entropy_per_layer,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


@torch.no_grad()
def logit_lens_summary(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    target_position: int = -1,
) -> dict:
    """Run logit lens and return summary statistics.

    Args:
        model: AureliusTransformer instance.
        input_ids: (batch, seq_len) token indices.
        target_position: Sequence position to analyse (-1 = last token).

    Returns:
        {
            'n_layers': int,
            'entropy_per_layer': list[float],  # decreasing = healthy
            'top1_token_per_layer': list[int],  # how token prediction evolves
            'convergence_layer': int,  # first layer where top1 matches final prediction
        }
    """
    lens = LogitLens(model)
    layer_logits = lens.run(input_ids)

    n_layers = len(layer_logits)
    seq_len = input_ids.shape[1]

    # Resolve negative position
    position = target_position if target_position >= 0 else seq_len + target_position

    entropy_per_layer = lens.entropy_profile(position)

    # Top-1 token at each layer
    top1_token_per_layer: list[int] = []
    for layer_idx in range(n_layers):
        top_tokens = lens.top_tokens_at_layer(layer_idx, position, k=1)
        top1_token_per_layer.append(top_tokens[0][0])

    # Final prediction = top-1 at the last layer
    final_top1 = top1_token_per_layer[-1]

    # First layer where top-1 matches the final prediction
    convergence_layer = n_layers - 1  # default: only converges at last layer
    for layer_idx, token_id in enumerate(top1_token_per_layer):
        if token_id == final_top1:
            convergence_layer = layer_idx
            break

    return {
        "n_layers": n_layers,
        "entropy_per_layer": entropy_per_layer,
        "top1_token_per_layer": top1_token_per_layer,
        "convergence_layer": convergence_layer,
    }
