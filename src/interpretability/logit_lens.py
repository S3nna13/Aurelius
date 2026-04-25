"""
logit_lens.py — Logit lens visualization (Nostalgebraist 2020).

Projects the residual stream through the unembedding matrix at each layer.
Pure Python, stdlib-only. No torch dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LogitLensResult:
    """Result of projecting a single (layer, position) through the unembedding."""
    layer: int
    position: int
    top_token_ids: List[int]
    top_logits: List[float]
    entropy: float


class LogitLensAnalyzer:
    """Projects hidden states through the unembedding matrix at each layer."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def _softmax(self, logits: List[float]) -> List[float]:
        """Numerically stable softmax (subtract max before exp).

        Args:
            logits: List of raw logit values.

        Returns:
            List of probabilities summing to 1.
        """
        if not logits:
            return []
        max_val = max(logits)
        exps = [math.exp(x - max_val) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def project(
        self,
        layer: int,
        position: int,
        hidden: List[float],
        unembedding: List[List[float]],
    ) -> LogitLensResult:
        """Project a hidden state through the unembedding matrix.

        Computes logits = unembedding @ hidden (matrix-vector product).
        Selects top-5 tokens by logit value.
        Computes entropy = -sum(softmax[i] * log(softmax[i] + 1e-10)).

        Args:
            layer: Layer index.
            position: Token position index.
            hidden: 1-D list of floats (hidden_dim,).
            unembedding: 2-D list [vocab_size][hidden_dim] of floats.

        Returns:
            LogitLensResult with top-5 token ids/logits and entropy.
        """
        # Matrix-vector product: logits[v] = dot(unembedding[v], hidden)
        logits: List[float] = []
        for vocab_row in unembedding:
            dot = sum(w * h for w, h in zip(vocab_row, hidden))
            logits.append(dot)

        # Top-5 by logit value (descending)
        indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        top_k = 5
        top_token_ids = [idx for idx, _ in indexed[:top_k]]
        top_logits_vals = [val for _, val in indexed[:top_k]]

        # Entropy over full distribution
        probs = self._softmax(logits)
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)

        return LogitLensResult(
            layer=layer,
            position=position,
            top_token_ids=top_token_ids,
            top_logits=top_logits_vals,
            entropy=entropy,
        )

    def layer_trajectories(
        self,
        results: List[LogitLensResult],
        token_id: int,
    ) -> List[float]:
        """Return the logit value for a given token_id across results sorted by layer.

        Args:
            results: List of LogitLensResult objects (may span multiple layers/positions).
            token_id: Vocabulary token index to track.

        Returns:
            List of logit values, one per result sorted by layer ascending.
        """
        sorted_results = sorted(results, key=lambda r: r.layer)
        trajectory: List[float] = []
        for result in sorted_results:
            # Find the logit for token_id among top tokens, else 0.0
            if token_id in result.top_token_ids:
                idx = result.top_token_ids.index(token_id)
                trajectory.append(result.top_logits[idx])
            else:
                trajectory.append(0.0)
        return trajectory

    def entropy_by_layer(
        self,
        results: List[LogitLensResult],
    ) -> Dict[int, float]:
        """Compute mean entropy for each layer across all results at that layer.

        Args:
            results: List of LogitLensResult objects.

        Returns:
            Dict mapping layer index to mean entropy at that layer.
        """
        layer_entropies: Dict[int, List[float]] = {}
        for result in results:
            if result.layer not in layer_entropies:
                layer_entropies[result.layer] = []
            layer_entropies[result.layer].append(result.entropy)

        return {
            layer: sum(entropies) / len(entropies)
            for layer, entropies in layer_entropies.items()
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOGIT_LENS_REGISTRY = {
    "default": LogitLensAnalyzer,
}
