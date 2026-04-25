"""Gradient-based attribution: integrated gradients, saliency maps."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from torch import Tensor


class AttributionMethod(str, Enum):
    SALIENCY = "saliency"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_X_INPUT = "gradient_x_input"


@dataclass
class Attribution:
    token_ids: List[int]
    scores: List[float]
    method: AttributionMethod
    normalized: bool = False


class GradientAttribution:
    """Compute gradient-based attribution scores."""

    def __init__(
        self,
        method: AttributionMethod = AttributionMethod.SALIENCY,
        n_steps: int = 50,
    ):
        self.method = method
        self.n_steps = n_steps

    def normalize(self, scores: List[float]) -> List[float]:
        """L2-normalize a list of scores.

        Returns zeros if the input is all-zero (avoids NaN).
        """
        sum_sq = sum(s * s for s in scores)
        norm = math.sqrt(sum_sq + 1e-8)
        return [s / norm for s in scores]

    def saliency(self, embeddings: Tensor, gradient: Tensor) -> List[float]:
        """Compute saliency scores: |gradient|.sum(-1).

        Args:
            embeddings: (seq_len, d_model) tensor (unused in saliency but kept for API symmetry).
            gradient: (seq_len, d_model) gradient tensor.

        Returns:
            list of floats, one per token.
        """
        return gradient.abs().sum(-1).tolist()

    def gradient_x_input(self, embeddings: Tensor, gradient: Tensor) -> List[float]:
        """Compute gradient * input scores.

        Args:
            embeddings: (seq_len, d_model) input embeddings.
            gradient: (seq_len, d_model) gradient tensor.

        Returns:
            list of floats, one per token.
        """
        return (gradient * embeddings).abs().sum(-1).tolist()

    def integrated_gradients_scores(
        self,
        baseline_grads: List[List[float]],
        input_grads: List[float],
    ) -> List[float]:
        """Approximate integrated gradients.

        Args:
            baseline_grads: list of per-step gradient lists (n_steps x seq_len).
            input_grads: final gradient at the input (seq_len,).

        Returns:
            list of floats (seq_len,).
        """
        if not baseline_grads:
            return [0.0 * g for g in input_grads]

        n_steps = len(baseline_grads)
        seq_len = len(input_grads)
        # Mean over steps element-wise
        mean_grads = [
            sum(baseline_grads[s][i] for s in range(n_steps)) / n_steps
            for i in range(seq_len)
        ]
        # Multiply by input grads element-wise
        return [m * g for m, g in zip(mean_grads, input_grads)]

    def create_attribution(
        self,
        token_ids: List[int],
        scores: List[float],
        normalize: bool = True,
    ) -> Attribution:
        """Create an Attribution object.

        Args:
            token_ids: list of token ids.
            scores: raw attribution scores.
            normalize: if True, L2-normalize the scores.

        Returns:
            Attribution dataclass instance.
        """
        if normalize:
            final_scores = self.normalize(scores)
        else:
            final_scores = scores
        return Attribution(
            token_ids=token_ids,
            scores=final_scores,
            method=self.method,
            normalized=normalize,
        )
