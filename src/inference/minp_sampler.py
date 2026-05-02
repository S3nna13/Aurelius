"""Min-P adaptive sampler for AureliusTransformer.

Implements Min-P Sampling (Nguyen et al. 2024): the probability floor scales
dynamically with the top-token probability, making the filter more restrictive
when the model is confident and more permissive when the distribution is flat.

    p_min = min_p * max(p_tokens)

This avoids the fixed-cutoff brittleness of top-p (nucleus) sampling while
still providing a sensible vocabulary pruning step.

Can be composed with temperature scaling, top-k, and top-p in that order:
    temperature → top_k → min_p → top_p → categorical
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MinPConfig:
    """Hyper-parameters for the Min-P sampler.

    Attributes
    ----------
    min_p:
        Fraction of the top-token probability used as the probability floor.
        Tokens with prob < min_p * p_max are masked out.
    temperature:
        Logit scaling factor applied before all filtering steps.
        Values > 1 flatten the distribution; < 1 sharpen it.
        Exactly 1.0 is a no-op (handled without division).
    top_k:
        If > 0, retain only the top-k logits before min_p filtering.
    top_p:
        If < 1.0, apply nucleus (top-p) filtering *after* min_p.
    """

    min_p: float = 0.05
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


class MinPSampler:
    """Adaptive probability-floor sampler.

    Parameters
    ----------
    config:
        :class:`MinPConfig` instance controlling all hyper-parameters.
    """

    def __init__(self, config: MinPConfig | None = None) -> None:
        self.config = config if config is not None else MinPConfig()

    # ------------------------------------------------------------------
    # Filtering primitives
    # ------------------------------------------------------------------

    def apply_temperature(self, logits: Tensor) -> Tensor:
        """Divide logits by temperature.

        Parameters
        ----------
        logits:
            Shape ``(V,)`` or ``(B, V)``.

        Returns
        -------
        Tensor
            Same shape; unchanged when ``temperature == 1.0``.
        """
        t = self.config.temperature
        if t == 1.0:
            return logits
        return logits / t

    def apply_top_k(self, logits: Tensor) -> Tensor:
        """Zero out (set to ``-inf``) all but the top-k logits.

        Parameters
        ----------
        logits:
            Shape ``(V,)`` or ``(B, V)``.

        Returns
        -------
        Tensor
            Same shape; unchanged when ``top_k == 0``.
        """
        k = self.config.top_k
        if k == 0:
            return logits

        # Work in 2-D for consistent indexing.
        squeezed = logits.ndim == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        # topk values: (B, k)
        top_values, _ = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
        # Threshold is the smallest kept value per row.
        kth_value = top_values[..., -1].unsqueeze(-1)  # (B, 1)
        filtered = logits.clone()
        filtered[logits < kth_value] = float("-inf")

        return filtered.squeeze(0) if squeezed else filtered

    def apply_min_p(self, logits: Tensor) -> Tensor:
        """Apply min-p filtering.

        Computes ``threshold = min_p * p_max`` from the probability
        distribution implied by *logits*, then masks every token whose
        probability falls below that threshold.

        Parameters
        ----------
        logits:
            Shape ``(V,)`` or ``(B, V)``.

        Returns
        -------
        Tensor
            Same shape with sub-threshold positions set to ``-inf``.
        """
        squeezed = logits.ndim == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)  # (B, V)
        p_max = probs.max(dim=-1, keepdim=True).values  # (B, 1)
        threshold = self.config.min_p * p_max  # (B, 1)

        filtered = logits.clone()
        filtered[probs < threshold] = float("-inf")

        return filtered.squeeze(0) if squeezed else filtered

    def apply_top_p(self, logits: Tensor) -> Tensor:
        """Apply nucleus (top-p) filtering.

        Sorts probabilities descending, accumulates, and masks all tokens
        beyond the nucleus (cumulative probability > top_p).

        Parameters
        ----------
        logits:
            Shape ``(V,)`` or ``(B, V)``.

        Returns
        -------
        Tensor
            Same shape; unchanged when ``top_p == 1.0``.
        """
        p = self.config.top_p
        if p >= 1.0:
            return logits

        squeezed = logits.ndim == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)  # (B, V)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)  # (B, V)

        # Remove tokens whose cumulative probability exceeds top_p.
        # Shift by one so the token that crosses the boundary is kept.
        remove_mask = cumprobs - sorted_probs > p  # (B, V)

        # Scatter mask back to original vocab ordering.
        mask = torch.zeros_like(remove_mask).scatter_(-1, sorted_idx, remove_mask)

        filtered = logits.clone()
        filtered[mask] = float("-inf")

        return filtered.squeeze(0) if squeezed else filtered

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, logits: Tensor) -> Tensor:
        """Sample a token from the filtered distribution.

        Pipeline: temperature → top_k → min_p → top_p → multinomial.

        Parameters
        ----------
        logits:
            Shape ``(V,)`` for a single example, or ``(B, V)`` for a batch.

        Returns
        -------
        Tensor
            Shape ``()`` (scalar) for unbatched input, or ``(B,)`` for
            batched input.  Values are integer token ids in ``[0, V)``.
        """
        squeezed = logits.ndim == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        logits = self.apply_temperature(logits)
        logits = self.apply_top_k(logits)
        logits = self.apply_min_p(logits)
        logits = self.apply_top_p(logits)

        probs = F.softmax(logits, dim=-1)  # (B, V)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

        return token_ids.squeeze(0) if squeezed else token_ids

    def sample_with_probs(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Sample a token and return the post-filter probability distribution.

        Parameters
        ----------
        logits:
            Shape ``(V,)`` or ``(B, V)``.

        Returns
        -------
        token_ids:
            Shape ``()`` or ``(B,)``.
        filtered_probs:
            Softmax over the filtered logits — shape ``(V,)`` or ``(B, V)``.
            Each row sums to 1.
        """
        squeezed = logits.ndim == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        logits = self.apply_temperature(logits)
        logits = self.apply_top_k(logits)
        logits = self.apply_min_p(logits)
        logits = self.apply_top_p(logits)

        filtered_probs = F.softmax(logits, dim=-1)  # (B, V)
        token_ids = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)  # (B,)

        if squeezed:
            return token_ids.squeeze(0), filtered_probs.squeeze(0)
        return token_ids, filtered_probs

    def effective_vocab_size(self, logits: Tensor) -> Tensor:
        """Count tokens that survive the min-p filter.

        Only the min-p filter is applied (temperature and top-k are applied
        first as they would be in a real forward pass, but top-p is excluded
        because it operates on cumulative probability).

        Parameters
        ----------
        logits:
            Shape ``(V,)`` or ``(B, V)``.

        Returns
        -------
        Tensor
            Scalar (unbatched) or shape ``(B,)`` (batched) integer counts.
        """
        squeezed = logits.ndim == 1
        if squeezed:
            logits = logits.unsqueeze(0)

        logits = self.apply_temperature(logits)
        logits = self.apply_top_k(logits)
        filtered = self.apply_min_p(logits)

        # Count finite (non-masked) logit positions.
        surviving = (filtered > float("-inf")).sum(dim=-1)  # (B,)

        return surviving.squeeze(0) if squeezed else surviving


__all__ = [
    "MinPConfig",
    "MinPSampler",
]
