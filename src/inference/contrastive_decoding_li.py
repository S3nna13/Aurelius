"""Contrastive Decoding (Li et al., arXiv:2210.15097).

Implements CD scoring and autoregressive generation using an expert model and
an amateur model.  The core idea: subtract the amateur's log-probabilities from
the expert's, but only for tokens in the *plausibility set* — tokens whose
expert probability is at least α-fraction of the maximum expert probability.

Reference:
    Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner,
    Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis.
    "Contrastive Decoding: Open-ended Text Generation as Optimization."
    arXiv:2210.15097, 2022.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


class CDScorer:
    """Computes Contrastive Decoding scores for a single token step.

    The CD score for a vocabulary token v is::

        CD(v) = log p_expert(v) - log p_amateur(v)   if v ∈ V_head
                -inf                                  otherwise

    where the plausibility set is::

        V_head = {v : p_expert(v) >= alpha * max_v p_expert(v)}

    Args:
        alpha: Plausibility threshold in [0, 1].  Tokens whose expert
            probability falls below ``alpha * max_expert_prob`` receive a
            score of ``-inf``.  Default: 0.1.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def score(
        self,
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CD scores for a single position.

        Args:
            expert_logits: Raw logits from the expert model, shape ``(V,)``.
            amateur_logits: Raw logits from the amateur model, shape ``(V,)``.

        Returns:
            CD scores of shape ``(V,)``.  Tokens outside the plausibility set
            have score ``-inf``; tokens inside have
            ``log_softmax(expert)[v] - log_softmax(amateur)[v]``.
        """
        # Compute probabilities for plausibility check
        expert_probs = F.softmax(expert_logits, dim=-1)  # (V,)
        max_prob = expert_probs.max()
        plausible = expert_probs >= self.alpha * max_prob  # (V,) bool

        # Compute log probabilities for the CD score
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)  # (V,)
        amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)  # (V,)

        cd_scores = expert_log_probs - amateur_log_probs  # (V,)
        cd_scores = cd_scores.masked_fill(~plausible, float("-inf"))

        return cd_scores


class CDSearcher:
    """Batched Contrastive Decoding token selection via argmax.

    Args:
        alpha: Plausibility threshold passed to :class:`CDScorer`.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._scorer = CDScorer(alpha=alpha)

    def search(
        self,
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Select one token per batch position using CD argmax.

        Args:
            expert_logits: Raw expert logits, shape ``(B, V)``.
            amateur_logits: Raw amateur logits, shape ``(B, V)``.

        Returns:
            LongTensor of selected token ids, shape ``(B,)``.
        """
        B, V = expert_logits.shape

        # Compute probabilities for plausibility check (batched)
        expert_probs = F.softmax(expert_logits, dim=-1)  # (B, V)
        max_prob = expert_probs.max(dim=-1, keepdim=True).values  # (B, 1)
        plausible = expert_probs >= self.alpha * max_prob  # (B, V)

        # CD log-prob difference
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)  # (B, V)
        amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)  # (B, V)

        cd_scores = expert_log_probs - amateur_log_probs  # (B, V)
        cd_scores = cd_scores.masked_fill(~plausible, float("-inf"))

        return cd_scores.argmax(dim=-1)  # (B,)


class ContrastiveDecoder:
    """Autoregressive text generation using Contrastive Decoding.

    At every generation step the decoder:
    1. Calls both ``expert_fn`` and ``amateur_fn`` with the current sequence.
    2. Extracts the last-position logits from each.
    3. Computes CD scores (plausibility-filtered log-prob differences).
    4. Picks the token with the highest CD score (argmax).
    5. Appends the token and repeats.

    Args:
        expert_fn: Callable ``(input_ids: LongTensor[1, T]) -> logits (1, T, V)``.
        amateur_fn: Callable ``(input_ids: LongTensor[1, T]) -> logits (1, T, V)``.
        alpha: Plausibility threshold.  Default: 0.1.
    """

    def __init__(
        self,
        expert_fn: Callable[[torch.Tensor], torch.Tensor],
        amateur_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float = 0.1,
    ) -> None:
        self.expert_fn = expert_fn
        self.amateur_fn = amateur_fn
        self._searcher = CDSearcher(alpha=alpha)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            prompt_ids: Prompt token ids, shape ``(1, T)``.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            LongTensor of *new* token ids only, shape ``(max_new_tokens,)``.
        """
        seq = prompt_ids.clone()
        new_tokens: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            expert_logits_seq = self.expert_fn(seq)  # (1, T, V)
            amateur_logits_seq = self.amateur_fn(seq)  # (1, T, V)

            # Last-position logits: (1, V)
            e_logits = expert_logits_seq[:, -1, :]
            a_logits = amateur_logits_seq[:, -1, :]

            # selected: (1,) LongTensor
            selected = self._searcher.search(e_logits, a_logits)  # (1,)

            new_tokens.append(selected)
            seq = torch.cat([seq, selected.unsqueeze(1)], dim=1)  # (1, T+1)

        # Stack and squeeze batch dim → (max_new_tokens,)
        return torch.stack(new_tokens, dim=0).squeeze(1)
