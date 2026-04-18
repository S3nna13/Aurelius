"""STRIP backdoor/trojan detection for the Aurelius LLM research platform.

Detects trojaned inputs by superimposing them with random clean reference inputs
and measuring the entropy of the model's output distribution. A trojaned input
produces consistently low-entropy (confident) predictions even after heavy
perturbation, while a clean input's prediction distribution shifts meaningfully.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


class STRIPDetector:
    """STRIP-based backdoor/trojan detector for language models.

    Superimpose a candidate input with randomly sampled clean reference inputs
    via token-level Bernoulli mixing. A backdoored input maintains a highly
    peaked output distribution regardless of perturbation because the embedded
    trigger dominates the model's response. A clean input's output distribution
    varies substantially as the superimposed tokens alter the meaning of the
    sequence.

    Args:
        model: The language model under evaluation.
        n_perturbations: Number of random perturbation rounds per candidate.
        entropy_threshold: Inputs whose mean entropy falls below this value are
            flagged as trojaned.
    """

    def __init__(
        self,
        model: AureliusTransformer,
        n_perturbations: int = 64,
        entropy_threshold: float = 1.0,
    ) -> None:
        self.model = model
        self.n_perturbations = n_perturbations
        self.entropy_threshold = entropy_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prediction_entropy(self, logits: torch.Tensor) -> float:
        """Compute Shannon entropy of the softmax distribution over the last token.

        Args:
            logits: Tensor of shape (batch, seq_len, vocab_size) or
                (seq_len, vocab_size) or (vocab_size,).

        Returns:
            Scalar entropy value in nats.
        """
        # Flatten to the last-token logit vector
        if logits.dim() == 3:
            last_logits = logits[0, -1, :]   # (vocab_size,)
        elif logits.dim() == 2:
            last_logits = logits[-1, :]       # (vocab_size,)
        else:
            last_logits = logits              # (vocab_size,)

        probs = F.softmax(last_logits, dim=-1)
        # H = -sum(p * log(p)); clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-12))
        entropy = -(probs * log_probs).sum()
        return entropy.item()

    def _superimpose(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor,
    ) -> torch.Tensor:
        """Token-level superimposition via Bernoulli mixing.

        For each position, independently draw a Bernoulli(0.5) mask. Where the
        mask is 1, keep the token from input_ids_a; where it is 0, take the
        token from input_ids_b. Both inputs must have the same shape.

        Args:
            input_ids_a: Integer token tensor of shape (..., seq_len).
            input_ids_b: Integer token tensor of the same shape.

        Returns:
            Mixed token tensor of the same shape as the inputs.
        """
        mask = torch.bernoulli(torch.full(input_ids_a.shape, 0.5)).bool()
        return torch.where(mask, input_ids_a, input_ids_b)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        input_ids: torch.Tensor,
        reference_pool: torch.Tensor,
    ) -> float:
        """Compute the mean prediction entropy across all perturbation rounds.

        Each round: sample a random reference from reference_pool,
        superimpose it with input_ids, run the model, and record the
        entropy of the last-token output distribution. The mean entropy across
        all rounds is the detection score — lower means more trojan-like.

        Args:
            input_ids: (1, seq_len) or (seq_len,) candidate input tokens.
            reference_pool: (N, seq_len) pool of clean reference sequences.

        Returns:
            Mean entropy (float) across n_perturbations rounds.
        """
        # Normalise candidate to shape (1, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        seq_len = input_ids.shape[-1]
        n_refs = reference_pool.shape[0]

        entropies: list[float] = []
        self.model.training  # access attribute to avoid unused-import warning
        self.model.eval()

        for _ in range(self.n_perturbations):
            ref_idx = int(torch.randint(0, n_refs, (1,)).item())
            ref = reference_pool[ref_idx : ref_idx + 1]  # (1, ref_seq_len)

            # Align lengths: truncate or pad reference to match candidate
            ref_len = ref.shape[-1]
            if ref_len > seq_len:
                ref = ref[:, :seq_len]
            elif ref_len < seq_len:
                pad = torch.zeros(1, seq_len - ref_len, dtype=ref.dtype)
                ref = torch.cat([ref, pad], dim=-1)

            mixed = self._superimpose(input_ids, ref)
            _loss, logits, _pkv = self.model(mixed)
            entropies.append(self._prediction_entropy(logits))

        return float(sum(entropies) / len(entropies))

    def is_trojan(
        self,
        input_ids: torch.Tensor,
        reference_pool: torch.Tensor,
    ) -> bool:
        """Return True if the candidate input is flagged as trojaned.

        A candidate is flagged when its mean perturbation entropy falls below
        self.entropy_threshold.

        Args:
            input_ids: Candidate token sequence.
            reference_pool: Pool of clean reference sequences.

        Returns:
            True if the input is suspected to contain a backdoor trigger.
        """
        return self.score(input_ids, reference_pool) < self.entropy_threshold

    @torch.no_grad()
    def batch_score(
        self,
        batch_ids: torch.Tensor,
        reference_pool: torch.Tensor,
    ) -> List[float]:
        """Score each sample in a batch independently.

        Args:
            batch_ids: (B, seq_len) batch of candidate token sequences.
            reference_pool: (N, seq_len) pool of clean reference sequences.

        Returns:
            List of per-sample mean entropy scores, length B.
        """
        scores: list[float] = []
        for i in range(batch_ids.shape[0]):
            scores.append(self.score(batch_ids[i : i + 1], reference_pool))
        return scores
