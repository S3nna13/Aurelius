"""Majority Voting / Self-Consistency decoding (Wang et al., 2022).

Self-Consistency improves reasoning by sampling multiple diverse chain-of-thought
responses and taking the majority vote on the final answer. Instead of greedy
decoding, sample k responses with temperature, extract the final answer from each,
and return the most common answer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency / majority-voting decoding."""

    n_samples: int = 8
    temperature: float = 0.7
    max_new_tokens: int = 32
    aggregation: str = "plurality"  # 'plurality' | 'weighted'


# ---------------------------------------------------------------------------
# AnswerExtractor
# ---------------------------------------------------------------------------

class AnswerExtractor:
    """Extracts the final answer from a generated token sequence.

    Args:
        extraction_pattern: Optional regex pattern applied to token ids
            interpreted as ASCII chars. Defaults to None (last non-padding
            token id used as answer).
    """

    PAD_TOKEN_ID: int = 0

    def __init__(self, extraction_pattern: Optional[str] = None) -> None:
        self.extraction_pattern = extraction_pattern

    def extract(self, response_ids: torch.Tensor, vocab_size: int = 256) -> Optional[int]:
        """Extract answer from a single response tensor.

        Args:
            response_ids: 1-D tensor of token ids, shape (seq_len,).
            vocab_size: Vocabulary size (unused in default extraction but
                        reserved for logit-based extensions).

        Returns:
            The last non-padding token id as an int, or None if all padding.
        """
        if response_ids.numel() == 0:
            return None

        # Find the last non-padding token
        non_pad_mask = response_ids != self.PAD_TOKEN_ID
        if not non_pad_mask.any():
            return None

        # Use extraction pattern if provided
        if self.extraction_pattern is not None:
            text = "".join(
                chr(max(0, min(127, int(t.item())))) for t in response_ids if t.item() != self.PAD_TOKEN_ID
            )
            match = re.search(self.extraction_pattern, text)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
            # Fall through to default

        # Default: return the last non-padding token id
        indices = non_pad_mask.nonzero(as_tuple=False).squeeze(-1)
        last_idx = int(indices[-1].item())
        return int(response_ids[last_idx].item())

    def extract_from_candidates(self, candidate_ids: torch.Tensor) -> List[Optional[int]]:
        """Extract answers from a batch of candidate responses.

        Args:
            candidate_ids: Token id tensor of shape (n_samples, seq_len).

        Returns:
            List of length n_samples with extracted answers (int or None).
        """
        n_samples = candidate_ids.shape[0]
        return [self.extract(candidate_ids[i]) for i in range(n_samples)]


# ---------------------------------------------------------------------------
# MajorityVoter
# ---------------------------------------------------------------------------

class MajorityVoter:
    """Aggregates multiple candidate answers via majority (plurality) or weighted vote.

    Args:
        n_samples: Expected number of samples (informational).
        temperature: Sampling temperature used upstream (informational).
        aggregation: 'plurality' for most-common, 'weighted' for confidence-weighted.
    """

    def __init__(
        self,
        n_samples: int = 8,
        temperature: float = 0.7,
        aggregation: str = "plurality",
    ) -> None:
        self.n_samples = n_samples
        self.temperature = temperature
        self.aggregation = aggregation

    def vote(
        self, answers: List[Optional[int]]
    ) -> Tuple[Optional[int], Dict]:
        """Plurality vote over a list of answers.

        None values are excluded from voting but counted in n_valid.

        Args:
            answers: List of extracted answer ints (may contain None).

        Returns:
            Tuple of (winning_answer, stats) where stats contains:
                - 'vote_counts': dict mapping answer -> count
                - 'confidence': fraction of valid answers that match winner
                - 'n_valid': number of non-None answers
        """
        valid = [a for a in answers if a is not None]
        n_valid = len(valid)

        if n_valid == 0:
            return (None, {"vote_counts": {}, "confidence": 0.0, "n_valid": 0})

        counts = Counter(valid)
        winner = counts.most_common(1)[0][0]
        winner_count = counts[winner]
        confidence = winner_count / n_valid

        return (
            winner,
            {
                "vote_counts": dict(counts),
                "confidence": confidence,
                "n_valid": n_valid,
            },
        )

    def weighted_vote(
        self, answers: List[Optional[int]], weights: List[float]
    ) -> Optional[int]:
        """Weight votes by confidence scores.

        Args:
            answers: List of extracted answers (may contain None).
            weights: Per-answer confidence weights.

        Returns:
            Winning answer according to weighted votes, or None if no valid answers.
        """
        if len(answers) != len(weights):
            raise ValueError("answers and weights must have the same length")

        weighted_counts: Dict[int, float] = {}
        for ans, w in zip(answers, weights):
            if ans is None:
                continue
            weighted_counts[ans] = weighted_counts.get(ans, 0.0) + w

        if not weighted_counts:
            return None

        return max(weighted_counts, key=lambda k: weighted_counts[k])


# ---------------------------------------------------------------------------
# SelfConsistencyDecoder
# ---------------------------------------------------------------------------

class SelfConsistencyDecoder:
    """Self-consistency decoder: sample → extract → vote.

    Combines a language model, a MajorityVoter, and an AnswerExtractor to
    implement Wang et al. 2022 self-consistency for improved reasoning.

    Args:
        model: Any nn.Module whose forward() returns logits of shape
               (batch, seq_len, vocab_size) as second element (following
               AureliusTransformer convention).
        voter: MajorityVoter instance.
        extractor: AnswerExtractor instance.
    """

    def __init__(
        self,
        model: nn.Module,
        voter: MajorityVoter,
        extractor: AnswerExtractor,
    ) -> None:
        self.model = model
        self.voter = voter
        self.extractor = extractor

    @torch.no_grad()
    def sample_responses(
        self,
        input_ids: torch.Tensor,
        n_samples: int,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """Sample n_samples independent responses from the model.

        Args:
            input_ids: Prompt token ids, shape (1, prompt_len) or (prompt_len,).
            n_samples: Number of responses to sample.
            max_new_tokens: Maximum tokens to generate per sample.
            temperature: Sampling temperature (>0 for stochastic, 0 for greedy).

        Returns:
            Generated token id tensor of shape (n_samples, max_new_tokens).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        all_samples: List[torch.Tensor] = []

        for _ in range(n_samples):
            generated: List[int] = []
            cur_ids = input_ids.clone()

            for _ in range(max_new_tokens):
                output = self.model(cur_ids)
                # Support models returning (hidden, logits, ...) or just logits
                if isinstance(output, tuple):
                    logits = output[1]
                else:
                    logits = output

                next_logits = logits[:, -1, :]  # (1, vocab_size)

                if temperature == 0.0:
                    next_token = int(next_logits.argmax(dim=-1).item())
                else:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())

                generated.append(next_token)
                cur_ids = torch.tensor([[next_token]], dtype=torch.long)

            # Pad or truncate to max_new_tokens
            gen_tensor = torch.tensor(generated[:max_new_tokens], dtype=torch.long)
            if gen_tensor.shape[0] < max_new_tokens:
                padding = torch.zeros(max_new_tokens - gen_tensor.shape[0], dtype=torch.long)
                gen_tensor = torch.cat([gen_tensor, padding])
            all_samples.append(gen_tensor)

        return torch.stack(all_samples, dim=0)  # (n_samples, max_new_tokens)

    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
    ) -> Tuple[Optional[int], Dict]:
        """Full self-consistency pipeline: sample → extract → vote.

        Args:
            input_ids: Prompt token ids.
            max_new_tokens: Max tokens per sample.

        Returns:
            Tuple of (winning_answer, stats) where stats contains vote_counts,
            confidence, n_valid, and n_samples.
        """
        n_samples = self.voter.n_samples
        temperature = self.voter.temperature

        # Sample responses
        candidate_ids = self.sample_responses(
            input_ids, n_samples=n_samples,
            max_new_tokens=max_new_tokens, temperature=temperature
        )

        # Extract answers
        answers = self.extractor.extract_from_candidates(candidate_ids)

        # Aggregate
        if self.voter.aggregation == "weighted":
            weights = [1.0] * len(answers)
            winner = self.voter.weighted_vote(answers, weights)
            _, stats = self.voter.vote(answers)
        else:
            winner, stats = self.voter.vote(answers)

        stats["n_samples"] = n_samples
        return (winner, stats)

    def get_confidence(self, answers: List[Optional[int]]) -> float:
        """Compute fraction of samples agreeing with the plurality winner.

        Args:
            answers: List of extracted answers.

        Returns:
            Confidence in [0, 1]. 0.0 if no valid answers.
        """
        _, stats = self.voter.vote(answers)
        return float(stats["confidence"])
