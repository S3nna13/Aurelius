"""Chain-of-Thought sampling utilities: CoT prompting, self-consistency voting, and answer extraction."""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import LongTensor, Tensor

# ---------------------------------------------------------------------------
# CoTConfig
# ---------------------------------------------------------------------------


@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought sampling."""

    n_samples: int = 8
    temperature: float = 0.7
    max_reasoning_tokens: int = 256
    answer_trigger: str = "The answer is"


# ---------------------------------------------------------------------------
# AnswerExtractor
# ---------------------------------------------------------------------------


class AnswerExtractor:
    """Extract answer spans from token sequences using a trigger phrase."""

    def __init__(self, trigger: str = "The answer is", vocab_size: int = 1000) -> None:
        self.trigger = trigger
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    def extract_span(
        self,
        token_ids: LongTensor,  # shape (T,)
        trigger_ids: LongTensor,  # shape (K,)
    ) -> tuple[int, int]:
        """
        Slide trigger_ids over token_ids to find the trigger, then return
        the (start, end) indices of the answer span that follows the trigger.

        Returns (-1, -1) if the trigger is not found.
        """
        T = token_ids.shape[0]
        K = trigger_ids.shape[0]

        if K == 0 or K > T:
            return (-1, -1)

        # Build a boolean mask: match[i] = True when token_ids[i:i+K] == trigger_ids
        # Use broadcasting to compare all windows at once.
        # windows: (T-K+1, K)
        indices = torch.arange(K, device=token_ids.device).unsqueeze(0)  # (1, K)
        offsets = torch.arange(T - K + 1, device=token_ids.device).unsqueeze(1)  # (T-K+1, 1)
        window_idx = offsets + indices  # (T-K+1, K)
        windows = token_ids[window_idx]  # (T-K+1, K)

        match = (windows == trigger_ids.unsqueeze(0)).all(dim=1)  # (T-K+1,)
        found = match.nonzero(as_tuple=False)

        if found.numel() == 0:
            return (-1, -1)

        # Use the first occurrence; answer starts right after the trigger.
        trigger_start = int(found[0].item())
        answer_start = trigger_start + K
        answer_end = T  # span runs to end of sequence
        return (answer_start, answer_end)

    # ------------------------------------------------------------------
    def extract_batch(
        self,
        batch_token_ids: LongTensor,  # shape (B, T)
        trigger_ids: LongTensor,  # shape (K,)
    ) -> list[tuple[int, int]]:
        """Apply extract_span to each row of a batch."""
        return [
            self.extract_span(batch_token_ids[i], trigger_ids)
            for i in range(batch_token_ids.shape[0])
        ]

    # ------------------------------------------------------------------
    def has_answer(
        self,
        token_ids: LongTensor,  # shape (T,)
        trigger_ids: LongTensor,  # shape (K,)
    ) -> bool:
        """Return True when the trigger is present in token_ids."""
        start, _ = self.extract_span(token_ids, trigger_ids)
        return start != -1


# ---------------------------------------------------------------------------
# CoTScorer
# ---------------------------------------------------------------------------


class CoTScorer:
    """Score chain-of-thought reasoning chains."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    def score_reasoning_quality(
        self,
        reasoning_ids: LongTensor,  # shape (T,)  — unused, kept for API consistency
        answer_log_probs: Tensor,  # shape (T2,)
    ) -> Tensor:
        """
        Quality score = mean of answer log-probabilities.
        Higher is better reasoning.  Returns a scalar tensor.
        """
        if answer_log_probs.numel() == 0:
            return torch.tensor(float("-inf"))
        return answer_log_probs.mean()

    # ------------------------------------------------------------------
    def length_penalty(self, length: int, target_length: int = 128) -> float:
        """
        Returns 1.0 for *ideal* length (target/2 <= length <= 2*target),
        and a value < 1.0 for too-short or too-long sequences.
        """
        lo = target_length / 2
        hi = target_length * 2

        if lo <= length <= hi:
            return 1.0

        if length < lo:
            # linearly decays from 1.0 at lo to 0.0 at 0
            return float(length / lo)

        # length > hi: linearly decays from 1.0 at hi toward 0
        # use a soft exponential decay so the penalty is < 1 for any length > hi
        excess = length - hi
        return float(1.0 / (1.0 + excess / hi))

    # ------------------------------------------------------------------
    def aggregate_scores(self, scores: Tensor) -> dict[str, float]:
        """Compute summary statistics over a 1-D score tensor."""
        return {
            "mean": float(scores.mean().item()),
            "max": float(scores.max().item()),
            "min": float(scores.min().item()),
            "std": float(scores.std().item()) if scores.numel() > 1 else 0.0,
        }


# ---------------------------------------------------------------------------
# SelfConsistencyCoT
# ---------------------------------------------------------------------------


class SelfConsistencyCoT:
    """Self-consistency voting over multiple chain-of-thought samples."""

    def __init__(self, config: CoTConfig, extractor: AnswerExtractor) -> None:
        self.config = config
        self.extractor = extractor

    # ------------------------------------------------------------------
    def vote(
        self,
        candidate_answer_spans: list[LongTensor],  # each shape (L_i,)
    ) -> tuple[LongTensor, int]:
        """
        Find the most common answer span (exact token-sequence match).
        Returns (most_common_span_tensor, count).
        """
        if not candidate_answer_spans:
            return torch.empty(0, dtype=torch.long), 0

        # Count exact matches using Python tuples as hashable keys.
        counts: dict[tuple[int, ...], int] = {}
        for span in candidate_answer_spans:
            key = tuple(span.tolist())
            counts[key] = counts.get(key, 0) + 1

        best_key = max(counts, key=lambda k: counts[k])
        best_count = counts[best_key]
        return torch.tensor(list(best_key), dtype=torch.long), best_count

    # ------------------------------------------------------------------
    def rank_by_quality(self, scores: Tensor) -> LongTensor:
        """Return indices sorted by descending score."""
        return torch.argsort(scores, descending=True)

    # ------------------------------------------------------------------
    def select_best(
        self,
        candidates: list[LongTensor],
        scores: Tensor,  # shape (N,)
    ) -> LongTensor:
        """Return the candidate with the highest score."""
        best_idx = int(scores.argmax().item())
        return candidates[best_idx]


# ---------------------------------------------------------------------------
# CoTBudgetAllocator
# ---------------------------------------------------------------------------


class CoTBudgetAllocator:
    """Allocate a fixed token budget across CoT samples."""

    def __init__(self, total_token_budget: int) -> None:
        self.total_token_budget = total_token_budget
        self._used_tokens: int = 0

    # ------------------------------------------------------------------
    def allocate(self, n_samples: int, max_reasoning_tokens: int) -> dict[str, int]:
        """
        Distribute the budget across samples.
        If n_samples * max_reasoning_tokens > budget, reduce n_samples.
        """
        if max_reasoning_tokens <= 0:
            raise ValueError("max_reasoning_tokens must be > 0")

        # Clamp n_samples so the total fits within the budget.
        adjusted_n = min(n_samples, self.total_token_budget // max_reasoning_tokens)
        adjusted_n = max(adjusted_n, 1)  # always allow at least 1 sample

        tokens_per_sample = max_reasoning_tokens
        total_tokens = adjusted_n * tokens_per_sample

        return {
            "tokens_per_sample": tokens_per_sample,
            "n_samples": adjusted_n,
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    def remaining(self, used_tokens: int) -> int:
        """Return how many tokens remain after used_tokens have been consumed."""
        return max(0, self.total_token_budget - used_tokens)
