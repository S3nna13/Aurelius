"""Statistical prompt injection detector for the Aurelius LLM research platform.

Detects instruction-hijacking attempts using token-level n-gram overlap analysis
and perplexity divergence between a system context and a user message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.model.transformer import AureliusTransformer


@dataclass
class InjectionPattern:
    """A known injection pattern represented as a token sequence with a detection weight.

    Attributes:
        pattern_ids: Token id sequence characteristic of an injection attempt.
        weight: Relative importance of this pattern when computing the aggregate score.
        label: Human-readable name for this pattern.
    """

    pattern_ids: list[int]
    weight: float
    label: str


# Default synthetic patterns that mimic short override commands.
# These are fixed token sequences used when no custom patterns are supplied.
_DEFAULT_PATTERNS: list[InjectionPattern] = [
    InjectionPattern(
        pattern_ids=[10, 11, 12, 13, 14],
        weight=1.0,
        label="ignore_previous_a",
    ),
    InjectionPattern(
        pattern_ids=[20, 21, 22, 23, 24],
        weight=1.0,
        label="new_instruction_a",
    ),
    InjectionPattern(
        pattern_ids=[30, 31, 32, 33, 34],
        weight=1.2,
        label="override_system_a",
    ),
    InjectionPattern(
        pattern_ids=[40, 41, 42, 43],
        weight=0.8,
        label="disregard_context_a",
    ),
    InjectionPattern(
        pattern_ids=[50, 51, 52, 53, 54, 55],
        weight=1.0,
        label="act_as_a",
    ),
]


class PromptInjectionDetector:
    """Statistical detector for prompt injection attacks.

    Combines three signals to flag potential instruction-hijacking:

    1. Pattern score: weighted n-gram Jaccard overlap between the user
       message and a library of known injection token patterns.
    2. Perplexity ratio: ratio of the model perplexity on the user message
       versus the system context. A large ratio indicates that the user
       content is statistically surprising relative to the system distribution.
    3. Overlap: raw n-gram overlap between the system context and the user
       message as a secondary signal.

    Args:
        model: The language model used to compute perplexity signals.
        patterns: Optional list of injection patterns. Uses a built-in
            synthetic default set when None.
    """

    def __init__(
        self,
        model: AureliusTransformer,
        patterns: list[InjectionPattern] | None = None,
    ) -> None:
        self.model = model
        self.patterns: list[InjectionPattern] = (
            patterns if patterns is not None else list(_DEFAULT_PATTERNS)
        )

    # ------------------------------------------------------------------
    # Core statistical methods
    # ------------------------------------------------------------------

    @staticmethod
    def ngram_overlap(
        ids_a: list[int],
        ids_b: list[int],
        n: int = 3,
    ) -> float:
        """Jaccard similarity of n-gram sets between two token sequences.

        Args:
            ids_a: First token id sequence.
            ids_b: Second token id sequence.
            n: n-gram order.

        Returns:
            Float in [0, 1]. Returns 0.0 when both sequences are too short
            to form any n-gram.
        """

        def _ngrams(ids: list[int], order: int):
            return set(tuple(ids[i : i + order]) for i in range(len(ids) - order + 1))

        grams_a = _ngrams(ids_a, n)
        grams_b = _ngrams(ids_b, n)

        union = grams_a | grams_b
        if not union:
            return 0.0
        intersection = grams_a & grams_b
        return len(intersection) / len(union)

    @torch.no_grad()
    def _compute_perplexity(self, ids: list[int]) -> float:
        """Compute perplexity of a token sequence under the model.

        Uses a single forward pass and computes per-token NLL from the
        shifted logits. Sequences of length < 2 return a large finite value.

        Args:
            ids: Token id list.

        Returns:
            Perplexity (float >= 1.0).
        """
        if len(ids) < 2:
            return float(math.exp(20))

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        # Truncate to model max sequence length
        max_len = self.model.config.max_seq_len
        if input_tensor.shape[1] > max_len:
            input_tensor = input_tensor[:, :max_len]

        self.model.eval()
        _, logits, _ = self.model(input_tensor)

        # NLL: logits[:, :-1] predicts ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_tensor[:, 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, logits.shape[-1]),
            shift_labels.view(-1),
            reduction="mean",
        )

        # Cap exponent to avoid overflow
        ppl = math.exp(min(nll.item(), 20.0))
        return float(ppl)

    def perplexity_ratio(
        self,
        system_ids: list[int],
        user_ids: list[int],
    ) -> float:
        """Ratio of model perplexity on user_ids versus system_ids.

        A high ratio indicates that the user content is statistically surprising
        relative to the system context, which may signal an injected override command.

        Args:
            system_ids: Token ids of the system context.
            user_ids: Token ids of the user message.

        Returns:
            Positive float. Division by zero is guarded with a 1e-8 floor.
        """
        ppl_system = self._compute_perplexity(system_ids)
        ppl_user = self._compute_perplexity(user_ids)
        return ppl_user / (ppl_system + 1e-8)

    def pattern_score(self, input_ids: list[int]) -> float:
        """Weighted sum of n-gram overlaps between input_ids and each known pattern.

        Args:
            input_ids: Token id sequence to evaluate.

        Returns:
            Non-negative float. Higher values indicate more overlap with
            known injection patterns.
        """
        total = 0.0
        for pattern in self.patterns:
            overlap = self.ngram_overlap(input_ids, pattern.pattern_ids, n=3)
            total += pattern.weight * overlap
        return total

    # ------------------------------------------------------------------
    # Detection API
    # ------------------------------------------------------------------

    def detect(
        self,
        system_ids: list[int],
        user_ids: list[int],
        overlap_threshold: float = 0.1,
        perplexity_ratio_threshold: float = 5.0,
    ) -> dict:
        """Detect whether user_ids constitutes a prompt injection attempt.

        An injection is flagged when either:
        - The pattern score exceeds overlap_threshold, or
        - The perplexity ratio exceeds perplexity_ratio_threshold.

        Args:
            system_ids: Token ids of the trusted system context.
            user_ids: Token ids of the potentially untrusted user message.
            overlap_threshold: Minimum pattern_score to trigger detection.
            perplexity_ratio_threshold: Minimum perplexity ratio to trigger detection.

        Returns:
            Dict with keys:
                'is_injection' (bool): True when injection is suspected.
                'pattern_score' (float): Weighted pattern overlap score.
                'perplexity_ratio' (float): PPL ratio (user / system).
                'overlap' (float): n-gram Jaccard overlap between system and user.
        """
        p_score = self.pattern_score(user_ids)
        ppl_ratio = self.perplexity_ratio(system_ids, user_ids)
        overlap = self.ngram_overlap(system_ids, user_ids, n=3)

        is_injection = bool(p_score > overlap_threshold or ppl_ratio > perplexity_ratio_threshold)

        return {
            "is_injection": is_injection,
            "pattern_score": float(p_score),
            "perplexity_ratio": float(ppl_ratio),
            "overlap": float(overlap),
        }

    def batch_detect(
        self,
        system_ids: list[int],
        list_of_user_ids: list[list[int]],
        overlap_threshold: float = 0.1,
        perplexity_ratio_threshold: float = 5.0,
    ) -> list[dict]:
        """Run detect() for a list of user messages against a fixed system context.

        Args:
            system_ids: Token ids of the trusted system context.
            list_of_user_ids: Each element is a list of token ids representing
                one user message.
            overlap_threshold: Forwarded to detect().
            perplexity_ratio_threshold: Forwarded to detect().

        Returns:
            List of detect() result dicts, one per user message.
        """
        return [
            self.detect(
                system_ids,
                user_ids,
                overlap_threshold=overlap_threshold,
                perplexity_ratio_threshold=perplexity_ratio_threshold,
            )
            for user_ids in list_of_user_ids
        ]
