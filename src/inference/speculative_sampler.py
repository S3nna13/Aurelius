"""Speculative decoding sampler: draft-verify cycle, acceptance criterion."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SpeculativeConfig:
    n_draft: int = 5
    temperature: float = 1.0
    top_p: float = 1.0
    min_acceptance_rate: float = 0.5


@dataclass
class DraftToken:
    token_id: int
    logprob: float
    position: int


@dataclass
class VerificationResult:
    accepted_tokens: list[int]
    rejection_point: int | None
    acceptance_rate: float


class SpeculativeSampler:
    """Draft-verify speculative decoding sampler."""

    def __init__(self, config: SpeculativeConfig | None = None) -> None:
        self.config = config if config is not None else SpeculativeConfig()

    def sample_draft(self, logits_sequence: list[list[float]]) -> list[DraftToken]:
        """For each position's logits: apply temperature, take argmax.

        Returns a list of DraftToken, one per position.
        """
        draft_tokens: list[DraftToken] = []
        for position, logits in enumerate(logits_sequence):
            # Apply temperature scaling
            temp = self.config.temperature
            if temp <= 0.0:
                temp = 1e-8
            scaled = [line / temp for line in logits]

            # Argmax selection (deterministic)
            token_id = max(range(len(scaled)), key=lambda i: scaled[i])

            # Compute log-softmax for the selected token to get logprob
            max_val = scaled[token_id]
            log_sum_exp = max_val + math.log(sum(math.exp(s - max_val) for s in scaled))
            logprob = scaled[token_id] - log_sum_exp

            draft_tokens.append(DraftToken(token_id=token_id, logprob=logprob, position=position))
        return draft_tokens

    def verify(
        self,
        draft_tokens: list[DraftToken],
        target_logprobs: list[float],
    ) -> VerificationResult:
        """Verify draft tokens against target model log-probs.

        Acceptance probability = min(1, exp(target_logprob - draft_logprob)).
        Uses a fixed threshold of 0.5 for determinism.
        """
        if not draft_tokens:
            return VerificationResult(
                accepted_tokens=[],
                rejection_point=None,
                acceptance_rate=0.0,
            )

        accepted_tokens: list[int] = []
        rejection_point: int | None = None
        fixed_threshold = 0.5

        for i, draft_token in enumerate(draft_tokens):
            target_lp = target_logprobs[i] if i < len(target_logprobs) else float("-inf")
            acceptance_prob = min(1.0, math.exp(target_lp - draft_token.logprob))

            if acceptance_prob >= fixed_threshold:
                accepted_tokens.append(draft_token.token_id)
            else:
                rejection_point = i
                break

        acceptance_rate = len(accepted_tokens) / len(draft_tokens)

        return VerificationResult(
            accepted_tokens=accepted_tokens,
            rejection_point=rejection_point,
            acceptance_rate=acceptance_rate,
        )

    def efficiency_gain(self, results: list[VerificationResult]) -> float:
        """Mean acceptance_rate across results; 0.0 if empty."""
        if not results:
            return 0.0
        return sum(r.acceptance_rate for r in results) / len(results)
