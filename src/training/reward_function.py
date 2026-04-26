"""Reward function framework for RL/GRPO training (Aurelius).

Generalised from Heavens_Gate bear_market_reward.py. Provides a small
abstract base class plus three concrete reward functions covering the
most common shaping signals: directional correctness, length, and
format/pattern compliance.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class RewardSignal(Enum):
    """Coarse reward classification buckets."""

    CORRECT = 1.0
    PARTIALLY_CORRECT = 0.5
    INCORRECT = 0.0
    PENALIZED = -0.5


@dataclass(frozen=True)
class RewardResult:
    """Result of scoring a single completion."""

    score: float
    signal: RewardSignal
    breakdown: dict = field(default_factory=dict)
    notes: str = ""


class BaseRewardFunction(ABC):
    """Abstract base for reward functions."""

    @abstractmethod
    def score(self, completion: str, ground_truth: str, **kwargs: object) -> RewardResult:
        """Score a single completion against ground truth."""


# ---------------------------------------------------------------------------
# DirectionalRewardFn
# ---------------------------------------------------------------------------

_DIRECTION_PATTERN = re.compile(r"\b(bullish|bearish|neutral|long|short|buy|sell|hold)\b", re.I)

_DIRECTION_MAP: dict[str, str] = {
    "bullish": "bullish",
    "long": "bullish",
    "buy": "bullish",
    "bearish": "bearish",
    "short": "bearish",
    "sell": "bearish",
    "neutral": "neutral",
    "hold": "neutral",
}


def _classify(score: float) -> RewardSignal:
    if score >= 1.0:
        return RewardSignal.CORRECT
    if score > 0.0:
        return RewardSignal.PARTIALLY_CORRECT
    if score < 0.0:
        return RewardSignal.PENALIZED
    return RewardSignal.INCORRECT


class DirectionalRewardFn(BaseRewardFunction):
    """Score directional (bullish/bearish/neutral) predictions.

    Rubric (ported from bear_market_reward):
      +1.0 correct directional prediction
      -0.5 wrong direction
      +0.2 includes a quantified confidence or price target
      +0.1 reasoning chain (>= 20 words)
    Clamped to [-0.5, 1.0].
    """

    def score(self, completion: str, ground_truth: str, **kwargs: object) -> RewardResult:
        gt_raw = ground_truth.strip().lower()
        gt_direction = _DIRECTION_MAP.get(gt_raw, gt_raw)
        breakdown: dict[str, float] = {}
        score = 0.0

        found = _DIRECTION_PATTERN.findall(completion)
        pred: str | None = None
        if found:
            pred = _DIRECTION_MAP.get(found[0].lower(), found[0].lower())
            if pred == gt_direction:
                score += 1.0
                breakdown["direction"] = 1.0
            elif gt_direction:
                score -= 0.5
                breakdown["direction"] = -0.5

        has_price_target = bool(re.search(r"\$[\d,]+|\d+%|\d+\.\d+", completion.lower()))
        if has_price_target:
            score += 0.2
            breakdown["price_target"] = 0.2

        word_count = len(completion.split())
        if word_count >= 20:
            score += 0.1
            breakdown["reasoning"] = 0.1

        score = max(-0.5, min(1.0, score))
        notes = f"predicted={pred} truth={gt_direction} words={word_count}"
        return RewardResult(score=score, signal=_classify(score), breakdown=breakdown, notes=notes)

    def score_batch(self, completions: list[str], ground_truth: str) -> list[RewardResult]:
        """Score a batch of completions against a shared ground truth."""
        return [self.score(c, ground_truth) for c in completions]


# ---------------------------------------------------------------------------
# LengthRewardFn
# ---------------------------------------------------------------------------


class LengthRewardFn(BaseRewardFunction):
    """Reward responses whose token count falls in [min_tokens, max_tokens].

    In-range -> 1.0. Outside the range, score decays linearly toward 0.0
    at 2x the gap from the nearest bound (floored at 0.0).
    """

    def __init__(self, min_tokens: int = 50, max_tokens: int = 500) -> None:
        if min_tokens < 0 or max_tokens < min_tokens:
            raise ValueError("require 0 <= min_tokens <= max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def score(self, completion: str, ground_truth: str = "", **kwargs: object) -> RewardResult:
        tokens = len(completion.split())
        if self.min_tokens <= tokens <= self.max_tokens:
            score = 1.0
        elif tokens < self.min_tokens:
            gap = self.min_tokens - tokens
            score = max(0.0, 1.0 - gap / max(self.min_tokens, 1))
        else:
            gap = tokens - self.max_tokens
            score = max(0.0, 1.0 - gap / max(self.max_tokens, 1))
        breakdown = {"tokens": float(tokens)}
        return RewardResult(
            score=score,
            signal=_classify(score),
            breakdown=breakdown,
            notes=f"tokens={tokens} range=[{self.min_tokens},{self.max_tokens}]",
        )


# ---------------------------------------------------------------------------
# FormatRewardFn
# ---------------------------------------------------------------------------


class FormatRewardFn(BaseRewardFunction):
    """Reward responses matching required patterns and avoiding forbidden ones.

    All required_patterns must match -> +1.0.
    Any forbidden_patterns present -> -0.5.
    Otherwise 0.0.
    """

    def __init__(
        self,
        required_patterns: list[str] | None = None,
        forbidden_patterns: list[str] | None = None,
    ) -> None:
        self.required_patterns = list(required_patterns or [])
        self.forbidden_patterns = list(forbidden_patterns or [])
        self._required = [re.compile(p) for p in self.required_patterns]
        self._forbidden = [re.compile(p) for p in self.forbidden_patterns]

    def score(self, completion: str, ground_truth: str = "", **kwargs: object) -> RewardResult:
        breakdown: dict[str, float] = {}

        required_hits = sum(1 for p in self._required if p.search(completion))
        forbidden_hits = sum(1 for p in self._forbidden if p.search(completion))
        breakdown["required_hits"] = float(required_hits)
        breakdown["forbidden_hits"] = float(forbidden_hits)

        if forbidden_hits > 0:
            score = -0.5
        elif self._required and required_hits == len(self._required):
            score = 1.0
        else:
            score = 0.0

        return RewardResult(
            score=score,
            signal=_classify(score),
            breakdown=breakdown,
            notes=(
                f"required {required_hits}/{len(self._required)} "
                f"forbidden {forbidden_hits}/{len(self._forbidden)}"
            ),
        )


REWARD_FUNCTION_REGISTRY = {
    "directional": DirectionalRewardFn,
    "length": LengthRewardFn,
    "format": FormatRewardFn,
}
