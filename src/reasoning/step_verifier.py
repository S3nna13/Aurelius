"""Process reward model verifier for step-level reasoning quality.

Inspired by Let's Verify Step by Step (Lightman et al. 2305.20050) and
Math-Shepherd (Wang et al. 2312.08935); Aurelius-native stub + interface.
License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

_MAX_STEP_LEN = 8192
_MAX_STEPS = 512


class VerificationLabel(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NEUTRAL = "neutral"


@dataclass
class StepScore:
    step_index: int
    content: str
    label: VerificationLabel
    confidence: float           # 0.0–1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.step_index < 0:
            raise ValueError(f"step_index must be >= 0, got {self.step_index}")


class StepVerifier:
    """Verifier that scores reasoning steps for correctness.

    Default implementation uses simple heuristics (length, keyword presence).
    Intended to be subclassed with a learned PRM.
    """

    def __init__(self, min_step_length: int = 5,
                 contradiction_keywords: list[str] | None = None) -> None:
        self.min_step_length = min_step_length
        self.contradiction_keywords: list[str] = contradiction_keywords or [
            "therefore not", "hence not", "which means not",
            "this is wrong", "incorrect because", "this fails",
        ]

    def verify_step(self, step_index: int, content: str,
                    context: str = "") -> StepScore:
        """Verify a single step. Returns StepScore with label + confidence."""
        if len(content) > _MAX_STEP_LEN:
            raise ValueError(f"step content exceeds {_MAX_STEP_LEN} chars")
        if step_index < 0:
            raise ValueError(f"step_index must be >= 0")

        content_lower = content.lower()
        # Heuristic 1: too short → neutral
        if len(content.strip()) < self.min_step_length:
            return StepScore(step_index=step_index, content=content,
                             label=VerificationLabel.NEUTRAL, confidence=0.5,
                             rationale="step too short to evaluate")
        # Heuristic 2: contains contradiction keyword → incorrect
        for kw in self.contradiction_keywords:
            if kw in content_lower:
                return StepScore(step_index=step_index, content=content,
                                 label=VerificationLabel.INCORRECT, confidence=0.75,
                                 rationale=f"contradiction keyword detected: {kw!r}")
        # Default: correct with moderate confidence
        # Confidence scales with step length (up to a cap)
        conf = min(0.5 + len(content.strip()) / 2000.0, 0.85)
        return StepScore(step_index=step_index, content=content,
                         label=VerificationLabel.CORRECT, confidence=conf,
                         rationale="heuristic pass")

    def verify_chain(self, steps: list[str],
                     context: str = "") -> list[StepScore]:
        """Verify a sequence of steps. Fails loudly if list is too long."""
        if len(steps) > _MAX_STEPS:
            raise ValueError(f"too many steps (max {_MAX_STEPS})")
        return [self.verify_step(i, s, context) for i, s in enumerate(steps)]

    def aggregate_score(self, scores: list[StepScore]) -> float:
        """Aggregate step scores into a chain-level score in [0, 1].

        Strategy: geometric mean of per-step confidence * label_weight.
        Incorrect steps with high confidence drag the score down sharply.
        """
        if not scores:
            return 0.0
        label_weights = {
            VerificationLabel.CORRECT: 1.0,
            VerificationLabel.NEUTRAL: 0.5,
            VerificationLabel.INCORRECT: 0.0,
        }
        import math
        product = 1.0
        for s in scores:
            w = label_weights[s.label]
            adjusted = s.confidence * w + (1 - s.confidence) * 0.1
            product *= adjusted
        return product ** (1.0 / len(scores))


STEP_VERIFIER = StepVerifier()
