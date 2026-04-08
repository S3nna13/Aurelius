"""Local utilities for aggregating model-as-judge style rubric scores."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SCORE_RE = re.compile(r"([a-zA-Z_]+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class JudgeRubric:
    coherence: float
    helpfulness: float
    safety: float

    @property
    def overall(self) -> float:
        return (self.coherence + self.helpfulness + self.safety) / 3.0


def parse_judge_output(text: str) -> JudgeRubric:
    """Parse a rubric-style text blob into numeric scores."""
    found = {name.lower(): float(value) for name, value in _SCORE_RE.findall(text)}
    required = ("coherence", "helpfulness", "safety")
    missing = [name for name in required if name not in found]
    if missing:
        raise ValueError(f"Missing rubric fields: {missing}")
    return JudgeRubric(
        coherence=found["coherence"],
        helpfulness=found["helpfulness"],
        safety=found["safety"],
    )


def pairwise_winner(left: JudgeRubric, right: JudgeRubric) -> str:
    """Return the winner under mean rubric score."""
    if left.overall > right.overall:
        return "left"
    if right.overall > left.overall:
        return "right"
    return "tie"


def judge_agreement(verdicts: list[str]) -> float:
    """Fraction of verdicts matching the majority label."""
    if not verdicts:
        return 0.0
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    majority = max(counts.values())
    return majority / len(verdicts)


def average_rubric(rubrics: list[JudgeRubric]) -> JudgeRubric:
    """Average multiple rubric outputs."""
    if not rubrics:
        raise ValueError("rubrics must be non-empty")
    return JudgeRubric(
        coherence=sum(rubric.coherence for rubric in rubrics) / len(rubrics),
        helpfulness=sum(rubric.helpfulness for rubric in rubrics) / len(rubrics),
        safety=sum(rubric.safety for rubric in rubrics) / len(rubrics),
    )
