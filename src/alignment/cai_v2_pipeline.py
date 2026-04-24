"""Constitutional AI v2 Pipeline: multi-principle critique → revise → evaluate.

Reference: Bai et al. 2022 "Constitutional AI: Harmlessness from AI Feedback"
"""
from __future__ import annotations

from dataclasses import dataclass, field


_FLAGGED_WORDS: frozenset[str] = frozenset(
    ["harm", "illegal", "dangerous", "kill", "attack", "exploit"]
)

_DEFAULT_PRINCIPLES: list[dict[str, object]] = [
    {
        "principle_id": "hhh_helpful",
        "text": "Be helpful to the user.",
        "weight": 1.0,
        "category": "helpfulness",
    },
    {
        "principle_id": "hhh_harmless",
        "text": "Be harmless and avoid dangerous content.",
        "weight": 1.5,
        "category": "safety",
    },
    {
        "principle_id": "hhh_honest",
        "text": "Be honest and do not deceive.",
        "weight": 1.0,
        "category": "ethics",
    },
]


@dataclass
class Principle:
    principle_id: str
    text: str
    weight: float = 1.0
    category: str = "general"


@dataclass
class CritiqueResult:
    principle_id: str
    critique: str
    violation_score: float
    revision_suggestion: str


@dataclass
class CAIv2Result:
    original_response: str
    critiques: list[CritiqueResult]
    revised_response: str
    overall_harmlessness_score: float


class ConstitutionalAIv2Pipeline:
    """Multi-principle CAI pipeline: critique → revise → evaluate (Bai et al. 2022)."""

    def __init__(self, principles: list[Principle] | None = None) -> None:
        if principles is not None:
            self._principles: list[Principle] = list(principles)
        else:
            self._principles = [Principle(**p) for p in _DEFAULT_PRINCIPLES]  # type: ignore[arg-type]

    def add_principle(self, principle: Principle) -> None:
        self._principles.append(principle)

    def remove_principle(self, principle_id: str) -> None:
        self._principles = [p for p in self._principles if p.principle_id != principle_id]

    @property
    def principles(self) -> list[Principle]:
        return list(self._principles)

    def critique(self, response: str, principle: Principle) -> CritiqueResult:
        lower = response.lower()
        hits = sum(1 for w in _FLAGGED_WORDS if w in lower)
        score = min(1.0, hits / 10.0)
        critique_text = (
            f"Response may violate '{principle.text}' with score {score:.2f}"
        )
        return CritiqueResult(
            principle_id=principle.principle_id,
            critique=critique_text,
            violation_score=score,
            revision_suggestion="Please revise to be more aligned with the principle.",
        )

    def revise(self, response: str, critiques: list[CritiqueResult]) -> str:
        if any(c.violation_score > 0.3 for c in critiques):
            return "[Revised per principles] " + response
        return response

    def evaluate(self, original: str, revised: str) -> float:
        if len(revised) > len(original) and "Revised" in revised:
            return 0.8
        return 0.5

    def run(self, response: str) -> CAIv2Result:
        critiques = [self.critique(response, p) for p in self._principles]
        revised = self.revise(response, critiques)

        total_weight = sum(p.weight for p in self._principles)
        if total_weight == 0.0:
            overall = 0.0
        else:
            weighted_harmlessness = sum(
                (1.0 - c.violation_score) * p.weight
                for c, p in zip(critiques, self._principles)
            )
            overall = weighted_harmlessness / total_weight

        return CAIv2Result(
            original_response=response,
            critiques=critiques,
            revised_response=revised,
            overall_harmlessness_score=overall,
        )

    def run_batch(self, responses: list[str]) -> list[CAIv2Result]:
        return [self.run(r) for r in responses]
