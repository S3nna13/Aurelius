"""
Personality Router

Standalone, stdlib-only router that maps a natural-language task description to
one or more agent "personalities". Inspired by the Architect/Detective/Guardian
coordinator in the autonomous-coding-agent reference, extended with ANALYST and
TEACHER personalities.

Keyword-based scoring: count how many personality keywords appear in the task,
pick the highest-scoring as primary, everything else with a non-zero score is
secondary. Collaboration is required when two or more secondaries trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class PersonalityType(str, Enum):
    """Enumerated personalities the router can dispatch to."""

    ARCHITECT = "architect"
    DETECTIVE = "detective"
    GUARDIAN = "guardian"
    ANALYST = "analyst"
    TEACHER = "teacher"


@dataclass(frozen=True)
class TaskAnalysis:
    """Immutable analysis result produced by ``PersonalityRouter.analyze``."""

    primary: PersonalityType
    secondary: List[PersonalityType]
    confidence: float
    requires_collaboration: bool


PERSONALITY_KEYWORDS: Dict[PersonalityType, List[str]] = {
    PersonalityType.ARCHITECT: [
        "design", "build", "implement", "create", "develop", "refactor",
        "architecture", "structure", "pattern", "scaffold", "compose", "plan",
    ],
    PersonalityType.DETECTIVE: [
        "debug", "fix", "investigate", "trace", "diagnose", "bug",
        "error", "issue", "broken", "crash", "reproduce", "locate",
    ],
    PersonalityType.GUARDIAN: [
        "security", "audit", "maintain", "vulnerability", "compliance",
        "harden", "monitor", "backup", "recovery", "upgrade", "dependency",
        "permissions",
    ],
    PersonalityType.ANALYST: [
        "analyze", "evaluate", "measure", "benchmark", "profile", "compare",
        "metric", "statistics", "report", "assess", "quantify", "inspect",
    ],
    PersonalityType.TEACHER: [
        "explain", "document", "guide", "teach", "tutorial", "describe",
        "walkthrough", "comment", "clarify", "onboard", "introduction",
        "example",
    ],
}


class PersonalityRouter:
    """Route free-form tasks to one or more ``PersonalityType`` handlers."""

    def __init__(self, keywords: Dict[PersonalityType, List[str]] | None = None) -> None:
        self.keywords: Dict[PersonalityType, List[str]] = dict(
            keywords if keywords is not None else PERSONALITY_KEYWORDS
        )

    # ------------------------------------------------------------------ core
    def _score(self, task: str) -> Dict[PersonalityType, int]:
        task_lower = task.lower()
        scores: Dict[PersonalityType, int] = {}
        for personality, words in self.keywords.items():
            scores[personality] = sum(1 for w in words if w in task_lower)
        return scores

    def analyze(self, task: str) -> TaskAnalysis:
        scores = self._score(task)
        total = sum(scores.values())
        # Pick primary = max score. On tie, prefer the enum declaration order.
        primary = max(
            self.keywords.keys(),
            key=lambda p: (scores[p], -list(PersonalityType).index(p)),
        )
        primary_score = scores[primary]
        # Secondary: any other personality with score > 0, ordered by score desc.
        secondary = sorted(
            [p for p, s in scores.items() if s > 0 and p != primary],
            key=lambda p: (-scores[p], list(PersonalityType).index(p)),
        )
        confidence = (primary_score / total) if total > 0 else 0.0
        requires_collaboration = len(secondary) >= 2
        return TaskAnalysis(
            primary=primary,
            secondary=list(secondary),
            confidence=float(confidence),
            requires_collaboration=requires_collaboration,
        )

    def route(self, task: str) -> List[PersonalityType]:
        analysis = self.analyze(task)
        return [analysis.primary] + list(analysis.secondary)

    def explain(self, task: str) -> str:
        analysis = self.analyze(task)
        scores = self._score(task)
        lines = [
            f"Task: {task!r}",
            f"Primary: {analysis.primary.value} (score={scores[analysis.primary]})",
        ]
        if analysis.secondary:
            secondary_parts = ", ".join(
                f"{p.value}({scores[p]})" for p in analysis.secondary
            )
            lines.append(f"Secondary: {secondary_parts}")
        else:
            lines.append("Secondary: none")
        lines.append(f"Confidence: {analysis.confidence:.2f}")
        lines.append(
            "Collaboration: required" if analysis.requires_collaboration else "Collaboration: not required"
        )
        return "\n".join(lines)


PERSONALITY_ROUTER_REGISTRY = {"default": PersonalityRouter}
