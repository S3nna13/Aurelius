"""Personality facet — keyword-triggered personality traits from PersonalityRouter."""

from __future__ import annotations

from ..unified_persona import PersonaFacet

PERSONALITY_KEYWORDS: dict[str, list[str]] = {
    "architect": [
        "design",
        "build",
        "implement",
        "create",
        "develop",
        "refactor",
        "architecture",
        "structure",
        "pattern",
        "scaffold",
        "compose",
        "plan",
    ],
    "detective": [
        "debug",
        "fix",
        "investigate",
        "trace",
        "diagnose",
        "bug",
        "error",
        "issue",
        "broken",
        "crash",
        "reproduce",
        "locate",
    ],
    "guardian": [
        "security",
        "audit",
        "maintain",
        "vulnerability",
        "compliance",
        "harden",
        "monitor",
        "backup",
        "recovery",
        "upgrade",
        "dependency",
        "permissions",
    ],
    "analyst": [
        "analyze",
        "evaluate",
        "measure",
        "benchmark",
        "profile",
        "compare",
        "metric",
        "statistics",
        "report",
        "assess",
        "quantify",
        "inspect",
    ],
    "teacher": [
        "explain",
        "document",
        "guide",
        "teach",
        "tutorial",
        "describe",
        "walkthrough",
        "comment",
        "clarify",
        "onboard",
        "introduction",
        "example",
    ],
}


def create_personality_facet(traits: list[str] | None = None) -> PersonaFacet:
    return PersonaFacet(
        facet_type="personality",
        config={"traits": traits or []},
    )


def score_input(facet: PersonaFacet, text: str) -> dict[str, float]:
    traits = facet.config.get("traits", [])
    if not traits:
        return {}
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for trait in traits:
        keywords = PERSONALITY_KEYWORDS.get(trait, [])
        if keywords:
            hits = sum(1 for kw in keywords if kw in text_lower)
            scores[trait] = hits / len(keywords) if keywords else 0.0
    return scores
