"""Constitutional scoring utilities for principle-based evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstitutionalPrinciple:
    name: str
    weight: float
    preferred_terms: tuple[str, ...]
    disallowed_terms: tuple[str, ...] = ()


def principle_score(text: str, principle: ConstitutionalPrinciple) -> float:
    """Score text against one constitutional principle using lexical cues."""
    lowered = text.lower()
    positive_hits = sum(term.lower() in lowered for term in principle.preferred_terms)
    negative_hits = sum(term.lower() in lowered for term in principle.disallowed_terms)
    raw_score = positive_hits - negative_hits
    return principle.weight * raw_score


def constitutional_score(text: str, principles: list[ConstitutionalPrinciple]) -> float:
    """Aggregate weighted constitutional scores."""
    return sum(principle_score(text, principle) for principle in principles)


def principle_breakdown(text: str, principles: list[ConstitutionalPrinciple]) -> dict[str, float]:
    """Return per-principle contributions."""
    return {principle.name: principle_score(text, principle) for principle in principles}


def passes_constitution(
    text: str, principles: list[ConstitutionalPrinciple], threshold: float = 0.0
) -> bool:
    """Check whether text exceeds a constitutional threshold."""
    return constitutional_score(text, principles) >= threshold
