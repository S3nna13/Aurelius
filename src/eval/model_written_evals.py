"""Utilities for model-written evaluation forms and rubric aggregation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FIELD_RE = re.compile(r"^\s*([A-Za-z_ ]+)\s*:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class WrittenEval:
    verdict: str
    rationale: str
    confidence: float
    rubric_score: float


def parse_written_eval(text: str) -> WrittenEval:
    """Parse a structured written-eval text block."""
    fields = {
        key.strip().lower().replace(" ", "_"): value.strip()
        for key, value in _FIELD_RE.findall(text)
    }
    required = ("verdict", "rationale", "confidence", "rubric_score")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    return WrittenEval(
        verdict=fields["verdict"],
        rationale=fields["rationale"],
        confidence=float(fields["confidence"]),
        rubric_score=float(fields["rubric_score"]),
    )


def weighted_verdict_score(evals: list[WrittenEval], positive_label: str = "pass") -> float:
    """Average confidence-weighted support for a positive verdict."""
    if not evals:
        return 0.0
    weighted = [
        evaluation.confidence
        for evaluation in evals
        if evaluation.verdict.lower() == positive_label
    ]
    return sum(weighted) / len(evals)


def average_rubric_score(evals: list[WrittenEval]) -> float:
    """Average rubric score over multiple written evals."""
    if not evals:
        return 0.0
    return sum(evaluation.rubric_score for evaluation in evals) / len(evals)


def written_eval_agreement(evals: list[WrittenEval]) -> float:
    """Fraction of evals matching the majority verdict."""
    if not evals:
        return 0.0
    counts: dict[str, int] = {}
    for evaluation in evals:
        verdict = evaluation.verdict.lower()
        counts[verdict] = counts.get(verdict, 0) + 1
    return max(counts.values()) / len(evals)
