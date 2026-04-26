"""Dataset-level evaluation for lexical safety rubrics."""

from __future__ import annotations

from dataclasses import dataclass

from src.alignment.safety_rubric import SafetyRubric, passes_safety_rubric, safety_score


@dataclass(frozen=True)
class SafetyEvalReport:
    mean_score: float
    pass_rate: float
    min_score: float
    max_score: float


def evaluate_safety_texts(
    texts: list[str], rubric: SafetyRubric, threshold: float = 0.0
) -> SafetyEvalReport:
    """Evaluate a batch of texts against a safety rubric."""
    if not texts:
        return SafetyEvalReport(mean_score=0.0, pass_rate=0.0, min_score=0.0, max_score=0.0)
    scores = [safety_score(text, rubric) for text in texts]
    passes = [passes_safety_rubric(text, rubric, threshold=threshold) for text in texts]
    return SafetyEvalReport(
        mean_score=sum(scores) / len(scores),
        pass_rate=sum(passes) / len(passes),
        min_score=min(scores),
        max_score=max(scores),
    )


def compare_safety_reports(left: SafetyEvalReport, right: SafetyEvalReport) -> str:
    """Choose the better report using pass rate then mean score."""
    if (left.pass_rate, left.mean_score) > (right.pass_rate, right.mean_score):
        return "left"
    if (right.pass_rate, right.mean_score) > (left.pass_rate, left.mean_score):
        return "right"
    return "tie"
