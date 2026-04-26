"""MMLU scoring harness (Hendrycks et al., 2020; arXiv:2009.03300).

Massive Multitask Language Understanding: 57-subject, 4-way multiple-choice
benchmark (A/B/C/D). Supports zero-shot, few-shot (default 5-shot), and
chain-of-thought variants. Accuracy is aggregated overall and per-subject.

Pure stdlib -- re + dataclasses. No silent fallbacks; invalid arguments
raise explicitly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MMLUProblem:
    question_id: str
    subject: str
    question: str
    choices: list[str]
    correct_index: int

    def __post_init__(self) -> None:
        if len(self.choices) != 4:
            raise ValueError(f"MMLU is 4-way multiple choice; got {len(self.choices)} choices")
        if not (0 <= self.correct_index < 4):
            raise ValueError(f"correct_index must be in [0,4); got {self.correct_index}")


@dataclass
class MMLUResult:
    question_id: str
    predicted_letter: str | None
    correct: bool
    subject: str


# ---------------------------------------------------------------------------
# Canonical 5-shot exemplars (from Hendrycks et al., dev-set style)
# ---------------------------------------------------------------------------


CANONICAL_EXEMPLARS: list[MMLUProblem] = [
    MMLUProblem(
        question_id="canonical_0",
        subject="abstract_algebra",
        question="Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.",
        choices=["0", "1", "2", "3"],
        correct_index=1,
    ),
    MMLUProblem(
        question_id="canonical_1",
        subject="anatomy",
        question="Which of the following is not a lobe of the brain?",
        choices=["Frontal", "Temporal", "Cardiac", "Parietal"],
        correct_index=2,
    ),
    MMLUProblem(
        question_id="canonical_2",
        subject="global_facts",
        question="As of 2020, which country had the largest population?",
        choices=["India", "United States", "China", "Indonesia"],
        correct_index=2,
    ),
    MMLUProblem(
        question_id="canonical_3",
        subject="high_school_mathematics",
        question="What is the value of 3! (three factorial)?",
        choices=["3", "6", "9", "27"],
        correct_index=1,
    ),
    MMLUProblem(
        question_id="canonical_4",
        subject="elementary_science",
        question="Which planet is closest to the Sun?",
        choices=["Venus", "Earth", "Mercury", "Mars"],
        correct_index=2,
    ),
]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


COT_SUFFIX = "Let's think step by step."


def _format_example(problem: MMLUProblem, include_answer: bool = False) -> str:
    a, b, c, d = problem.choices
    body = f"{problem.question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
    if include_answer:
        letter = _INDEX_TO_LETTER[problem.correct_index]
        body = body + f"\nAnswer: {letter}"
    return body


def format_prompt(
    problem: MMLUProblem,
    few_shot_examples: list[MMLUProblem] | None = None,
    cot: bool = False,
) -> str:
    """Format an MMLU prompt.

    If ``few_shot_examples`` is given, those exemplars are prepended with
    their gold letters. If ``cot`` is True, the final instruction asks the
    model to reason step by step and end with ``Final answer: (X)``.
    """
    parts: list[str] = []
    subject_name = problem.subject.replace("_", " ")
    parts.append(
        f"The following are multiple choice questions (with answers) about {subject_name}."
    )
    parts.append("")

    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(_format_example(ex, include_answer=True))
            parts.append("")

    parts.append(_format_example(problem, include_answer=False))

    if cot:
        parts.append(
            f"{COT_SUFFIX} Then give your final answer in the form "
            f"'Final answer: (X)' where X is one of A, B, C, D."
        )
    else:
        parts.append("Answer:")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\[\[\s*([A-Da-d])\s*\]\]"),
    re.compile(r"final\s+answer\s*(?:is\s*)?[:\-]?\s*\(?\s*([A-Da-d])\s*\)?", re.IGNORECASE),
    re.compile(r"\banswer\s*(?:is|:)\s*\(?\s*([A-Da-d])\s*\)?", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is\s+\(?\s*([A-Da-d])\s*\)?", re.IGNORECASE),
    re.compile(r"\(\s*([A-Da-d])\s*\)"),
    re.compile(r"\b([A-Da-d])\s*[\).:\-]"),
    re.compile(r"^\s*([A-Da-d])\s*$", re.MULTILINE),
]


def parse_answer_letter(response: str) -> str | None:
    """Extract the chosen letter from a free-form response.

    Returns one of "A", "B", "C", "D" or None if no letter is found.
    """
    if not isinstance(response, str):
        raise TypeError(f"response must be str, got {type(response).__name__}")
    for pat in _ANSWER_PATTERNS:
        m = pat.search(response)
        if m:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


_INDEX_TO_LETTER = ("A", "B", "C", "D")


class MMLUScorer:
    """Scorer for MMLU multiple-choice problems.

    Parameters
    ----------
    generate_fn:
        Optional callable mapping a formatted prompt to the model's raw
        response. Required only for :meth:`run`.
    n_shots:
        Number of few-shot exemplars to include (default 5). Set to 0 for
        zero-shot.
    cot:
        If True, use chain-of-thought prompting (asks the model to reason
        then emit ``Final answer: (X)``).
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str] | None = None,
        n_shots: int = 5,
        cot: bool = False,
    ) -> None:
        if not isinstance(n_shots, int) or n_shots < 0:
            raise ValueError(f"n_shots must be a non-negative int; got {n_shots!r}")
        self.generate_fn = generate_fn
        self.n_shots = n_shots
        self.cot = bool(cot)

    # -- prompt ---------------------------------------------------------

    def format_prompt(
        self,
        problem: MMLUProblem,
        few_shot_examples: list[MMLUProblem] | None = None,
    ) -> str:
        if few_shot_examples is None and self.n_shots > 0:
            few_shot_examples = CANONICAL_EXEMPLARS[: self.n_shots]
        elif few_shot_examples is not None:
            # Truncate to requested n_shots if pool is longer.
            few_shot_examples = list(few_shot_examples)[: self.n_shots]
        return format_prompt(problem, few_shot_examples, cot=self.cot)

    # -- single-example grading ----------------------------------------

    def score_one(self, problem: MMLUProblem, response: str) -> MMLUResult:
        predicted = parse_answer_letter(response)
        gold = _INDEX_TO_LETTER[problem.correct_index]
        correct = predicted is not None and predicted.upper() == gold
        return MMLUResult(
            question_id=problem.question_id,
            predicted_letter=predicted,
            correct=correct,
            subject=problem.subject,
        )

    # -- aggregate scoring ---------------------------------------------

    def score(
        self,
        problems: list[MMLUProblem],
        responses: list[str],
    ) -> dict[str, object]:
        if len(problems) != len(responses):
            raise ValueError(
                f"problems/responses length mismatch: {len(problems)} vs {len(responses)}"
            )

        n = len(problems)
        if n == 0:
            return {
                "overall_accuracy": 0.0,
                "n_valid": 0,
                "n_total": 0,
                "per_subject": {},
            }

        results = [self.score_one(p, r) for p, r in zip(problems, responses)]

        n_correct = sum(1 for r in results if r.correct)
        n_valid = sum(1 for r in results if r.predicted_letter is not None)

        per_subject: dict[str, dict[str, float]] = {}
        for prob, res in zip(problems, results):
            entry = per_subject.setdefault(prob.subject, {"n": 0, "correct": 0, "accuracy": 0.0})
            entry["n"] += 1
            if res.correct:
                entry["correct"] += 1
        for entry in per_subject.values():
            entry["accuracy"] = entry["correct"] / entry["n"] if entry["n"] else 0.0

        return {
            "overall_accuracy": n_correct / n,
            "n_valid": n_valid,
            "n_total": n,
            "per_subject": per_subject,
        }

    # -- end-to-end generation -----------------------------------------

    def run(
        self,
        problems: list[MMLUProblem],
        few_shot_pool: list[MMLUProblem] | None = None,
    ) -> list[MMLUResult]:
        if self.generate_fn is None:
            raise ValueError("MMLUScorer.run requires generate_fn to be provided at construction")

        if self.n_shots > 0:
            if few_shot_pool is None:
                shots = CANONICAL_EXEMPLARS[: self.n_shots]
            else:
                # Use all available if pool shorter than n_shots.
                shots = list(few_shot_pool)[: self.n_shots]
        else:
            shots = None

        out: list[MMLUResult] = []
        for p in problems:
            prompt = format_prompt(p, shots, cot=self.cot)
            resp = self.generate_fn(prompt)
            if not isinstance(resp, str):
                raise TypeError(f"generate_fn must return str; got {type(resp).__name__}")
            out.append(self.score_one(p, resp))
        return out


__all__ = [
    "MMLUProblem",
    "MMLUResult",
    "MMLUScorer",
    "parse_answer_letter",
    "format_prompt",
    "CANONICAL_EXEMPLARS",
    "COT_SUFFIX",
]
