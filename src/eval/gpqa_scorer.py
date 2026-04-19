"""GPQA scoring harness (Rein et al., 2023; arXiv:2311.12022).

Graduate-Level Google-Proof Q&A: 4-way multiple-choice questions written by
domain experts. The scorer extracts the model's chosen letter (A/B/C/D) via
regex patterns and grades against a gold index. Accuracy is aggregated
overall, per-domain (biology/chemistry/physics), and per-difficulty.

Pure stdlib -- re + dataclasses. No silent fallbacks; invalid arguments
raise explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GPQAProblem:
    question_id: str
    question: str
    choices: List[str]
    correct_index: int
    domain: str = "general"
    difficulty: str = "hard"

    def __post_init__(self) -> None:
        if len(self.choices) != 4:
            raise ValueError(
                f"GPQA is 4-way multiple choice; got {len(self.choices)} choices"
            )
        if not (0 <= self.correct_index < 4):
            raise ValueError(
                f"correct_index must be in [0,4); got {self.correct_index}"
            )


@dataclass
class GPQAResult:
    question_id: str
    predicted_letter: Optional[str]
    correct: bool
    raw_response: str


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


PROMPT_TEMPLATE = (
    "{question}\n\n"
    "A) {a}\n"
    "B) {b}\n"
    "C) {c}\n"
    "D) {d}\n\n"
    "Answer (just the letter):"
)


def format_prompt(problem: GPQAProblem) -> str:
    a, b, c, d = problem.choices
    return PROMPT_TEMPLATE.format(
        question=problem.question, a=a, b=b, c=c, d=d
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


# Patterns are checked in order; first match wins. Each pattern captures one
# letter in group 1. Inputs are matched case-insensitively; the returned
# letter is always uppercased A/B/C/D.
_ANSWER_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\[\[\s*([A-Da-d])\s*\]\]"),
    re.compile(r"\banswer\s*(?:is|:)?\s*\(?\s*([A-Da-d])\s*\)?", re.IGNORECASE),
    re.compile(r"final\s+answer\s+(?:is\s+)?\(?\s*([A-Da-d])\s*\)?", re.IGNORECASE),
    re.compile(r"\(\s*([A-Da-d])\s*\)"),
    re.compile(r"\b([A-Da-d])\s*[\).:\-]"),
    re.compile(r"^\s*([A-Da-d])\s*$", re.MULTILINE),
]


def parse_answer_letter(response: str) -> Optional[str]:
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


class GPQAScorer:
    """Scorer for GPQA multiple-choice problems.

    Parameters
    ----------
    generate_fn:
        Optional callable mapping a formatted prompt to the model's raw
        response. Required only for :meth:`run`.
    case_insensitive:
        If True (default), letter comparison ignores case. The parser
        already normalizes to upper-case so this is effectively always on,
        but the flag is preserved for symmetry with other scorers and so
        callers can opt into strict-case matching.
    """

    def __init__(
        self,
        generate_fn: Optional[Callable[[str], str]] = None,
        case_insensitive: bool = True,
    ) -> None:
        self.generate_fn = generate_fn
        self.case_insensitive = bool(case_insensitive)

    # -- prompt ---------------------------------------------------------

    @staticmethod
    def format_prompt(problem: GPQAProblem) -> str:
        return format_prompt(problem)

    # -- single-example grading ----------------------------------------

    def score_one(self, problem: GPQAProblem, response: str) -> GPQAResult:
        predicted = parse_answer_letter(response)
        gold = _INDEX_TO_LETTER[problem.correct_index]
        if predicted is None:
            correct = False
        elif self.case_insensitive:
            correct = predicted.upper() == gold.upper()
        else:
            correct = predicted == gold
        return GPQAResult(
            question_id=problem.question_id,
            predicted_letter=predicted,
            correct=correct,
            raw_response=response,
        )

    # -- aggregate scoring ---------------------------------------------

    def score(
        self,
        problems: List[GPQAProblem],
        responses: List[str],
    ) -> Dict[str, object]:
        if len(problems) != len(responses):
            raise ValueError(
                f"problems/responses length mismatch: "
                f"{len(problems)} vs {len(responses)}"
            )

        n = len(problems)
        if n == 0:
            return {
                "overall_accuracy": 0.0,
                "n_valid": 0,
                "n_total": 0,
                "per_domain": {},
                "per_difficulty": {},
            }

        results = [self.score_one(p, r) for p, r in zip(problems, responses)]

        n_correct = sum(1 for r in results if r.correct)
        n_valid = sum(1 for r in results if r.predicted_letter is not None)

        per_domain: Dict[str, Dict[str, float]] = {}
        per_difficulty: Dict[str, Dict[str, float]] = {}

        def _bump(
            bucket: Dict[str, Dict[str, float]],
            key: str,
            correct: bool,
        ) -> None:
            entry = bucket.setdefault(key, {"n": 0, "correct": 0, "accuracy": 0.0})
            entry["n"] += 1
            if correct:
                entry["correct"] += 1

        for prob, res in zip(problems, results):
            _bump(per_domain, prob.domain, res.correct)
            _bump(per_difficulty, prob.difficulty, res.correct)

        for bucket in (per_domain, per_difficulty):
            for entry in bucket.values():
                entry["accuracy"] = entry["correct"] / entry["n"] if entry["n"] else 0.0

        return {
            "overall_accuracy": n_correct / n,
            "n_valid": n_valid,
            "n_total": n,
            "per_domain": per_domain,
            "per_difficulty": per_difficulty,
        }

    # -- end-to-end generation -----------------------------------------

    def run(self, problems: List[GPQAProblem]) -> List[GPQAResult]:
        if self.generate_fn is None:
            raise ValueError(
                "GPQAScorer.run requires generate_fn to be provided at construction"
            )
        out: List[GPQAResult] = []
        for p in problems:
            prompt = format_prompt(p)
            resp = self.generate_fn(prompt)
            if not isinstance(resp, str):
                raise TypeError(
                    f"generate_fn must return str; got {type(resp).__name__}"
                )
            out.append(self.score_one(p, resp))
        return out


__all__ = [
    "GPQAProblem",
    "GPQAResult",
    "GPQAScorer",
    "parse_answer_letter",
    "format_prompt",
    "PROMPT_TEMPLATE",
]
