"""MT-Bench LLM-as-judge scoring harness (Zheng et al., 2023; arXiv:2306.05685).

Given a multi-turn conversation and candidate responses, formats prompts for a
judge LLM, parses numerical scores (1-10) or pairwise verdicts, and aggregates.
The judge is caller-provided via ``judge_fn: Callable[[str], str]``.

Supports:

* Single-answer scoring (1-10 scale)
* Pairwise comparison (A / B / tie)
* Multi-turn scoring (all conversation turns formatted in the prompt)
* Score parsing with graceful fallback on malformed output (``None``, never 0)
* Aggregation: mean, median, win-rate (pairwise)

Pure stdlib: ``re`` + ``statistics``. No silent fallbacks -- an unparseable
judge output produces an explicit ``None`` / ``"invalid"`` result.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MTBenchQuestion:
    """A single MT-Bench question, possibly multi-turn."""

    question_id: str
    category: str
    turns: list[str]
    reference: str | None = None


@dataclass
class SingleAnswerScore:
    question_id: str
    score: float | None
    judge_output: str
    judge_reasoning: str


@dataclass
class PairwiseResult:
    question_id: str
    winner: str  # one of {"A", "B", "tie", "invalid"}
    judge_output: str


# ---------------------------------------------------------------------------
# Prompt templates (Zheng et al., 2023, Appendix E)
# ---------------------------------------------------------------------------


SINGLE_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your "
    "evaluation should consider factors such as the helpfulness, relevance, "
    "accuracy, depth, creativity, and level of detail of the response. Begin "
    "your evaluation by providing a short explanation. Be as objective as "
    "possible. After providing your explanation, please rate the response on a "
    'scale of 1 to 10 by strictly following this format: "[[rating]]", for '
    'example: "Rating: [[5]]".'
)

SINGLE_SYSTEM_PROMPT_WITH_REF = (
    "Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your "
    "evaluation should consider correctness and helpfulness. You will be given "
    "a reference answer and the assistant's answer. Begin your evaluation by "
    "comparing the assistant's answer with the reference answer. Identify and "
    "correct any mistakes. Be as objective as possible. After providing your "
    "explanation, please rate the response on a scale of 1 to 10 by strictly "
    'following this format: "[[rating]]".'
)

PAIRWISE_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses "
    "provided by two AI assistants to the user question displayed below. You "
    "should choose the assistant that follows the user's instructions and "
    "answers the user's question better. Avoid position bias. Be as objective "
    "as possible. After providing your explanation, output your final verdict "
    'by strictly following this format: "[[A]]" if assistant A is better, '
    '"[[B]]" if assistant B is better, and "[[C]]" for a tie.'
)

PAIRWISE_SYSTEM_PROMPT_WITH_REF = (
    "Please act as an impartial judge and evaluate the quality of the responses "
    "provided by two AI assistants to the user question displayed below. You "
    "will be given a reference answer. Your evaluation should consider "
    "correctness and helpfulness. Avoid position bias. After providing your "
    "explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better, '
    'and "[[C]]" for a tie.'
)


# Regexes for parsing judge output. ``[[N]]`` preferred; ``Rating: N`` fallback.
_BRACKET_NUM_RE = re.compile(r"\[\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]\]")
_RATING_RE = re.compile(r"[Rr]ating\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
_BRACKET_VERDICT_RE = re.compile(r"\[\[\s*([A-Za-z]+)\s*\]\]")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_turns(turns: list[str]) -> str:
    """Render a multi-turn conversation as numbered turns."""
    if not turns:
        return ""
    if len(turns) == 1:
        return turns[0]
    return "\n\n".join(f"### User Turn {i + 1}\n{t}" for i, t in enumerate(turns))


def _format_single_prompt(question: MTBenchQuestion, answer: str) -> str:
    system = (
        SINGLE_SYSTEM_PROMPT_WITH_REF if question.reference is not None else SINGLE_SYSTEM_PROMPT
    )
    parts = [system, "", "[Question]", _format_turns(question.turns)]
    if question.reference is not None:
        parts += ["", "[Reference Answer]", question.reference]
    parts += ["", "[The Start of Assistant's Answer]", answer, "[The End of Assistant's Answer]"]
    return "\n".join(parts)


def _format_pairwise_prompt(question: MTBenchQuestion, answer_a: str, answer_b: str) -> str:
    system = (
        PAIRWISE_SYSTEM_PROMPT_WITH_REF
        if question.reference is not None
        else PAIRWISE_SYSTEM_PROMPT
    )
    parts = [system, "", "[Question]", _format_turns(question.turns)]
    if question.reference is not None:
        parts += ["", "[Reference Answer]", question.reference]
    parts += [
        "",
        "[The Start of Assistant A's Answer]",
        answer_a,
        "[The End of Assistant A's Answer]",
        "",
        "[The Start of Assistant B's Answer]",
        answer_b,
        "[The End of Assistant B's Answer]",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_single_score(output: str) -> float | None:
    """Parse a score in [1, 10]; return None on malformed or out-of-range.

    Out-of-range values are rejected (returned as None) rather than clipped,
    per spec. A valid 0.0 cannot occur because the scale is 1-10.
    """
    if not isinstance(output, str):
        return None
    m = _BRACKET_NUM_RE.search(output)
    if m is None:
        m = _RATING_RE.search(output)
    if m is None:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    if val < 1.0 or val > 10.0:
        return None
    return val


def _parse_pairwise_verdict(output: str) -> str:
    """Parse a pairwise verdict. Returns one of {'A','B','tie','invalid'}."""
    if not isinstance(output, str):
        return "invalid"
    m = _BRACKET_VERDICT_RE.search(output)
    if m is None:
        return "invalid"
    token = m.group(1).strip().lower()
    if token == "a":
        return "A"
    if token == "b":
        return "B"
    if token in ("c", "tie"):
        return "tie"
    return "invalid"


def _extract_reasoning(output: str) -> str:
    """Return everything before the final ``[[...]]`` verdict, trimmed."""
    if not isinstance(output, str):
        return ""
    # Take text up to the last bracketed token, if any.
    last = None
    for m in _BRACKET_NUM_RE.finditer(output):
        last = m
    if last is None:
        for m in _BRACKET_VERDICT_RE.finditer(output):
            last = m
    if last is None:
        return output.strip()
    return output[: last.start()].strip()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class MTBenchJudge:
    """MT-Bench LLM-as-judge scoring harness.

    The judge is any callable mapping a formatted prompt string to the judge's
    raw textual output. Parsing is resilient: malformed output yields explicit
    ``None`` (single) or ``"invalid"`` (pairwise) markers, never silent zeros.
    """

    def __init__(self, judge_fn: Callable[[str], str]) -> None:
        if not callable(judge_fn):
            raise TypeError("judge_fn must be callable")
        self._judge_fn = judge_fn

    # ------------------------------------------------------------------ format
    def format_single_prompt(self, question: MTBenchQuestion, answer: str) -> str:
        return _format_single_prompt(question, answer)

    def format_pairwise_prompt(
        self,
        question: MTBenchQuestion,
        answer_a: str,
        answer_b: str,
    ) -> str:
        return _format_pairwise_prompt(question, answer_a, answer_b)

    # ------------------------------------------------------------------- score
    def score_single(self, question: MTBenchQuestion, answer: str) -> SingleAnswerScore:
        prompt = _format_single_prompt(question, answer)
        try:
            output = self._judge_fn(prompt)
        except Exception as exc:  # noqa: BLE001 - we explicitly mark invalid
            return SingleAnswerScore(
                question_id=question.question_id,
                score=None,
                judge_output="",
                judge_reasoning=f"judge_fn raised: {type(exc).__name__}: {exc}",
            )
        if not isinstance(output, str):
            return SingleAnswerScore(
                question_id=question.question_id,
                score=None,
                judge_output=str(output) if output is not None else "",
                judge_reasoning="judge_fn returned non-string output",
            )
        score = _parse_single_score(output)
        reasoning = _extract_reasoning(output)
        return SingleAnswerScore(
            question_id=question.question_id,
            score=score,
            judge_output=output,
            judge_reasoning=reasoning,
        )

    def score_pairwise(
        self,
        question: MTBenchQuestion,
        answer_a: str,
        answer_b: str,
    ) -> PairwiseResult:
        prompt = _format_pairwise_prompt(question, answer_a, answer_b)
        try:
            output = self._judge_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            return PairwiseResult(
                question_id=question.question_id,
                winner="invalid",
                judge_output=f"judge_fn raised: {type(exc).__name__}: {exc}",
            )
        if not isinstance(output, str):
            return PairwiseResult(
                question_id=question.question_id,
                winner="invalid",
                judge_output=str(output) if output is not None else "",
            )
        winner = _parse_pairwise_verdict(output)
        return PairwiseResult(
            question_id=question.question_id,
            winner=winner,
            judge_output=output,
        )

    # -------------------------------------------------------------- aggregate
    @staticmethod
    def aggregate_single(results: list[SingleAnswerScore]) -> dict:
        """Return {mean, median, n_valid, n_total} over non-None scores.

        Empty or all-invalid input yields zeros -- no exception.
        """
        valid = [r.score for r in results if r.score is not None]
        n_total = len(results)
        n_valid = len(valid)
        if n_valid == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "n_valid": 0,
                "n_total": n_total,
            }
        return {
            "mean": float(statistics.fmean(valid)),
            "median": float(statistics.median(valid)),
            "n_valid": n_valid,
            "n_total": n_total,
        }

    @staticmethod
    def aggregate_pairwise(results: list[PairwiseResult]) -> dict:
        """Return {win_rate_a, win_rate_b, tie_rate, n_valid, n_total}.

        Rates are over valid results only (excluding ``invalid``), so
        ``win_rate_a + win_rate_b + tie_rate == 1.0`` when n_valid > 0.
        Empty / all-invalid input yields zeros.
        """
        n_total = len(results)
        a = sum(1 for r in results if r.winner == "A")
        b = sum(1 for r in results if r.winner == "B")
        t = sum(1 for r in results if r.winner == "tie")
        n_valid = a + b + t
        if n_valid == 0:
            return {
                "win_rate_a": 0.0,
                "win_rate_b": 0.0,
                "tie_rate": 0.0,
                "n_valid": 0,
                "n_total": n_total,
            }
        return {
            "win_rate_a": a / n_valid,
            "win_rate_b": b / n_valid,
            "tie_rate": t / n_valid,
            "n_valid": n_valid,
            "n_total": n_total,
        }


__all__ = [
    "MTBenchQuestion",
    "SingleAnswerScore",
    "PairwiseResult",
    "MTBenchJudge",
    "SINGLE_SYSTEM_PROMPT",
    "SINGLE_SYSTEM_PROMPT_WITH_REF",
    "PAIRWISE_SYSTEM_PROMPT",
    "PAIRWISE_SYSTEM_PROMPT_WITH_REF",
]
