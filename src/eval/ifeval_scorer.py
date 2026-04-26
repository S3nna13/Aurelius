"""IFEval scoring harness (Zhou et al., 2023; arXiv:2311.07911).

Evaluates instruction-following by checking verifiable constraints on model
outputs. Each problem has a prompt and a list of (constraint_type, kwargs)
tuples; the scorer aggregates per-constraint pass/fail into strict/loose
accuracy.

Pure stdlib: json + re only. No silent fallbacks -- unknown constraint types
raise ValueError.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IFEvalConstraint:
    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class IFEvalProblem:
    prompt: str
    constraints: list[IFEvalConstraint]


@dataclass
class IFEvalResult:
    passed: list[bool]
    strict_pass: bool
    per_constraint: list[tuple[str, bool]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENT_SPLIT_RE.split(text.strip()) if p.strip()]
    return len(parts)


def _loose_variants(text: str) -> list[str]:
    """Variants used for loose accuracy (Zhou et al. Appendix): strip
    markdown emphasis and leading/trailing whitespace, try lower-cased."""
    variants = [text]
    stripped = text.strip()
    variants.append(stripped)
    no_md = re.sub(r"[*_`]", "", stripped)
    variants.append(no_md)
    # Drop leading/trailing lines that look like preamble/epilogue.
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= 2:
        variants.append("\n".join(lines[1:]))
        variants.append("\n".join(lines[:-1]))
        variants.append("\n".join(lines[1:-1]))
    # Deduplicate preserving order.
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Constraint checkers
# ---------------------------------------------------------------------------


def _check_length_words(response: str, kwargs: dict[str, Any]) -> bool:
    n = _word_count(response)
    lo = kwargs.get("min")
    hi = kwargs.get("max")
    if lo is not None and n < lo:
        return False
    if hi is not None and n > hi:
        return False
    return True


def _check_length_sentences(response: str, kwargs: dict[str, Any]) -> bool:
    n = _sentence_count(response)
    lo = kwargs.get("min")
    hi = kwargs.get("max")
    if lo is not None and n < lo:
        return False
    if hi is not None and n > hi:
        return False
    return True


def _check_contains_keyword(response: str, kwargs: dict[str, Any]) -> bool:
    if "keyword" not in kwargs:
        raise ValueError("contains_keyword requires 'keyword'")
    kw = kwargs["keyword"]
    cs = bool(kwargs.get("case_sensitive", False))
    hay = response if cs else response.lower()
    needle = kw if cs else kw.lower()
    return needle in hay


def _check_avoids_keyword(response: str, kwargs: dict[str, Any]) -> bool:
    if "keyword" not in kwargs:
        raise ValueError("avoids_keyword requires 'keyword'")
    kw = kwargs["keyword"]
    cs = bool(kwargs.get("case_sensitive", False))
    hay = response if cs else response.lower()
    needle = kw if cs else kw.lower()
    return needle not in hay


def _check_case(response: str, kwargs: dict[str, Any]) -> bool:
    mode = kwargs.get("mode")
    if mode not in ("lower", "upper", "title"):
        raise ValueError(f"case mode must be lower/upper/title, got {mode!r}")
    stripped = response.strip()
    if not stripped:
        return False
    if mode == "lower":
        return stripped == stripped.lower()
    if mode == "upper":
        return stripped == stripped.upper()
    # title: every alphabetic word starts with an uppercase letter; rest lower.
    words = stripped.split()
    if not words:
        return False
    for w in words:
        # Strip surrounding punctuation for the check.
        core = re.sub(r"^[^\w]+|[^\w]+$", "", w)
        if not core:
            continue
        if not core[0].isupper():
            return False
        if len(core) > 1 and not core[1:].islower():
            return False
    return True


def _check_json_format(response: str, kwargs: dict[str, Any]) -> bool:
    text = response.strip()
    # Allow fenced code blocks like ```json ... ```
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text:
        return False
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


def _check_start_with(response: str, kwargs: dict[str, Any]) -> bool:
    if "phrase" not in kwargs:
        raise ValueError("start_with requires 'phrase'")
    phrase = kwargs["phrase"]
    cs = bool(kwargs.get("case_sensitive", False))
    text = response.strip()
    if cs:
        return text.startswith(phrase)
    return text.lower().startswith(phrase.lower())


def _check_end_with(response: str, kwargs: dict[str, Any]) -> bool:
    if "phrase" not in kwargs:
        raise ValueError("end_with requires 'phrase'")
    phrase = kwargs["phrase"]
    cs = bool(kwargs.get("case_sensitive", False))
    text = response.strip()
    if cs:
        return text.endswith(phrase)
    return text.lower().endswith(phrase.lower())


def _check_min_bullets(response: str, kwargs: dict[str, Any]) -> bool:
    n = kwargs.get("n")
    if n is None:
        raise ValueError("min_bullets requires 'n'")
    marker = kwargs.get("marker", "- ")
    count = 0
    for line in response.splitlines():
        if line.lstrip().startswith(marker):
            count += 1
    return count >= int(n)


def _check_placeholder_present(response: str, kwargs: dict[str, Any]) -> bool:
    marker = kwargs.get("marker")
    if marker is None:
        raise ValueError("placeholder_present requires 'marker'")
    min_count = int(kwargs.get("min_count", 1))
    return response.count(marker) >= min_count


def _check_quote_count(response: str, kwargs: dict[str, Any]) -> bool:
    # Count matched pairs of straight double quotes.
    lo = kwargs.get("min")
    hi = kwargs.get("max")
    matches = re.findall(r'"[^"\n]*"', response)
    n = len(matches)
    if lo is not None and n < lo:
        return False
    if hi is not None and n > hi:
        return False
    return True


def _check_max_punctuation(response: str, kwargs: dict[str, Any]) -> bool:
    chars = kwargs.get("chars")
    hi = kwargs.get("max")
    if chars is None or hi is None:
        raise ValueError("max_punctuation requires 'chars' and 'max'")
    total = sum(1 for c in response if c in chars)
    return total <= int(hi)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


CheckerFn = Callable[[str, dict[str, Any]], bool]


class IFEvalScorer:
    """Scores model responses against verifiable IFEval constraints."""

    CHECKERS: dict[str, CheckerFn] = {
        "length_words": _check_length_words,
        "length_sentences": _check_length_sentences,
        "contains_keyword": _check_contains_keyword,
        "avoids_keyword": _check_avoids_keyword,
        "case": _check_case,
        "json_format": _check_json_format,
        "start_with": _check_start_with,
        "end_with": _check_end_with,
        "min_bullets": _check_min_bullets,
        "placeholder_present": _check_placeholder_present,
        "quote_count": _check_quote_count,
        "max_punctuation": _check_max_punctuation,
    }

    def __init__(self, case_sensitive_default: bool = False) -> None:
        self.case_sensitive_default = bool(case_sensitive_default)

    # -- single-problem scoring -------------------------------------------------

    def _check_one(self, constraint: IFEvalConstraint, response: str) -> bool:
        fn = self.CHECKERS.get(constraint.type)
        if fn is None:
            raise ValueError(f"Unknown IFEval constraint type: {constraint.type!r}")
        kwargs = dict(constraint.kwargs)
        # Inject default case-sensitivity for text-matching constraints.
        if constraint.type in ("contains_keyword", "avoids_keyword", "start_with", "end_with"):
            kwargs.setdefault("case_sensitive", self.case_sensitive_default)
        return bool(fn(response, kwargs))

    def score_one(self, problem: IFEvalProblem, response: str) -> IFEvalResult:
        passed: list[bool] = []
        per_constraint: list[tuple[str, bool]] = []
        for c in problem.constraints:
            ok = self._check_one(c, response)
            passed.append(ok)
            per_constraint.append((c.type, ok))
        strict = bool(passed) and all(passed)
        return IFEvalResult(
            passed=passed,
            strict_pass=strict,
            per_constraint=per_constraint,
        )

    def _loose_pass(self, problem: IFEvalProblem, response: str) -> bool:
        """Loose accuracy: strict_pass holds under at least one variant."""
        for variant in _loose_variants(response):
            results = [self._check_one(c, variant) for c in problem.constraints]
            if results and all(results):
                return True
        return False

    # -- aggregate scoring ------------------------------------------------------

    def score(
        self,
        problems: list[IFEvalProblem],
        responses: list[str],
    ) -> dict[str, Any]:
        if len(problems) != len(responses):
            raise ValueError(
                f"problems ({len(problems)}) and responses ({len(responses)}) "
                "must have equal length"
            )
        if not problems:
            raise ValueError("problems/responses must be non-empty")

        strict_hits = 0
        loose_hits = 0
        per_type_total: dict[str, int] = {}
        per_type_pass: dict[str, int] = {}

        for problem, response in zip(problems, responses):
            result = self.score_one(problem, response)
            if result.strict_pass:
                strict_hits += 1
            if self._loose_pass(problem, response):
                loose_hits += 1
            for ctype, ok in result.per_constraint:
                per_type_total[ctype] = per_type_total.get(ctype, 0) + 1
                if ok:
                    per_type_pass[ctype] = per_type_pass.get(ctype, 0) + 1

        n = len(problems)
        per_type_accuracy = {t: per_type_pass.get(t, 0) / per_type_total[t] for t in per_type_total}
        return {
            "strict_accuracy": strict_hits / n,
            "loose_accuracy": loose_hits / n,
            "per_type_accuracy": per_type_accuracy,
            "n_problems": n,
        }


__all__ = [
    "IFEvalConstraint",
    "IFEvalProblem",
    "IFEvalResult",
    "IFEvalScorer",
]
