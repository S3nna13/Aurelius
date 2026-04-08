"""Response faithfulness checks against supporting evidence."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def support_coverage(response: str, evidence: str) -> float:
    """Fraction of response tokens supported by evidence."""
    response_tokens = _tokens(response)
    evidence_tokens = _tokens(evidence)
    if not response_tokens:
        return 1.0
    return len(response_tokens & evidence_tokens) / len(response_tokens)


def contradiction_rate(response: str, evidence: str, contradiction_terms: tuple[str, ...] = ("not", "never", "no")) -> float:
    """Lightweight contradiction heuristic using negation mismatch."""
    response_lower = response.lower()
    evidence_lower = evidence.lower()
    mismatches = 0
    for term in contradiction_terms:
        in_response = term in response_lower
        in_evidence = term in evidence_lower
        if in_response != in_evidence:
            mismatches += 1
    return mismatches / max(len(contradiction_terms), 1)


def faithfulness_score(response: str, evidence: str) -> float:
    """Combine support coverage and contradiction penalty."""
    coverage = support_coverage(response, evidence)
    contradiction = contradiction_rate(response, evidence)
    return max(coverage - contradiction, 0.0)
