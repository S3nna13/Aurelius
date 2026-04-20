"""Hallucination guard: claim-vs-evidence validator.

A deterministic, stdlib-only heuristic that validates model-produced claims
against a corpus of evidence (tool outputs, retrieved docs, system facts).

A claim is considered VALIDATED when at least one evidence item covers a
majority of its key terms (lexical overlap above ``similarity_threshold``)
and any numeric values in the claim match a numeric value in that evidence.

A claim is considered CONTRADICTED when evidence contains either:

* a negation of the claim (e.g. claim "X is Y" vs evidence "X is not Y"), or
* a conflicting numeric value (same surrounding key terms, different number).

Unvalidated-but-not-contradicted claims are returned as ``unvalidated_claims``
so downstream policies can decide to withhold, caveat, or request more tools.

Inspired by BlueGuardian AI's ``core/hallucination_guard.py`` but reimplemented
from scratch with no third-party dependencies beyond Python stdlib.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """A single factual assertion produced by a model."""

    text: str
    source_model: str = "unknown"
    confidence: float = 1.0


@dataclass(frozen=True)
class Evidence:
    """A single piece of supporting material."""

    text: str
    source: str = "unknown"
    source_type: str = "generic"  # e.g. "tool_output", "retrieved_doc", "system"


@dataclass
class ValidationResult:
    """Outcome of validating a batch of claims against evidence."""

    is_valid: bool
    confidence: float
    validated_claims: list[Claim] = field(default_factory=list)
    unvalidated_claims: list[Claim] = field(default_factory=list)
    contradictions: list[tuple[Claim, Evidence]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenization / key-term extraction
# ---------------------------------------------------------------------------


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "doing", "have", "has", "had", "having",
    "and", "or", "but", "if", "then", "else", "when", "while", "for",
    "of", "on", "in", "at", "to", "from", "by", "with", "about", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "should", "now", "that", "this",
    "these", "those", "it", "its", "they", "them", "their", "i", "you",
    "he", "she", "we", "us", "me", "my", "your", "our", "his", "her",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")
# Numeric literal: optional sign, integer or decimal, optional scientific form.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

_NEGATION_TOKENS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "cannot", "cant", "isnt", "arent",
    "wasnt", "werent", "dont", "doesnt", "didnt", "wont", "wouldnt",
    "shouldnt", "couldnt", "hasnt", "havent", "hadnt",
})


def _normalize(text: str) -> str:
    """Unicode-normalize, lowercase, and strip accents for robust matching."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.lower()


def extract_key_terms(text: str) -> set[str]:
    """Tokenize ``text`` and return the set of non-stop-word key terms.

    Pure regex + stopword removal; no nltk. Numbers are excluded (they are
    handled separately by :func:`extract_numbers`). Accents are stripped and
    text is lowercased so matching is case- and diacritic-insensitive.
    """
    if text is None:
        return set()
    norm = _normalize(text)
    terms: set[str] = set()
    for match in _TOKEN_RE.findall(norm):
        if _NUMBER_RE.fullmatch(match):
            continue
        if match in _STOP_WORDS:
            continue
        if len(match) < 2:
            continue
        terms.add(match)
    return terms


def extract_numbers(text: str) -> list[float]:
    """Extract numeric literals as floats from ``text``.

    Malformed parses are silently skipped but the scan itself never raises.
    """
    if text is None:
        return []
    out: list[float] = []
    for tok in _NUMBER_RE.findall(_normalize(text)):
        try:
            out.append(float(tok))
        except ValueError:
            # Impossible given the regex, but fail-safe.
            continue
    return out


def _has_negation(text: str) -> bool:
    norm = _normalize(text)
    tokens = _TOKEN_RE.findall(norm)
    return any(t in _NEGATION_TOKENS for t in tokens)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class HallucinationGuard:
    """Validate claims against evidence via lexical + numeric overlap.

    Parameters
    ----------
    similarity_threshold:
        Fraction of a claim's key terms that must appear in a single evidence
        item for that evidence to count as "covering" the claim. Must lie in
        ``(0.0, 1.0]``.
    confidence_decay:
        Multiplicative penalty applied to the aggregate confidence for each
        unvalidated claim. Must lie in ``[0.0, 1.0]``.
    numeric_tolerance:
        Relative tolerance for numeric equality. ``1e-6`` is strict exact
        match for integers; bump for floaty domains.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        confidence_decay: float = 0.5,
        numeric_tolerance: float = 1e-6,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {similarity_threshold!r}"
            )
        if not (0.0 <= confidence_decay <= 1.0):
            raise ValueError(
                f"confidence_decay must be in [0, 1], got {confidence_decay!r}"
            )
        if numeric_tolerance < 0.0:
            raise ValueError(
                f"numeric_tolerance must be >= 0, got {numeric_tolerance!r}"
            )
        self.similarity_threshold = float(similarity_threshold)
        self.confidence_decay = float(confidence_decay)
        self.numeric_tolerance = float(numeric_tolerance)

    # -- internal helpers --------------------------------------------------

    def _numbers_match(self, a: float, b: float) -> bool:
        if a == b:
            return True
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) <= self.numeric_tolerance * scale

    def _lexical_overlap(self, claim_terms: set[str], evidence_terms: set[str]) -> float:
        if not claim_terms:
            return 0.0
        return len(claim_terms & evidence_terms) / len(claim_terms)

    def _classify_against(
        self,
        claim: Claim,
        evidence_list: Sequence[Evidence],
    ) -> tuple[str, Evidence | None]:
        """Return (status, evidence) where status in {validated, contradicted, unvalidated}."""
        claim_terms = extract_key_terms(claim.text)
        claim_numbers = extract_numbers(claim.text)
        claim_negated = _has_negation(claim.text)

        best_support: Evidence | None = None
        best_contradiction: Evidence | None = None

        for ev in evidence_list:
            ev_terms = extract_key_terms(ev.text)
            overlap = self._lexical_overlap(claim_terms, ev_terms)
            if overlap < self.similarity_threshold:
                continue

            ev_negated = _has_negation(ev.text)
            ev_numbers = extract_numbers(ev.text)

            # Negation mismatch -> contradiction
            if claim_negated != ev_negated:
                best_contradiction = ev
                continue

            # Numeric check: if claim has numbers, require at least one match.
            if claim_numbers:
                if not ev_numbers:
                    # Lexically similar but numbers absent -> ambiguous, skip
                    continue
                all_match = all(
                    any(self._numbers_match(cn, en) for en in ev_numbers)
                    for cn in claim_numbers
                )
                if not all_match:
                    # Numeric conflict under matching key-terms -> contradiction
                    best_contradiction = ev
                    continue

            best_support = ev
            break

        if best_support is not None:
            return "validated", best_support
        if best_contradiction is not None:
            return "contradicted", best_contradiction
        return "unvalidated", None

    # -- public API --------------------------------------------------------

    def validate(
        self,
        claims: Iterable[Claim],
        evidence: Iterable[Evidence],
    ) -> ValidationResult:
        """Validate each claim against the evidence set.

        Never raises on malformed input; instead degrades gracefully and
        accumulates entries in ``result.warnings``.
        """
        warnings: list[str] = []

        safe_claims: list[Claim] = []
        for idx, c in enumerate(claims or []):
            if not isinstance(c, Claim):
                warnings.append(f"claim[{idx}] is not a Claim instance; skipping")
                continue
            if not isinstance(c.text, str) or not c.text.strip():
                warnings.append(f"claim[{idx}] has empty/non-string text; skipping")
                continue
            safe_claims.append(c)

        safe_evidence: list[Evidence] = []
        for idx, e in enumerate(evidence or []):
            if not isinstance(e, Evidence):
                warnings.append(f"evidence[{idx}] is not an Evidence instance; skipping")
                continue
            if not isinstance(e.text, str):
                warnings.append(f"evidence[{idx}] has non-string text; skipping")
                continue
            safe_evidence.append(e)

        validated: list[Claim] = []
        unvalidated: list[Claim] = []
        contradictions: list[tuple[Claim, Evidence]] = []

        if not safe_claims:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                validated_claims=[],
                unvalidated_claims=[],
                contradictions=[],
                warnings=warnings or ["no claims provided"],
            )

        if not safe_evidence:
            # With no evidence, every claim is unvalidated.
            unvalidated = list(safe_claims)
            decayed = 1.0
            for c in safe_claims:
                decayed *= self.confidence_decay * max(0.0, min(1.0, c.confidence))
            return ValidationResult(
                is_valid=False,
                confidence=decayed,
                validated_claims=[],
                unvalidated_claims=unvalidated,
                contradictions=[],
                warnings=warnings + ["no evidence provided"],
            )

        aggregate_conf = 1.0
        for claim in safe_claims:
            c_conf = max(0.0, min(1.0, float(claim.confidence)))
            status, ev = self._classify_against(claim, safe_evidence)
            if status == "validated":
                validated.append(claim)
                aggregate_conf *= c_conf
            elif status == "contradicted":
                assert ev is not None  # invariant from _classify_against
                contradictions.append((claim, ev))
                aggregate_conf *= 0.0
            else:  # unvalidated
                unvalidated.append(claim)
                aggregate_conf *= self.confidence_decay * c_conf

        is_valid = not unvalidated and not contradictions
        return ValidationResult(
            is_valid=is_valid,
            confidence=aggregate_conf,
            validated_claims=validated,
            unvalidated_claims=unvalidated,
            contradictions=contradictions,
            warnings=warnings,
        )


__all__ = [
    "Claim",
    "Evidence",
    "ValidationResult",
    "HallucinationGuard",
    "extract_key_terms",
    "extract_numbers",
]
