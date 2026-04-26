"""Multi-check verifier for security-finding records.

Stdlib-only. Adapted from the bugbounty-agent critic/verifier reference.
Runs a small set of deterministic checks and returns a structured,
frozen ``VerificationResult`` with per-check accounting and a confidence
score derived from the pass ratio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})

SEVERITY_CVSS_MINIMUMS: dict[str, float] = {
    "critical": 9.0,
    "high": 7.0,
    "medium": 4.0,
    "low": 0.1,
    "info": 0.0,
}

_GENERIC_TITLE_BLOCKLIST = frozenset(
    {
        "finding",
        "vulnerability",
        "issue",
        "bug",
        "test",
        "todo",
        "fixme",
        "untitled",
        "n/a",
        "none",
    }
)

_INJECTION_PATTERNS = [
    re.compile(r"Confidence\s*\(\s*\d+\s*-\s*\d+\s*\)\s*:\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE),
]

_MIN_DESCRIPTION_LEN = 20


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    confidence: float
    reason: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


def _sanitize_for_verifier(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


class FindingVerifier:
    """Runs deterministic checks on a prospective finding."""

    CHECK_TITLE = "title_not_generic"
    CHECK_SEVERITY = "severity_valid"
    CHECK_DESCRIPTION = "description_length"
    CHECK_CVSS = "cvss_consistent"

    def verify(
        self,
        title: str,
        severity: str,
        description: str,
        cvss_score: float | None = None,
    ) -> VerificationResult:
        passed: list[str] = []
        failed: list[str] = []
        reasons: list[str] = []

        title_norm = (title or "").strip().lower()
        if title_norm and title_norm not in _GENERIC_TITLE_BLOCKLIST:
            passed.append(self.CHECK_TITLE)
        else:
            failed.append(self.CHECK_TITLE)
            reasons.append(f"generic_or_empty_title: {title!r}")

        sev_norm = (severity or "").strip().lower()
        if sev_norm in VALID_SEVERITIES:
            passed.append(self.CHECK_SEVERITY)
        else:
            failed.append(self.CHECK_SEVERITY)
            reasons.append(f"invalid_severity: {severity!r}")

        desc_clean = _sanitize_for_verifier(description or "")
        if len(desc_clean) >= _MIN_DESCRIPTION_LEN:
            passed.append(self.CHECK_DESCRIPTION)
        else:
            failed.append(self.CHECK_DESCRIPTION)
            reasons.append(f"description_too_short: {len(desc_clean)} < {_MIN_DESCRIPTION_LEN}")

        total_checks = 3
        if cvss_score is not None:
            total_checks = 4
            try:
                cvss_val = float(cvss_score)
            except (TypeError, ValueError):
                failed.append(self.CHECK_CVSS)
                reasons.append(f"cvss_not_numeric: {cvss_score!r}")
            else:
                min_cvss = SEVERITY_CVSS_MINIMUMS.get(sev_norm, 0.0)
                if sev_norm in VALID_SEVERITIES and cvss_val >= min_cvss:
                    passed.append(self.CHECK_CVSS)
                else:
                    failed.append(self.CHECK_CVSS)
                    reasons.append(
                        f"cvss_severity_mismatch: severity={sev_norm!r} requires "
                        f"cvss>={min_cvss} got {cvss_val}"
                    )

        all_passed = len(failed) == 0
        confidence = len(passed) / total_checks if total_checks else 0.0
        reason = "ok" if all_passed else "; ".join(reasons)
        return VerificationResult(
            passed=all_passed,
            confidence=confidence,
            reason=reason,
            checks_passed=passed,
            checks_failed=failed,
        )


FINDING_VERIFIER_REGISTRY = {"default": FindingVerifier}
