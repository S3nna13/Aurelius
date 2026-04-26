"""CVSS assessor — draft vector + score estimation without external deps.

Adapted from BUGBOUNTY_AGENT/reporting/cvss.py. Stdlib-only: does not require
the `cvss` library; uses a static severity -> score table and vector parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CVSSVersion(StrEnum):
    """Supported CVSS specification versions."""

    V31 = "3.1"
    V40 = "4.0"


@dataclass(frozen=True)
class CVSSMetric:
    """A parsed CVSS metric / vector result."""

    vector: str
    base_score: float
    severity: str
    is_draft: bool = False


_DRAFT_DISCLAIMER = " (DRAFT — pending manual CVSS assessment)"

_SEVERITY_VECTORS: dict[str, str] = {
    "critical": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    "high": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:L",
    "medium": "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:L/I:L/A:N",
    "low": "CVSS:3.1/AV:N/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N",
    "info": "CVSS:3.1/AV:N/AC:H/PR:H/UI:R/S:U/C:N/I:N/A:N",
}

_STATIC_SCORES: dict[str, float] = {
    "critical": 9.8,
    "high": 7.1,
    "medium": 4.2,
    "low": 2.4,
    "info": 0.0,
}

_SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


class CVSSAssessor:
    """Estimate CVSS vectors / scores for a given severity label."""

    def severity_to_vector(self, severity: str) -> str:
        """Return a draft CVSS vector string annotated with a disclaimer."""
        key = (severity or "").lower().strip()
        raw = _SEVERITY_VECTORS.get(key, _SEVERITY_VECTORS["info"])
        return raw + _DRAFT_DISCLAIMER

    def severity_to_score(self, severity: str) -> float:
        """Return the static base score for a severity label."""
        key = (severity or "").lower().strip()
        return _STATIC_SCORES.get(key, 0.0)

    def parse_vector(self, vector: str) -> dict[str, str]:
        """Parse a CVSS:3.1 vector into a mapping of metric keys to values."""
        out: dict[str, str] = {}
        if not vector:
            return out
        # strip draft disclaimer if present
        raw = vector.split(" (DRAFT")[0].strip()
        if not (raw.startswith("CVSS:3.1") or raw.startswith("CVSS:4.0")):
            return out
        parts = raw.split("/")
        if parts:
            head = parts[0]
            if ":" in head:
                k, v = head.split(":", 1)
                out[k] = v
        for part in parts[1:]:
            if ":" in part:
                k, v = part.split(":", 1)
                out[k] = v
        return out

    def vector_to_metric(self, vector: str, is_draft: bool = False) -> CVSSMetric:
        """Build a CVSSMetric from a vector, estimating score + severity."""
        parsed = self.parse_vector(vector)
        c = parsed.get("C", "N")
        i = parsed.get("I", "N")
        a = parsed.get("A", "N")

        if c == "H" and i == "H" and a == "H":
            severity = "critical"
        elif "H" in (c, i, a):
            severity = "high"
        elif "L" in (c, i, a):
            # medium vs low: distinguish by AC
            ac = parsed.get("AC", "H")
            pr = parsed.get("PR", "N")
            severity = "low" if (ac == "H" and pr == "H") else "medium"
        else:
            severity = "info"

        base_score = _STATIC_SCORES.get(severity, 0.0)
        return CVSSMetric(
            vector=vector,
            base_score=base_score,
            severity=severity,
            is_draft=is_draft,
        )

    def compare_severity(self, s1: str, s2: str) -> int:
        """Return -1/0/1 by severity ordering (info<low<medium<high<critical)."""
        a = _SEVERITY_ORDER.get((s1 or "").lower().strip(), -1)
        b = _SEVERITY_ORDER.get((s2 or "").lower().strip(), -1)
        if a < b:
            return -1
        if a > b:
            return 1
        return 0


CVSS_ASSESSOR_REGISTRY: dict[str, type[CVSSAssessor]] = {"default": CVSSAssessor}
