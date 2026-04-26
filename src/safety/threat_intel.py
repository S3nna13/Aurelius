"""Local threat-intelligence prioritization engine.

Inspired by BUGBOUNTY_AGENT/intel/epss.py but strictly local (no network):
combines EPSS exploitation probability with CVSS base score to produce a
unified priority signal suitable for offline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ThreatLevel(StrEnum):
    """Coarse-grained threat level labels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True)
class ThreatEntry:
    """A single vulnerability threat-intelligence record."""

    vuln_id: str
    epss_score: float
    cvss_score: float
    threat_level: ThreatLevel
    tags: list[str]
    notes: str = ""


def _combined(epss: float, cvss: float) -> float:
    """Weighted score: 60% EPSS + 40% normalized CVSS (both clamped 0..1)."""
    e = max(0.0, min(1.0, float(epss)))
    c = max(0.0, min(10.0, float(cvss)))
    return 0.6 * e + 0.4 * (c / 10.0)


class ThreatIntelEngine:
    """In-memory threat-intelligence store + prioritization."""

    def __init__(self) -> None:
        self._entries: dict[str, ThreatEntry] = {}

    def add(self, entry: ThreatEntry) -> None:
        """Insert or replace a threat entry keyed by vuln_id."""
        self._entries[entry.vuln_id] = entry

    def get(self, vuln_id: str) -> ThreatEntry | None:
        """Retrieve a threat entry by vuln_id or None."""
        return self._entries.get(vuln_id)

    def score(self, vuln_id: str) -> float:
        """Return the combined priority score for a vuln_id (0.0 if unknown)."""
        entry = self._entries.get(vuln_id)
        if entry is None:
            return 0.0
        return _combined(entry.epss_score, entry.cvss_score)

    def prioritize(self, vuln_ids: list[str]) -> list[ThreatEntry]:
        """Return known entries sorted by combined score descending."""
        entries = [self._entries[v] for v in vuln_ids or [] if v in self._entries]
        entries.sort(key=lambda e: _combined(e.epss_score, e.cvss_score), reverse=True)
        return entries

    def high_risk_ids(self, threshold: float = 0.7) -> list[str]:
        """Return vuln IDs whose combined score meets/exceeds threshold."""
        out: list[tuple[str, float]] = []
        for vid, entry in self._entries.items():
            s = _combined(entry.epss_score, entry.cvss_score)
            if s >= threshold:
                out.append((vid, s))
        out.sort(key=lambda t: t[1], reverse=True)
        return [vid for vid, _ in out]

    def to_dict(self) -> dict:
        """Serialize all entries to a plain dict."""
        return {
            "entries": [
                {
                    "vuln_id": e.vuln_id,
                    "epss_score": e.epss_score,
                    "cvss_score": e.cvss_score,
                    "threat_level": e.threat_level.value,
                    "tags": list(e.tags),
                    "notes": e.notes,
                }
                for e in self._entries.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThreatIntelEngine:
        """Load an engine from a dict produced by ``to_dict``."""
        engine = cls()
        for raw in (data or {}).get("entries", []) or []:
            level_raw = raw.get("threat_level", "info")
            try:
                level = ThreatLevel(level_raw)
            except ValueError:
                level = ThreatLevel.INFO
            entry = ThreatEntry(
                vuln_id=str(raw.get("vuln_id", "")),
                epss_score=float(raw.get("epss_score", 0.0)),
                cvss_score=float(raw.get("cvss_score", 0.0)),
                threat_level=level,
                tags=list(raw.get("tags", []) or []),
                notes=str(raw.get("notes", "")),
            )
            if entry.vuln_id:
                engine.add(entry)
        return engine


THREAT_INTEL_REGISTRY: dict[str, type[ThreatIntelEngine]] = {"default": ThreatIntelEngine}
