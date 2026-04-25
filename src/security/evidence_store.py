"""SQLite-backed finding persistence with SARIF 2.1 export.

Findings are stored in a lightweight SQLite database (defaulting to
in-memory) and can be exported as a SARIF 2.1 document for integration
with standard security tooling.
"""

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

try:
    from src.security import SECURITY_REGISTRY
except ImportError:
    SECURITY_REGISTRY: dict = {}


Severity = Literal["info", "low", "medium", "high", "critical"]

_SEV_ORDER: dict = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

_SEV_MAP: dict = {
    "info": "note",
    "low": "note",
    "medium": "warning",
    "high": "error",
    "critical": "error",
}

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS findings (
        id        TEXT PRIMARY KEY,
        tool      TEXT NOT NULL,
        severity  TEXT NOT NULL,
        title     TEXT NOT NULL,
        url       TEXT,
        details   TEXT,
        poc       TEXT,
        created_at REAL NOT NULL
    )
"""


@dataclass
class Finding:
    tool: str
    severity: Severity
    title: str
    url: str = ""
    details: str = ""
    poc: str = ""
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


class EvidenceStore:
    """Persist security findings to SQLite and export as SARIF 2.1.

    Parameters
    ----------
    db_path:
        File path for the SQLite database, or ``":memory:"`` for an ephemeral
        in-process store (the default).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_finding(self, finding: Finding) -> None:
        """Insert or replace a finding (keyed by ``finding_id``)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?,?)",
            (
                finding.finding_id,
                finding.tool,
                finding.severity,
                finding.title,
                finding.url,
                finding.details,
                finding.poc,
                finding.created_at,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_findings(self, min_severity: Severity = "info") -> list:
        """Return all findings at or above *min_severity*, ordered by creation time."""
        min_order = _SEV_ORDER[min_severity]
        rows = self._conn.execute(
            "SELECT * FROM findings ORDER BY created_at"
        ).fetchall()
        result = []
        for r in rows:
            if _SEV_ORDER.get(r[2], 0) >= min_order:
                result.append(
                    Finding(
                        tool=r[1],
                        severity=r[2],
                        title=r[3],
                        url=r[4] or "",
                        details=r[5] or "",
                        poc=r[6] or "",
                        finding_id=r[0],
                        created_at=r[7],
                    )
                )
        return result

    def finding_count(self) -> int:
        """Return the total number of findings stored."""
        return self._conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_sarif(self, tool_name: str = "aurelius-security") -> dict:
        """Export all findings as a SARIF 2.1.0 document (dict)."""
        findings = self.list_findings()
        results = []
        for f in findings:
            results.append(
                {
                    "ruleId": f.finding_id,
                    "level": _SEV_MAP.get(f.severity, "note"),
                    "message": {"text": f.title},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": f.url or "unknown"
                                }
                            }
                        }
                    ],
                }
            )
        return {
            "version": "2.1.0",
            "$schema": (
                "https://raw.githubusercontent.com/oasis-tcs/sarif-spec"
                "/master/Schemata/sarif-schema-2.1.0.json"
            ),
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": tool_name,
                            "version": "1.0.0",
                        }
                    },
                    "results": results,
                }
            ],
        }


# Register the class reference (not an instance) so callers can
# instantiate with custom db_path values.
SECURITY_REGISTRY["evidence_store"] = EvidenceStore

__all__ = [
    "Severity",
    "Finding",
    "EvidenceStore",
    "SECURITY_REGISTRY",
]
