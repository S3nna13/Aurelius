"""Tests for src.safety.threat_intel."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.safety.threat_intel import (  # noqa: E402
    THREAT_INTEL_REGISTRY,
    ThreatEntry,
    ThreatIntelEngine,
    ThreatLevel,
)


def _entry(
    vuln_id: str,
    epss: float,
    cvss: float,
    level: ThreatLevel = ThreatLevel.MEDIUM,
    tags=None,
    notes: str = "",
) -> ThreatEntry:
    return ThreatEntry(
        vuln_id=vuln_id,
        epss_score=epss,
        cvss_score=cvss,
        threat_level=level,
        tags=list(tags or []),
        notes=notes,
    )


def test_threat_level_values():
    assert ThreatLevel.CRITICAL.value == "critical"
    assert ThreatLevel.HIGH.value == "high"
    assert ThreatLevel.MEDIUM.value == "medium"
    assert ThreatLevel.LOW.value == "low"
    assert ThreatLevel.INFO.value == "info"


def test_threat_entry_is_frozen():
    e = _entry("CVE-1", 0.1, 5.0)
    try:
        e.notes = "x"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen")


def test_threat_entry_default_notes():
    e = _entry("CVE-1", 0.1, 5.0)
    assert e.notes == ""


def test_add_and_get():
    eng = ThreatIntelEngine()
    e = _entry("CVE-1", 0.5, 7.0)
    eng.add(e)
    assert eng.get("CVE-1") is e


def test_get_unknown_returns_none():
    assert ThreatIntelEngine().get("CVE-missing") is None


def test_add_replaces():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-1", 0.1, 1.0))
    eng.add(_entry("CVE-1", 0.9, 9.0))
    got = eng.get("CVE-1")
    assert got is not None and got.epss_score == 0.9


def test_score_formula():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-1", 1.0, 10.0))
    # 0.6 * 1.0 + 0.4 * 1.0 == 1.0
    assert abs(eng.score("CVE-1") - 1.0) < 1e-9


def test_score_mixed():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-2", 0.5, 5.0))
    # 0.6 * 0.5 + 0.4 * 0.5 == 0.5
    assert abs(eng.score("CVE-2") - 0.5) < 1e-9


def test_score_zero_values():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-3", 0.0, 0.0))
    assert eng.score("CVE-3") == 0.0


def test_score_unknown_zero():
    assert ThreatIntelEngine().score("CVE-ghost") == 0.0


def test_score_clamps_high_cvss():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-4", 0.0, 999.0))
    # cvss clamped to 10 -> 0.4
    assert abs(eng.score("CVE-4") - 0.4) < 1e-9


def test_score_clamps_high_epss():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-5", 2.0, 0.0))
    # epss clamped to 1.0 -> 0.6
    assert abs(eng.score("CVE-5") - 0.6) < 1e-9


def test_prioritize_sorts_desc():
    eng = ThreatIntelEngine()
    eng.add(_entry("A", 0.1, 1.0))
    eng.add(_entry("B", 0.9, 9.0))
    eng.add(_entry("C", 0.5, 5.0))
    result = eng.prioritize(["A", "B", "C"])
    assert [e.vuln_id for e in result] == ["B", "C", "A"]


def test_prioritize_skips_unknown():
    eng = ThreatIntelEngine()
    eng.add(_entry("A", 0.5, 5.0))
    result = eng.prioritize(["A", "MISSING"])
    assert [e.vuln_id for e in result] == ["A"]


def test_prioritize_empty():
    assert ThreatIntelEngine().prioritize([]) == []


def test_prioritize_no_known():
    eng = ThreatIntelEngine()
    eng.add(_entry("A", 0.1, 1.0))
    assert eng.prioritize(["X", "Y"]) == []


def test_high_risk_ids_above_threshold():
    eng = ThreatIntelEngine()
    eng.add(_entry("A", 0.1, 1.0))  # score 0.10
    eng.add(_entry("B", 0.9, 9.0))  # score 0.90
    eng.add(_entry("C", 0.8, 8.0))  # score 0.80
    ids = eng.high_risk_ids(threshold=0.7)
    assert set(ids) == {"B", "C"}


def test_high_risk_ids_default_threshold():
    eng = ThreatIntelEngine()
    eng.add(_entry("A", 1.0, 10.0))
    assert "A" in eng.high_risk_ids()


def test_high_risk_ids_empty():
    assert ThreatIntelEngine().high_risk_ids() == []


def test_high_risk_ids_sorted_desc():
    eng = ThreatIntelEngine()
    eng.add(_entry("B", 0.9, 9.0))
    eng.add(_entry("A", 0.95, 9.5))
    ids = eng.high_risk_ids(0.5)
    assert ids == ["A", "B"]


def test_high_risk_threshold_strict():
    eng = ThreatIntelEngine()
    eng.add(_entry("A", 0.2, 2.0))
    assert eng.high_risk_ids(0.9) == []


def test_to_dict_structure():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-1", 0.5, 5.0, ThreatLevel.HIGH, tags=["web"], notes="n"))
    d = eng.to_dict()
    assert "entries" in d and len(d["entries"]) == 1
    row = d["entries"][0]
    assert row["vuln_id"] == "CVE-1"
    assert row["threat_level"] == "high"
    assert row["tags"] == ["web"]
    assert row["notes"] == "n"


def test_from_dict_roundtrip():
    eng = ThreatIntelEngine()
    eng.add(_entry("CVE-1", 0.5, 5.0, ThreatLevel.HIGH, tags=["web"]))
    eng.add(_entry("CVE-2", 0.1, 1.0, ThreatLevel.LOW))
    data = eng.to_dict()
    rebuilt = ThreatIntelEngine.from_dict(data)
    assert rebuilt.get("CVE-1") is not None
    assert rebuilt.get("CVE-2") is not None
    assert rebuilt.get("CVE-1").threat_level == ThreatLevel.HIGH


def test_from_dict_empty():
    eng = ThreatIntelEngine.from_dict({})
    assert eng.high_risk_ids(0.0) == []


def test_from_dict_missing_entries_key():
    eng = ThreatIntelEngine.from_dict({"entries": None})
    assert eng.to_dict() == {"entries": []}


def test_from_dict_invalid_level_defaults_info():
    eng = ThreatIntelEngine.from_dict(
        {
            "entries": [
                {
                    "vuln_id": "X",
                    "epss_score": 0.1,
                    "cvss_score": 1.0,
                    "threat_level": "bogus",
                    "tags": [],
                }
            ],
        }
    )
    got = eng.get("X")
    assert got is not None
    assert got.threat_level == ThreatLevel.INFO


def test_from_dict_skips_missing_vuln_id():
    eng = ThreatIntelEngine.from_dict(
        {
            "entries": [
                {
                    "vuln_id": "",
                    "epss_score": 0.1,
                    "cvss_score": 1.0,
                    "threat_level": "low",
                    "tags": [],
                }
            ],
        }
    )
    assert eng.to_dict() == {"entries": []}


def test_registry_default_is_class():
    assert THREAT_INTEL_REGISTRY["default"] is ThreatIntelEngine


def test_registry_instantiates():
    assert isinstance(THREAT_INTEL_REGISTRY["default"](), ThreatIntelEngine)


def test_score_ordering_consistent_with_prioritize():
    eng = ThreatIntelEngine()
    eng.add(_entry("A", 0.2, 2.0))
    eng.add(_entry("B", 0.8, 8.0))
    s_a = eng.score("A")
    s_b = eng.score("B")
    result = eng.prioritize(["A", "B"])
    assert s_b > s_a
    assert result[0].vuln_id == "B"
