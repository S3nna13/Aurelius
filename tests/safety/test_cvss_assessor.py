"""Tests for src.safety.cvss_assessor."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.safety.cvss_assessor import (  # noqa: E402
    CVSS_ASSESSOR_REGISTRY,
    CVSSAssessor,
    CVSSMetric,
    CVSSVersion,
    _SEVERITY_VECTORS,
    _STATIC_SCORES,
)


def test_cvss_version_enum_values():
    assert CVSSVersion.V31.value == "3.1"
    assert CVSSVersion.V40.value == "4.0"


def test_cvss_metric_is_frozen():
    m = CVSSMetric(vector="v", base_score=1.0, severity="low")
    try:
        m.base_score = 5.0  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen dataclass")


def test_cvss_metric_default_is_draft_false():
    m = CVSSMetric(vector="v", base_score=1.0, severity="low")
    assert m.is_draft is False


def test_severity_to_vector_critical_includes_draft():
    v = CVSSAssessor().severity_to_vector("critical")
    assert "CVSS:3.1" in v
    assert "DRAFT" in v


def test_severity_to_vector_high_includes_draft():
    assert "DRAFT" in CVSSAssessor().severity_to_vector("high")


def test_severity_to_vector_medium_includes_draft():
    assert "DRAFT" in CVSSAssessor().severity_to_vector("medium")


def test_severity_to_vector_low_includes_draft():
    assert "DRAFT" in CVSSAssessor().severity_to_vector("low")


def test_severity_to_vector_info_includes_draft():
    assert "DRAFT" in CVSSAssessor().severity_to_vector("info")


def test_severity_to_vector_case_insensitive():
    a = CVSSAssessor().severity_to_vector("Critical")
    b = CVSSAssessor().severity_to_vector("critical")
    assert a == b


def test_severity_to_vector_unknown_falls_back():
    v = CVSSAssessor().severity_to_vector("mystery")
    assert "DRAFT" in v


def test_severity_to_score_values():
    a = CVSSAssessor()
    assert a.severity_to_score("critical") == 9.8
    assert a.severity_to_score("high") == 7.1
    assert a.severity_to_score("medium") == 4.2
    assert a.severity_to_score("low") == 2.4
    assert a.severity_to_score("info") == 0.0


def test_severity_to_score_unknown_zero():
    assert CVSSAssessor().severity_to_score("bogus") == 0.0


def test_severity_to_score_case_insensitive():
    assert CVSSAssessor().severity_to_score("HIGH") == 7.1


def test_parse_vector_extracts_keys():
    parsed = CVSSAssessor().parse_vector(
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    )
    assert parsed["AV"] == "N"
    assert parsed["AC"] == "L"
    assert parsed["C"] == "H"
    assert parsed["I"] == "H"
    assert parsed["A"] == "H"


def test_parse_vector_handles_draft_suffix():
    a = CVSSAssessor()
    raw = a.severity_to_vector("critical")
    parsed = a.parse_vector(raw)
    assert parsed.get("C") == "H"
    assert parsed.get("A") == "H"


def test_parse_vector_empty_string():
    assert CVSSAssessor().parse_vector("") == {}


def test_parse_vector_non_cvss_returns_empty():
    assert CVSSAssessor().parse_vector("not-a-vector") == {}


def test_parse_vector_v40_supported():
    parsed = CVSSAssessor().parse_vector("CVSS:4.0/AV:N/AC:L")
    assert parsed.get("AV") == "N"


def test_vector_to_metric_critical_fields():
    a = CVSSAssessor()
    m = a.vector_to_metric(_SEVERITY_VECTORS["critical"])
    assert m.severity == "critical"
    assert m.base_score == 9.8
    assert m.is_draft is False


def test_vector_to_metric_draft_flag():
    a = CVSSAssessor()
    m = a.vector_to_metric(_SEVERITY_VECTORS["high"], is_draft=True)
    assert m.is_draft is True


def test_vector_to_metric_high_severity():
    a = CVSSAssessor()
    m = a.vector_to_metric(_SEVERITY_VECTORS["high"])
    assert m.severity == "high"


def test_vector_to_metric_low_severity():
    a = CVSSAssessor()
    m = a.vector_to_metric(_SEVERITY_VECTORS["low"])
    assert m.severity == "low"


def test_vector_to_metric_info_severity():
    a = CVSSAssessor()
    m = a.vector_to_metric(_SEVERITY_VECTORS["info"])
    assert m.severity == "info"
    assert m.base_score == 0.0


def test_compare_severity_orders():
    a = CVSSAssessor()
    assert a.compare_severity("critical", "high") == 1
    assert a.compare_severity("low", "medium") == -1
    assert a.compare_severity("info", "info") == 0


def test_compare_severity_full_chain():
    a = CVSSAssessor()
    order = ["info", "low", "medium", "high", "critical"]
    for i in range(len(order) - 1):
        assert a.compare_severity(order[i + 1], order[i]) == 1


def test_compare_severity_case_insensitive():
    assert CVSSAssessor().compare_severity("HIGH", "low") == 1


def test_static_scores_all_severities_present():
    for k in ("critical", "high", "medium", "low", "info"):
        assert k in _STATIC_SCORES


def test_severity_vectors_all_severities_present():
    for k in ("critical", "high", "medium", "low", "info"):
        assert k in _SEVERITY_VECTORS


def test_registry_default_is_class():
    assert CVSS_ASSESSOR_REGISTRY["default"] is CVSSAssessor


def test_registry_instantiates():
    inst = CVSS_ASSESSOR_REGISTRY["default"]()
    assert isinstance(inst, CVSSAssessor)
