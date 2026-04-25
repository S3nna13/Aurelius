"""Tests for src/security/evidence_store.py — 12+ cases."""

import time
import pytest

from src.security.evidence_store import (
    EvidenceStore,
    Finding,
    _SEV_ORDER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_finding(
    tool: str = "scanner",
    severity: str = "medium",
    title: str = "Test Finding",
    url: str = "https://example.com",
    details: str = "details",
    poc: str = "poc",
) -> Finding:
    return Finding(tool=tool, severity=severity, title=title,
                   url=url, details=details, poc=poc)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_single_finding(self):
        store = EvidenceStore()
        f = make_finding()
        store.upsert_finding(f)
        assert store.finding_count() == 1

    def test_upsert_duplicate_replaces(self):
        store = EvidenceStore()
        f = make_finding(title="Original")
        store.upsert_finding(f)
        f.title = "Updated"
        store.upsert_finding(f)
        # Same finding_id → count stays 1
        assert store.finding_count() == 1
        findings = store.list_findings()
        assert findings[0].title == "Updated"

    def test_upsert_multiple_distinct(self):
        store = EvidenceStore()
        store.upsert_finding(make_finding(title="A"))
        store.upsert_finding(make_finding(title="B"))
        assert store.finding_count() == 2

    def test_upsert_preserves_fields(self):
        store = EvidenceStore()
        f = make_finding(tool="mytool", severity="high",
                         title="XSS", url="https://t.com",
                         details="payload", poc="<script>")
        store.upsert_finding(f)
        result = store.list_findings()[0]
        assert result.tool == "mytool"
        assert result.severity == "high"
        assert result.title == "XSS"
        assert result.url == "https://t.com"
        assert result.details == "payload"
        assert result.poc == "<script>"


# ---------------------------------------------------------------------------
# list_findings with severity filter
# ---------------------------------------------------------------------------

class TestListFindings:
    def test_list_empty_store(self):
        store = EvidenceStore()
        assert store.list_findings() == []

    def test_list_all_severities(self):
        store = EvidenceStore()
        for sev in ("info", "low", "medium", "high", "critical"):
            store.upsert_finding(make_finding(severity=sev, title=sev))
        assert len(store.list_findings()) == 5

    def test_filter_min_medium(self):
        store = EvidenceStore()
        for sev in ("info", "low", "medium", "high", "critical"):
            store.upsert_finding(make_finding(severity=sev, title=sev))
        results = store.list_findings(min_severity="medium")
        severities = {f.severity for f in results}
        assert "info" not in severities
        assert "low" not in severities
        assert "medium" in severities
        assert "high" in severities
        assert "critical" in severities

    def test_filter_min_critical(self):
        store = EvidenceStore()
        for sev in ("info", "low", "medium", "high", "critical"):
            store.upsert_finding(make_finding(severity=sev, title=sev))
        results = store.list_findings(min_severity="critical")
        assert len(results) == 1
        assert results[0].severity == "critical"

    def test_filter_min_info_returns_all(self):
        store = EvidenceStore()
        for sev in ("info", "low", "high"):
            store.upsert_finding(make_finding(severity=sev, title=sev))
        assert len(store.list_findings(min_severity="info")) == 3

    def test_list_ordered_by_created_at(self):
        store = EvidenceStore()
        f1 = Finding(tool="t", severity="low", title="first",
                     created_at=1000.0)
        f2 = Finding(tool="t", severity="low", title="second",
                     created_at=2000.0)
        store.upsert_finding(f2)
        store.upsert_finding(f1)
        results = store.list_findings()
        assert results[0].title == "first"
        assert results[1].title == "second"


# ---------------------------------------------------------------------------
# finding_count
# ---------------------------------------------------------------------------

class TestFindingCount:
    def test_count_empty(self):
        store = EvidenceStore()
        assert store.finding_count() == 0

    def test_count_after_inserts(self):
        store = EvidenceStore()
        for i in range(5):
            store.upsert_finding(make_finding(title=f"f{i}"))
        assert store.finding_count() == 5


# ---------------------------------------------------------------------------
# SARIF export
# ---------------------------------------------------------------------------

class TestExportSarif:
    def test_sarif_version(self):
        store = EvidenceStore()
        doc = store.export_sarif()
        assert doc["version"] == "2.1.0"

    def test_sarif_schema_present(self):
        store = EvidenceStore()
        doc = store.export_sarif()
        assert "$schema" in doc
        assert "sarif" in doc["$schema"].lower() or "oasis" in doc["$schema"].lower()

    def test_sarif_tool_name(self):
        store = EvidenceStore()
        doc = store.export_sarif(tool_name="my-scanner")
        driver = doc["runs"][0]["tool"]["driver"]
        assert driver["name"] == "my-scanner"

    def test_sarif_default_tool_name(self):
        store = EvidenceStore()
        doc = store.export_sarif()
        driver = doc["runs"][0]["tool"]["driver"]
        assert driver["name"] == "aurelius-security"

    def test_sarif_empty_results(self):
        store = EvidenceStore()
        doc = store.export_sarif()
        assert doc["runs"][0]["results"] == []

    def test_sarif_result_count(self):
        store = EvidenceStore()
        store.upsert_finding(make_finding(title="A"))
        store.upsert_finding(make_finding(title="B"))
        doc = store.export_sarif()
        assert len(doc["runs"][0]["results"]) == 2

    def test_sarif_severity_mapping_high(self):
        store = EvidenceStore()
        store.upsert_finding(make_finding(severity="high", title="H"))
        doc = store.export_sarif()
        assert doc["runs"][0]["results"][0]["level"] == "error"

    def test_sarif_severity_mapping_info(self):
        store = EvidenceStore()
        store.upsert_finding(make_finding(severity="info", title="I"))
        doc = store.export_sarif()
        assert doc["runs"][0]["results"][0]["level"] == "note"

    def test_sarif_result_structure(self):
        store = EvidenceStore()
        f = make_finding(url="https://target.com/path", title="SQLi")
        store.upsert_finding(f)
        result = store.export_sarif()["runs"][0]["results"][0]
        assert "ruleId" in result
        assert "level" in result
        assert result["message"]["text"] == "SQLi"
        assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "https://target.com/path"

    def test_sarif_empty_url_falls_back_to_unknown(self):
        store = EvidenceStore()
        f = Finding(tool="t", severity="low", title="NoURL", url="")
        store.upsert_finding(f)
        result = store.export_sarif()["runs"][0]["results"][0]
        assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "unknown"


# ---------------------------------------------------------------------------
# Severity ordering sanity
# ---------------------------------------------------------------------------

class TestSeverityOrder:
    def test_order_values(self):
        assert _SEV_ORDER["info"] < _SEV_ORDER["low"]
        assert _SEV_ORDER["low"] < _SEV_ORDER["medium"]
        assert _SEV_ORDER["medium"] < _SEV_ORDER["high"]
        assert _SEV_ORDER["high"] < _SEV_ORDER["critical"]


# ---------------------------------------------------------------------------
# Finding dataclass defaults
# ---------------------------------------------------------------------------

class TestFindingDefaults:
    def test_finding_id_auto_generated(self):
        f = Finding(tool="t", severity="low", title="x")
        assert f.finding_id and len(f.finding_id) == 36  # UUID4 length

    def test_finding_ids_unique(self):
        f1 = Finding(tool="t", severity="low", title="x")
        f2 = Finding(tool="t", severity="low", title="x")
        assert f1.finding_id != f2.finding_id

    def test_finding_created_at_close_to_now(self):
        before = time.time()
        f = Finding(tool="t", severity="low", title="x")
        after = time.time()
        assert before <= f.created_at <= after
