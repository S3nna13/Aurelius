"""Tests for sarif_exporter."""

from __future__ import annotations

import json
import os
import tempfile

from src.safety.sarif_exporter import (
    SARIF_EXPORTER_REGISTRY,
    SARIFExporter,
    SARIFLevel,
    _severity_to_sarif_level,
)
from src.safety.vulnerability_schema import (
    VulnCollection,
    VulnFinding,
    VulnSeverity,
)


def _collection(*findings):
    c = VulnCollection(target="t")
    for f in findings:
        c.add(f)
    return c


def test_sarif_level_enum():
    assert SARIFLevel.ERROR.value == "error"
    assert SARIFLevel.WARNING.value == "warning"
    assert SARIFLevel.NOTE.value == "note"


def test_severity_mapping_critical():
    assert _severity_to_sarif_level(VulnSeverity.CRITICAL) == "error"


def test_severity_mapping_high():
    assert _severity_to_sarif_level(VulnSeverity.HIGH) == "error"


def test_severity_mapping_medium():
    assert _severity_to_sarif_level(VulnSeverity.MEDIUM) == "warning"


def test_severity_mapping_low():
    assert _severity_to_sarif_level(VulnSeverity.LOW) == "note"


def test_severity_mapping_info():
    assert _severity_to_sarif_level(VulnSeverity.INFO) == "note"


def test_export_version():
    s = SARIFExporter().export(VulnCollection())
    assert s["version"] == "2.1.0"


def test_export_schema_present():
    s = SARIFExporter().export(VulnCollection())
    assert "$schema" in s


def test_export_has_runs():
    s = SARIFExporter().export(VulnCollection())
    assert isinstance(s["runs"], list) and len(s["runs"]) == 1


def test_export_tool_name():
    s = SARIFExporter().export(VulnCollection())
    assert s["runs"][0]["tool"]["driver"]["name"] == "Aurelius"


def test_export_tool_version():
    s = SARIFExporter().export(VulnCollection())
    assert s["runs"][0]["tool"]["driver"]["version"] == "1.0"


def test_build_rule_id():
    f = VulnFinding(title="t")
    r = SARIFExporter().build_rule(f)
    assert r["id"] == f.finding_id


def test_build_rule_short_description():
    f = VulnFinding(title="x" * 600)
    r = SARIFExporter().build_rule(f)
    assert len(r["shortDescription"]["text"]) == 500


def test_build_rule_full_description_truncated():
    f = VulnFinding(description="y" * 3000)
    r = SARIFExporter().build_rule(f)
    assert len(r["fullDescription"]["text"]) == 2000


def test_build_rule_cwe_tags():
    f = VulnFinding(cwe_ids=["CWE-79"])
    r = SARIFExporter().build_rule(f)
    assert r["properties"]["tags"] == ["CWE-79"]


def test_build_rule_no_cwe_no_props():
    r = SARIFExporter().build_rule(VulnFinding())
    assert "properties" not in r


def test_build_result_rule_id():
    f = VulnFinding(title="a")
    r = SARIFExporter().build_result(f)
    assert r["ruleId"] == f.finding_id


def test_build_result_level_critical():
    r = SARIFExporter().build_result(VulnFinding(severity=VulnSeverity.CRITICAL))
    assert r["level"] == "error"


def test_build_result_message_contains_title():
    r = SARIFExporter().build_result(VulnFinding(title="SQLi", description="d"))
    assert "SQLi" in r["message"]["text"]


def test_build_result_locations_when_file():
    r = SARIFExporter().build_result(VulnFinding(affected_file="x.py", affected_line=10))
    loc = r["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "x.py"
    assert loc["region"]["startLine"] == 10


def test_build_result_no_locations_when_no_file():
    r = SARIFExporter().build_result(VulnFinding())
    assert "locations" not in r


def test_build_result_no_region_when_line_zero():
    r = SARIFExporter().build_result(VulnFinding(affected_file="x.py"))
    loc = r["locations"][0]["physicalLocation"]
    assert "region" not in loc


def test_results_count_matches_findings():
    c = _collection(VulnFinding(title="a"), VulnFinding(title="b"), VulnFinding(title="c"))
    s = SARIFExporter().export(c)
    assert len(s["runs"][0]["results"]) == 3


def test_rules_deduplicated():
    f = VulnFinding(title="dup")
    c = VulnCollection()
    c.add(f)
    c.add(f)  # same finding_id
    s = SARIFExporter().export(c)
    assert len(s["runs"][0]["tool"]["driver"]["rules"]) == 1
    assert len(s["runs"][0]["results"]) == 2


def test_rules_unique_ids():
    c = _collection(VulnFinding(title="a"), VulnFinding(title="b"))
    s = SARIFExporter().export(c)
    ids = [r["id"] for r in s["runs"][0]["tool"]["driver"]["rules"]]
    assert len(set(ids)) == len(ids)


def test_export_json_is_valid_json():
    c = _collection(VulnFinding(title="x"))
    txt = SARIFExporter().export_json(c)
    parsed = json.loads(txt)
    assert parsed["version"] == "2.1.0"


def test_export_json_indent():
    txt = SARIFExporter().export_json(VulnCollection())
    assert "\n" in txt


def test_export_to_file(tmp_path=None):
    path = os.path.join(tempfile.mkdtemp(), "out.sarif")
    c = _collection(VulnFinding(title="x"))
    SARIFExporter().export_to_file(c, path)
    with open(path) as fh:
        data = json.load(fh)
    assert data["version"] == "2.1.0"


def test_finding_count():
    c = _collection(VulnFinding(), VulnFinding())
    exp = SARIFExporter()
    assert exp.finding_count(exp.export(c)) == 2


def test_finding_count_empty():
    exp = SARIFExporter()
    assert exp.finding_count(exp.export(VulnCollection())) == 0


def test_finding_count_handles_missing_runs():
    assert SARIFExporter().finding_count({}) == 0


def test_registry_default():
    assert SARIF_EXPORTER_REGISTRY["default"] is SARIFExporter
