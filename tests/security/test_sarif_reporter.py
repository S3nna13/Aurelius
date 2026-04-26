"""Tests for src.security.sarif_reporter."""

from __future__ import annotations

import json

import pytest

from src.security.sarif_reporter import (
    SARIF_REPORTER_REGISTRY,
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    SarifLevel,
    SarifReport,
    SarifResult,
    SarifRule,
)


def test_empty_report_valid_sarif():
    report = SarifReport()
    doc = report.to_dict()
    assert doc["version"] == "2.1.0"
    assert doc["$schema"] == SARIF_SCHEMA_URI
    assert isinstance(doc["runs"], list) and len(doc["runs"]) == 1
    run = doc["runs"][0]
    assert run["tool"]["driver"]["name"] == "Aurelius"
    assert run["tool"]["driver"]["rules"] == []
    assert run["results"] == []


def test_add_rule_and_result():
    report = SarifReport()
    rule = SarifRule(
        rule_id="AUR-001",
        short_description="Unauthorized policy action",
        full_description="A policy gate denial occurred.",
        help_text="See docs/security/policy_gate.md",
        tags=["security", "policy"],
    )
    report.add_rule(rule)
    result = SarifResult(
        rule_id="AUR-001",
        message="denied action on target",
        level=SarifLevel.ERROR,
        uri="src/demo.py",
        start_line=10,
        end_line=12,
    )
    report.add_result(result)

    doc = report.to_dict()
    run = doc["runs"][0]
    assert len(run["tool"]["driver"]["rules"]) == 1
    assert run["tool"]["driver"]["rules"][0]["id"] == "AUR-001"
    assert run["tool"]["driver"]["rules"][0]["properties"]["tags"] == [
        "security",
        "policy",
    ]
    assert len(run["results"]) == 1
    r0 = run["results"][0]
    assert r0["ruleId"] == "AUR-001"
    assert r0["level"] == "error"
    assert r0["message"]["text"] == "denied action on target"
    loc = r0["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "src/demo.py"
    assert loc["region"]["startLine"] == 10
    assert loc["region"]["endLine"] == 12


def test_to_json_is_valid_json():
    report = SarifReport()
    report.add_rule(SarifRule(rule_id="AUR-002", short_description="x"))
    report.add_result(SarifResult(rule_id="AUR-002", message="hi", level=SarifLevel.NOTE))
    payload = report.to_json()
    parsed = json.loads(payload)
    assert parsed["version"] == SARIF_VERSION
    assert parsed["runs"][0]["results"][0]["level"] == "note"


def test_sarif_version_field():
    report = SarifReport()
    assert report.to_dict()["version"] == "2.1.0"


def test_result_level_mapping():
    SarifReport()
    for level, expected in (
        (SarifLevel.NOTE, "note"),
        (SarifLevel.WARNING, "warning"),
        (SarifLevel.ERROR, "error"),
    ):
        result = SarifResult(rule_id="R", message="m", level=level)
        assert result.to_dict()["level"] == expected


def test_result_accepts_plain_string_level():
    result = SarifResult(rule_id="R", message="m", level="warning")  # type: ignore[arg-type]
    assert result.level is SarifLevel.WARNING


def test_result_rejects_invalid_string_level():
    with pytest.raises(ValueError):
        SarifResult(rule_id="R", message="m", level="critical")  # type: ignore[arg-type]


def test_result_rejects_bad_line_numbers():
    with pytest.raises(ValueError):
        SarifResult(rule_id="R", message="m", uri="a.py", start_line=0)
    with pytest.raises(ValueError):
        SarifResult(rule_id="R", message="m", uri="a.py", end_line=-1)


def test_rule_rejects_empty_id():
    with pytest.raises(ValueError):
        SarifRule(rule_id="", short_description="x")


def test_add_rule_is_idempotent():
    report = SarifReport()
    rule = SarifRule(rule_id="AUR-003", short_description="x")
    report.add_rule(rule)
    report.add_rule(rule)
    assert len(report.rules) == 1


def test_add_rule_type_check():
    report = SarifReport()
    with pytest.raises(TypeError):
        report.add_rule("nope")  # type: ignore[arg-type]


def test_add_result_type_check():
    report = SarifReport()
    with pytest.raises(TypeError):
        report.add_result({"ruleId": "x"})  # type: ignore[arg-type]


def test_default_registry_present():
    assert "default" in SARIF_REPORTER_REGISTRY
    assert isinstance(SARIF_REPORTER_REGISTRY["default"], SarifReport)


def test_result_without_uri_has_no_locations():
    result = SarifResult(rule_id="R", message="m")
    assert "locations" not in result.to_dict()
