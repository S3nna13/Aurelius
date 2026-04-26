"""Tests for security_report_builder."""

from __future__ import annotations

import json

from src.safety.security_report_builder import (
    SECURITY_REPORT_BUILDER_REGISTRY,
    ReportFormat,
    ReportSection,
    SecurityReportBuilder,
)
from src.safety.vulnerability_schema import (
    VulnCollection,
    VulnFinding,
    VulnSeverity,
)


def _collection():
    c = VulnCollection(target="app")
    c.add(
        VulnFinding(
            title="SQLi",
            severity=VulnSeverity.CRITICAL,
            description="dbad",
            recommendation="escape",
        )
    )
    c.add(VulnFinding(title="XSS", severity=VulnSeverity.HIGH, description="reflected"))
    c.add(VulnFinding(title="Info leak", severity=VulnSeverity.LOW))
    c.add(VulnFinding(title="Debug banner", severity=VulnSeverity.INFO))
    return c


def test_report_format_values():
    assert ReportFormat.MARKDOWN.value == "markdown"
    assert ReportFormat.HTML.value == "html"
    assert ReportFormat.TEXT.value == "text"
    assert ReportFormat.JSON.value == "json"


def test_report_section_is_frozen():
    s = ReportSection(title="t", content="c")
    try:
        s.title = "x"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ReportSection should be frozen")


def test_executive_summary_title():
    s = SecurityReportBuilder().build_executive_summary(_collection())
    assert s.title == "Executive Summary"


def test_executive_summary_total():
    s = SecurityReportBuilder().build_executive_summary(_collection())
    assert "Total findings: 4" in s.content


def test_executive_summary_ch_count():
    s = SecurityReportBuilder().build_executive_summary(_collection())
    assert "Critical + High: 2" in s.content


def test_executive_summary_top_3():
    s = SecurityReportBuilder().build_executive_summary(_collection())
    # Top 3 lines prefixed with '- '
    top_lines = [ln for ln in s.content.splitlines() if ln.startswith("- ")]
    assert len(top_lines) == 3


def test_executive_summary_ordering_critical_first():
    s = SecurityReportBuilder().build_executive_summary(_collection())
    top = [ln for ln in s.content.splitlines() if ln.startswith("- ")]
    assert "CRITICAL" in top[0]


def test_executive_summary_severity_counts():
    s = SecurityReportBuilder().build_executive_summary(_collection())
    assert s.severity_counts["critical"] == 1
    assert s.severity_counts["high"] == 1


def test_findings_section_title():
    s = SecurityReportBuilder().build_findings_section(_collection())
    assert s.title == "Findings"


def test_findings_section_groups_by_severity():
    s = SecurityReportBuilder().build_findings_section(_collection())
    assert "CRITICAL" in s.content
    assert "HIGH" in s.content
    assert "LOW" in s.content


def test_findings_section_includes_recommendation():
    s = SecurityReportBuilder().build_findings_section(_collection())
    assert "Recommendation: escape" in s.content


def test_findings_section_includes_description():
    s = SecurityReportBuilder().build_findings_section(_collection())
    assert "Description: dbad" in s.content


def test_findings_section_empty():
    s = SecurityReportBuilder().build_findings_section(VulnCollection())
    assert s.content == "No findings."


def test_to_markdown_has_header():
    b = SecurityReportBuilder()
    sections = [b.build_executive_summary(_collection())]
    md = b.to_markdown(sections)
    assert "## Executive Summary" in md


def test_to_markdown_bullets():
    b = SecurityReportBuilder()
    md = b.to_markdown([b.build_executive_summary(_collection())])
    assert "* " in md


def test_to_text_has_separator():
    b = SecurityReportBuilder()
    txt = b.to_text([b.build_executive_summary(_collection())])
    assert "===" in txt


def test_to_text_has_title():
    b = SecurityReportBuilder()
    txt = b.to_text([b.build_executive_summary(_collection())])
    assert "Executive Summary" in txt


def test_to_json_is_valid():
    b = SecurityReportBuilder()
    out = b.to_json([b.build_executive_summary(_collection())])
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["title"] == "Executive Summary"


def test_to_json_multiple_sections():
    b = SecurityReportBuilder()
    c = _collection()
    out = b.to_json([b.build_executive_summary(c), b.build_findings_section(c)])
    assert len(json.loads(out)) == 2


def test_to_html_has_h2():
    b = SecurityReportBuilder()
    html = b.to_html([b.build_executive_summary(_collection())])
    assert "<h2>" in html


def test_to_html_has_body():
    b = SecurityReportBuilder()
    html = b.to_html([b.build_executive_summary(_collection())])
    assert "<html>" in html and "</html>" in html


def test_to_html_has_ul():
    b = SecurityReportBuilder()
    html = b.to_html([b.build_executive_summary(_collection())])
    assert "<ul>" in html


def test_generate_markdown():
    out = SecurityReportBuilder().generate(_collection(), ReportFormat.MARKDOWN)
    assert "## Executive Summary" in out


def test_generate_text():
    out = SecurityReportBuilder().generate(_collection(), ReportFormat.TEXT)
    assert "===" in out


def test_generate_json():
    out = SecurityReportBuilder().generate(_collection(), ReportFormat.JSON)
    json.loads(out)


def test_generate_html():
    out = SecurityReportBuilder().generate(_collection(), ReportFormat.HTML)
    assert "<h2>" in out


def test_generate_default_is_markdown():
    out = SecurityReportBuilder().generate(_collection())
    assert "## " in out


def test_generate_empty_collection_markdown():
    out = SecurityReportBuilder().generate(VulnCollection(), ReportFormat.MARKDOWN)
    assert "Total findings: 0" in out


def test_generate_empty_collection_all_formats():
    b = SecurityReportBuilder()
    c = VulnCollection()
    for fmt in ReportFormat:
        assert b.generate(c, fmt)


def test_registry_default():
    assert SECURITY_REPORT_BUILDER_REGISTRY["default"] is SecurityReportBuilder
