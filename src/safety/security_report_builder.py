"""Build structured security reports in multiple output formats."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from .vulnerability_schema import VulnCollection, VulnFinding, VulnSeverity


class ReportFormat(StrEnum):
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


@dataclass(frozen=True)
class ReportSection:
    title: str
    content: str
    severity_counts: dict = field(default_factory=dict)


_SEVERITY_ORDER = [
    VulnSeverity.CRITICAL,
    VulnSeverity.HIGH,
    VulnSeverity.MEDIUM,
    VulnSeverity.LOW,
    VulnSeverity.INFO,
]


class SecurityReportBuilder:
    def build_executive_summary(self, collection: VulnCollection) -> ReportSection:
        stats = collection.stats()
        total = stats.get("total", 0)
        ch = len(collection.critical_and_high())
        lines = [
            f"Total findings: {total}",
            f"Critical + High: {ch}",
            f"Scan ID: {collection.scan_id}",
            f"Target: {collection.target or 'N/A'}",
        ]

        def _rank(f: VulnFinding) -> int:
            try:
                return _SEVERITY_ORDER.index(f.severity)
            except ValueError:
                return len(_SEVERITY_ORDER)

        top = sorted(collection.findings, key=_rank)[:3]
        if top:
            lines.append("Top findings:")
            for f in top:
                lines.append(f"- [{f.severity.value.upper()}] {f.title}")
        content = "\n".join(lines)
        severity_counts = {k: v for k, v in stats.items() if k != "total"}
        return ReportSection(
            title="Executive Summary",
            content=content,
            severity_counts=severity_counts,
        )

    def build_findings_section(self, collection: VulnCollection) -> ReportSection:
        chunks: list[str] = []
        severity_counts: dict = {}
        for sev in _SEVERITY_ORDER:
            group = collection.by_severity(sev)
            severity_counts[sev.value] = len(group)
            if not group:
                continue
            chunks.append(f"Severity: {sev.value.upper()} ({len(group)})")
            for f in group:
                chunks.append(f"- {f.title}")
                if f.description:
                    chunks.append(f"  Description: {f.description}")
                if f.recommendation:
                    chunks.append(f"  Recommendation: {f.recommendation}")
        content = "\n".join(chunks) if chunks else "No findings."
        return ReportSection(
            title="Findings",
            content=content,
            severity_counts=severity_counts,
        )

    def to_markdown(self, sections: list[ReportSection]) -> str:
        out: list[str] = []
        for s in sections:
            out.append(f"## {s.title}")
            for line in s.content.splitlines():
                if line.startswith("- "):
                    out.append(f"* {line[2:]}")
                else:
                    out.append(line)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def to_text(self, sections: list[ReportSection]) -> str:
        out: list[str] = []
        for s in sections:
            out.append("=" * 60)
            out.append(s.title)
            out.append("=" * 60)
            out.append(s.content)
            out.append("")
        return "\n".join(out)

    def to_json(self, sections: list[ReportSection]) -> str:
        payload = [{"title": s.title, "content": s.content} for s in sections]
        return json.dumps(payload, indent=2)

    def to_html(self, sections: list[ReportSection]) -> str:
        out: list[str] = ["<html><body>"]
        for s in sections:
            out.append(f"<h2>{s.title}</h2>")
            bullets = [line[2:] for line in s.content.splitlines() if line.startswith("- ")]
            non_bullets = [line for line in s.content.splitlines() if not line.startswith("- ")]
            if non_bullets:
                out.append("<p>" + "<br/>".join(non_bullets) + "</p>")
            if bullets:
                out.append("<ul>")
                for b in bullets:
                    out.append(f"<li>{b}</li>")
                out.append("</ul>")
        out.append("</body></html>")
        return "\n".join(out)

    def generate(
        self,
        collection: VulnCollection,
        fmt: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        sections = [
            self.build_executive_summary(collection),
            self.build_findings_section(collection),
        ]
        if fmt == ReportFormat.MARKDOWN:
            return self.to_markdown(sections)
        if fmt == ReportFormat.TEXT:
            return self.to_text(sections)
        if fmt == ReportFormat.JSON:
            return self.to_json(sections)
        if fmt == ReportFormat.HTML:
            return self.to_html(sections)
        raise ValueError(f"Unsupported report format: {fmt}")


SECURITY_REPORT_BUILDER_REGISTRY = {"default": SecurityReportBuilder}
