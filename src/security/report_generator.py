from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FindingEntry:
    title: str
    severity: str
    cwe_id: str = ""
    description: str = ""
    remediation: str = ""
    evidence: str = ""
    cvss_score: float = 0.0


@dataclass
class ReportConfig:
    title: str = "Security Assessment Report"
    author: str = "Aurelius Security"
    company: str = ""
    include_remediation: bool = True


class ReportGenerator:
    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()
        self._findings: list[FindingEntry] = []

    def add_finding(self, finding: FindingEntry) -> None:
        self._findings.append(finding)

    def add_findings(self, findings: list[FindingEntry]) -> None:
        self._findings.extend(findings)

    def summary_stats(self) -> dict[str, int]:
        counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        for f in self._findings:
            sev = f.severity.upper()
            if sev in counts:
                counts[sev] += 1
            else:
                counts["INFO"] += 1
        return counts

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# {self.config.title}")
        lines.append("")
        lines.append(f"**Author:** {self.config.author}")
        lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        if self.config.company:
            lines.append(f"**Company:** {self.config.company}")
        lines.append("")
        stats = self.summary_stats()
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Severity | Count |")
        lines.append(f"|----------|-------|")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
            lines.append(f"| {sev} | {stats[sev]} |")
        lines.append("")
        lines.append(f"**Total Findings:** {len(self._findings)}")
        lines.append("")
        for i, f in enumerate(self._findings, 1):
            lines.append(f"## Finding {i}: {f.title}")
            lines.append("")
            lines.append(f"- **Severity:** {f.severity}")
            if f.cwe_id:
                lines.append(f"- **CWE:** {f.cwe_id}")
            if f.cvss_score:
                lines.append(f"- **CVSS Score:** {f.cvss_score}")
            lines.append("")
            if f.description:
                lines.append(f"{f.description}")
                lines.append("")
            if f.evidence:
                lines.append("### Evidence")
                lines.append("")
                lines.append(f"```\n{f.evidence}\n```")
                lines.append("")
            if self.config.include_remediation and f.remediation:
                lines.append("### Remediation")
                lines.append("")
                lines.append(f"{f.remediation}")
                lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "report": {
                "title": self.config.title,
                "author": self.config.author,
                "date": datetime.now(timezone.utc).isoformat(),
                "company": self.config.company,
            },
            "summary": self.summary_stats(),
            "findings": [
                {
                    "title": f.title,
                    "severity": f.severity,
                    "cwe_id": f.cwe_id,
                    "description": f.description,
                    "remediation": f.remediation,
                    "evidence": f.evidence,
                    "cvss_score": f.cvss_score,
                }
                for f in self._findings
            ],
        }
        return json.dumps(data, indent=2)

    def to_html(self) -> str:
        md = self.to_markdown()
        html_lines: list[str] = [
            "<!DOCTYPE html>",
            '<html><head><meta charset="utf-8">',
            f"<title>{self.config.title}</title>",
            "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:2em}"
            "h1{border-bottom:2px solid #333}"
            ".finding{border:1px solid #ddd;padding:1em;margin:1em 0;border-radius:4px}"
            ".critical{border-left:4px solid #c00}.high{border-left:4px solid #e80}"
            ".medium{border-left:4px solid #ea0}.low{border-left:4px solid #8c0}"
            "</style></head><body>",
            f"<h1>{self.config.title}</h1>",
            f"<p><strong>Author:</strong> {self.config.author}</p>",
        ]
        stats = self.summary_stats()
        html_lines.append("<h2>Summary</h2><table><tr><th>Severity</th><th>Count</th></tr>")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
            html_lines.append(f"<tr><td>{sev}</td><td>{stats[sev]}</td></tr>")
        html_lines.append("</table>")
        for i, f in enumerate(self._findings, 1):
            css_class = f.severity.lower() if f.severity.lower() in ("critical", "high", "medium", "low") else "info"
            html_lines.append(f'<div class="finding {css_class}">')
            html_lines.append(f"<h3>Finding {i}: {f.title}</h3>")
            html_lines.append(f"<p><strong>Severity:</strong> {f.severity}</p>")
            if f.cwe_id:
                html_lines.append(f"<p><strong>CWE:</strong> {f.cwe_id}</p>")
            if f.description:
                html_lines.append(f"<p>{f.description}</p>")
            if f.evidence:
                html_lines.append(f"<pre>{f.evidence}</pre>")
            if self.config.include_remediation and f.remediation:
                html_lines.append(f"<p><strong>Remediation:</strong> {f.remediation}</p>")
            html_lines.append("</div>")
        html_lines.append("</body></html>")
        return "\n".join(html_lines)

    def clear(self) -> None:
        self._findings.clear()
