"""Export vulnerability findings as SARIF 2.1.0 bundles."""

from __future__ import annotations

import json
from enum import StrEnum

from .vulnerability_schema import VulnCollection, VulnFinding, VulnSeverity


class SARIFLevel(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


def _severity_to_sarif_level(severity: VulnSeverity) -> str:
    if severity in (VulnSeverity.CRITICAL, VulnSeverity.HIGH):
        return SARIFLevel.ERROR.value
    if severity == VulnSeverity.MEDIUM:
        return SARIFLevel.WARNING.value
    return SARIFLevel.NOTE.value


class SARIFExporter:
    """Convert a VulnCollection to a SARIF 2.1.0 dict / JSON string / file."""

    TOOL_NAME = "Aurelius"
    TOOL_VERSION = "1.0"
    SCHEMA_URL = "https://json.schemastore.org/sarif-2.1.0.json"

    def build_rule(self, finding: VulnFinding) -> dict:
        rule: dict = {
            "id": finding.finding_id,
            "shortDescription": {"text": (finding.title or "")[:500]},
            "fullDescription": {"text": (finding.description or "")[:2000]},
        }
        if finding.cwe_ids:
            rule["properties"] = {"tags": list(finding.cwe_ids)}
        return rule

    def build_result(self, finding: VulnFinding) -> dict:
        msg_text = f"{finding.title}: {(finding.description or '')[:500]}"
        result: dict = {
            "ruleId": finding.finding_id,
            "level": _severity_to_sarif_level(finding.severity),
            "message": {"text": msg_text},
        }
        if finding.affected_file:
            region: dict = {}
            if finding.affected_line and finding.affected_line > 0:
                region["startLine"] = finding.affected_line
            physical: dict = {
                "artifactLocation": {"uri": finding.affected_file},
            }
            if region:
                physical["region"] = region
            result["locations"] = [{"physicalLocation": physical}]
        return result

    def export(self, collection: VulnCollection) -> dict:
        seen_rule_ids: set[str] = set()
        rules: list[dict] = []
        for f in collection.findings:
            if f.finding_id in seen_rule_ids:
                continue
            seen_rule_ids.add(f.finding_id)
            rules.append(self.build_rule(f))

        results = [self.build_result(f) for f in collection.findings]
        return {
            "version": "2.1.0",
            "$schema": self.SCHEMA_URL,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.TOOL_NAME,
                            "version": self.TOOL_VERSION,
                            "rules": rules,
                        }
                    },
                    "results": results,
                }
            ],
        }

    def export_json(self, collection: VulnCollection) -> str:
        return json.dumps(self.export(collection), indent=2)

    def export_to_file(self, collection: VulnCollection, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.export_json(collection))

    def finding_count(self, sarif_dict: dict) -> int:
        runs = sarif_dict.get("runs", [])
        if not runs:
            return 0
        return len(runs[0].get("results", []))


SARIF_EXPORTER_REGISTRY = {"default": SARIFExporter}
