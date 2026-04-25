"""Dependency vulnerability scanner — checks against known CVEs.

Trail of Bits: validate external dependencies, separate signal from noise.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VulnFinding:
    package: str
    installed: str
    cve_id: str
    severity: str
    description: str = ""
    fixed_in: str = ""

    @property
    def critical(self) -> bool:
        return self.severity.upper() == "CRITICAL"

    @property
    def high(self) -> bool:
        return self.severity.upper() == "HIGH"


@dataclass
class DependencyScanner:
    findings: list[VulnFinding] = field(default_factory=list, repr=False)

    def scan_pip(self, requirements_path: str | None = None) -> list[VulnFinding]:
        self.findings = []
        try:
            cmd = ["pip-audit", "--format", "json"]
            if requirements_path:
                cmd.extend(["-r", requirements_path])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode not in (0, 1):
                return self.findings
            data = json.loads(result.stdout)
            for vuln in data.get("vulnerabilities", []):
                self.findings.append(VulnFinding(
                    package=vuln.get("name", "unknown"),
                    installed=vuln.get("version", "?"),
                    cve_id=vuln.get("aliases", ["unknown"])[0],
                    severity=vuln.get("severity", "UNKNOWN"),
                    description=vuln.get("description", ""),
                    fixed_in=vuln.get("fixed_version", ""),
                ))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass
        return self.findings

    def critical_findings(self) -> list[VulnFinding]:
        return [f for f in self.findings if f.critical]

    def high_findings(self) -> list[VulnFinding]:
        return [f for f in self.findings if f.high]

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.findings),
            "critical": len(self.critical_findings()),
            "high": len(self.high_findings()),
        }


DEPENDENCY_SCANNER = DependencyScanner()