"""Simple vulnerability scanner that checks for common misconfigurations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MisconfigCheck:
    name: str
    check_fn: Callable[[], tuple[bool, str]]
    severity: str = "MEDIUM"


@dataclass
class MisconfigScanner:
    checks: list[MisconfigCheck] = field(default_factory=list)

    def add_check(self, check: MisconfigCheck) -> None:
        self.checks.append(check)

    def run_all(self) -> list[dict]:
        results = []
        for check in self.checks:
            try:
                passed, message = check.check_fn()
                results.append({
                    "check": check.name,
                    "passed": passed,
                    "severity": check.severity,
                    "message": message,
                })
            except Exception as e:
                results.append({
                    "check": check.name,
                    "passed": False,
                    "severity": check.severity,
                    "message": f"error: {e}",
                })
        return results

    def failures(self) -> list[dict]:
        return [r for r in self.run_all() if not r["passed"]]


MISCONFIG_SCANNER = MisconfigScanner()