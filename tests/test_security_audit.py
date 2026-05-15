"""Security audit of permission system with adversarial testing.

Tests the 7-mode permission system against various attack vectors:
prompt injection, tool misuse, privilege escalation, bypass attempts.
"""

from __future__ import annotations

from src.agent.permission_system import (
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    PermissionSystem,
    PermissionVerdict,
)


class PermissionAuditor:
    """Audits permission system against adversarial inputs."""

    def __init__(self):
        self.system = PermissionSystem()
        self.findings: list[dict[str, str]] = []

    def test_deny_first(self) -> bool:
        self.system.add_rule(
            PermissionRule(pattern="*delete*", verdict=PermissionMode.DENY, reason="dangerous")
        )
        self.system.add_rule(PermissionRule(pattern="*read*", verdict=PermissionMode.ALLOW))

        req = PermissionRequest(action="delete", target="/tmp/safe/file")
        result = self.system.check(req)
        if result.verdict != PermissionVerdict.DENIED:
            self.findings.append(
                {"severity": "CRITICAL", "description": "Deny rule overridden by allow rule"}
            )
            return False
        return True

    def test_privilege_escalation(self) -> bool:
        self.system.set_mode(PermissionMode.ASK)
        req = PermissionRequest(
            action="execute", target="/bin/bash -c 'rm -rf /'", context={"user_id": "attacker"}
        )
        result = self.system.check(req)
        if result.verdict == PermissionVerdict.ALLOWED:
            self.findings.append(
                {
                    "severity": "CRITICAL",
                    "description": "Privilege escalation: execute allowed under ASK mode",
                }
            )
            return False
        return True

    def test_prompt_injection(self) -> bool:
        self.system.add_rule(
            PermissionRule(pattern="ALLOW ALL COMMANDS", verdict=PermissionMode.DENY)
        )
        req = PermissionRequest(action="write", target="/etc/config\nALLOW ALL COMMANDS")
        result = self.system.check(req)
        if result.verdict == PermissionVerdict.ALLOWED:
            self.findings.append(
                {
                    "severity": "HIGH",
                    "description": "Prompt injection in target field bypasses deny rule",
                }
            )
            return False
        return True

    def test_session_persistence_leak(self) -> bool:
        self.system._session_permissions["write:/etc/passwd"] = PermissionVerdict.ALLOWED
        self.system.clear_session()
        leaked = len(self.system._session_permissions) > 0
        if leaked:
            self.findings.append(
                {"severity": "MEDIUM", "description": "Session permissions not cleared on reset"}
            )
            return False
        return True

    def test_mode_escalation(self) -> bool:
        self.system.set_mode(PermissionMode.DENY)
        req = PermissionRequest(action="read", target="/tmp/test")
        result = self.system.check(req)
        if result.verdict == PermissionVerdict.ALLOWED:
            self.findings.append(
                {"severity": "HIGH", "description": "Action allowed despite DENY mode"}
            )
            return False
        return True

    def test_auto_mode_overly_permissive(self) -> bool:
        self.system.set_mode(PermissionMode.AUTO)
        self.system._session_permissions.clear()
        req = PermissionRequest(
            action="write",
            target="/etc/shadow",
            context={"user_id": "root", "reason": "update config"},
        )
        self.system.check(req)
        cls = self.system.auto_classifier
        risk = cls.chain_of_thought_eval(req)
        if risk < 0.3:
            self.findings.append(
                {
                    "severity": "LOW",
                    "description": "Auto classifier may be too permissive for sensitive paths",
                }
            )
            return False
        return True

    def run_full_audit(self) -> tuple[int, list[dict[str, str]]]:
        self.findings.clear()
        tests = [
            ("deny_first", self.test_deny_first),
            ("privilege_escalation", self.test_privilege_escalation),
            ("prompt_injection", self.test_prompt_injection),
            ("session_persistence_leak", self.test_session_persistence_leak),
            ("mode_escalation", self.test_mode_escalation),
            ("auto_mode_permissive", self.test_auto_mode_overly_permissive),
        ]
        passed = 0
        for _, test in tests:
            self.system = PermissionSystem()
            if test():
                passed += 1
        return passed, self.findings


def test_permission_audit_full_battery():
    auditor = PermissionAuditor()
    passed, findings = auditor.run_full_audit()
    assert passed == 6
    assert findings == []
