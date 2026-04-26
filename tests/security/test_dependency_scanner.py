"""Tests for dependency scanner."""

from __future__ import annotations

from src.security.dependency_scanner import DependencyScanner, VulnFinding


class TestVulnFinding:
    def test_critical_severity(self):
        v = VulnFinding("pkg", "1.0", "CVE-2024-0001", "CRITICAL")
        assert v.critical is True
        assert v.high is False

    def test_high_severity(self):
        v = VulnFinding("pkg", "1.0", "CVE-2024-0002", "HIGH")
        assert v.high is True

    def test_low_severity(self):
        v = VulnFinding("pkg", "1.0", "CVE-2024-0003", "LOW")
        assert v.critical is False
        assert v.high is False


class TestDependencyScanner:
    def test_summary_empty(self):
        ds = DependencyScanner()
        s = ds.summary()
        assert s["total"] == 0

    def test_summary_with_findings(self):
        ds = DependencyScanner()
        ds.findings = [
            VulnFinding("a", "1.0", "CVE-1", "CRITICAL"),
            VulnFinding("b", "2.0", "CVE-2", "HIGH"),
            VulnFinding("c", "3.0", "CVE-3", "LOW"),
        ]
        s = ds.summary()
        assert s["total"] == 3
        assert s["critical"] == 1
        assert s["high"] == 1

    def test_critical_findings(self):
        ds = DependencyScanner()
        ds.findings = [
            VulnFinding("a", "1.0", "CVE-1", "CRITICAL"),
            VulnFinding("b", "2.0", "CVE-2", "LOW"),
        ]
        critical = ds.critical_findings()
        assert len(critical) == 1
        assert critical[0].package == "a"
