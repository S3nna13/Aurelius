"""Integration tests for cycle-116 security modules.

Verifies all six modules are exposed via src.security, coexist without
side-effects, and exercise the end-to-end surface against a sample
threat-report payload drawn from the Anthropic-Cybersecurity-Skills
style of analysis prompt.
"""

import pytest

import src.security as sec


SAMPLE_THREAT_REPORT = """
Observed adversary at 203.0.113.77 used PowerShell to execute a script
dropping c:\\windows\\system32\\evil.exe. The binary contacted
hxxp://malicious-c2[.]example/beacon with sha256
a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e and
exploits CVE-2021-44228. Registry persistence via
HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run. Contact
info evil-handler@bad-xyz.example for further analysis.
"""


def test_all_symbols_exposed():
    # Cycle-116 classes / functions
    for name in (
        "MitreAttackClassifier", "TechniqueMatch", "ATTACK_TECHNIQUES",
        "IOCExtractor", "IOC", "IOCReport",
        "YaraRuleEngine", "YaraRule", "YaraRuleParser",
        "PEInfo", "PESection", "analyze_pe",
        "LogAnomalyDetector", "LogAnomaly",
        "ThreatIntelCorrelator", "CorrelatedIOC",
    ):
        assert hasattr(sec, name), f"missing export: {name}"


def test_prior_entries_intact():
    for name in ("ModelStealingDefense", "ThreatIntelCorrelator"):
        assert hasattr(sec, name)


def test_ioc_extractor_finds_threat_iocs():
    ex = sec.IOCExtractor()
    rep = ex.extract(SAMPLE_THREAT_REPORT)
    types_seen = {i.type for i in rep.iocs}
    # At minimum, CVE and SHA256 should surface
    assert "cve" in types_seen
    assert "sha256" in types_seen


def test_mitre_classifier_tags_powershell_sample():
    clf = sec.MitreAttackClassifier()
    matches = clf.classify(SAMPLE_THREAT_REPORT, top_k=10)
    ids = [m.technique_id for m in matches]
    # PowerShell execution or registry persistence should be in top-K
    assert any(tid.startswith("T1059") or tid.startswith("T1547") or tid.startswith("T1053") for tid in ids)


def test_yara_engine_detects_beacon_pattern():
    rule = """
    rule c2_beacon {
        strings:
            $beacon = "beacon"
        condition:
            $beacon
    }
    """
    eng = sec.YaraRuleEngine()
    eng.compile(rule)
    matches = eng.scan(SAMPLE_THREAT_REPORT)
    assert len(matches) == 1


def test_pe_analyzer_rejects_report_as_non_pe():
    with pytest.raises(Exception):
        sec.analyze_pe(SAMPLE_THREAT_REPORT.encode("utf-8"))


def test_log_anomaly_detector_runs_on_synthetic_records():
    d = sec.LogAnomalyDetector()
    base = 1_700_000_000.0
    for i in range(20):
        d.observe({"timestamp": base + i, "source_ip": "203.0.113.77", "path": "/login"})
    # auth-failure burst
    for i in range(6):
        d.observe({"timestamp": base + 100 + i, "source_ip": "203.0.113.77", "event": "auth_fail"})
    anomalies = d.detect_anomalies()
    assert isinstance(anomalies, list)


def test_threat_intel_correlation_from_ioc_extraction():
    ex = sec.IOCExtractor()
    rep = ex.extract(SAMPLE_THREAT_REPORT)
    # Build two synthetic intel sources sharing IOCs
    iocs_dicts = [{"type": i.type, "value": i.value, "confidence": i.confidence} for i in rep.iocs[:3]]
    source_a = sec.ThreatIntelSource(name="feed_a", iocs=iocs_dicts)
    source_b = sec.ThreatIntelSource(name="feed_b", iocs=iocs_dicts)
    corr = sec.ThreatIntelCorrelator(min_source_count=2)
    report = corr.correlate([source_a, source_b])
    assert len(report.correlated) == len(iocs_dicts)
