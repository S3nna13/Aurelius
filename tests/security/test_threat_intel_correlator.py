"""Unit tests for ThreatIntelCorrelator."""

from __future__ import annotations

import pytest

from src.security.threat_intel_correlator import (
    CorrelatedIOC,
    CorrelationReport,
    ThreatIntelCorrelator,
    ThreatIntelSource,
)


def _src(name: str, *iocs: dict) -> ThreatIntelSource:
    return ThreatIntelSource(name=name, iocs=list(iocs))


def _ioc(t: str, v: str, c: float = 0.7, **extra) -> dict:
    d = {"type": t, "value": v, "confidence": c}
    d.update(extra)
    return d


def test_correlate_three_sources_ioc_in_two_has_source_count_two():
    c = ThreatIntelCorrelator(min_source_count=1)
    sources = [
        _src("A", _ioc("ip", "9.9.9.9")),
        _src("B", _ioc("ip", "9.9.9.9")),
        _src("C", _ioc("ip", "10.10.10.10")),
    ]
    report = c.correlate(sources)
    hit = next(x for x in report.correlated if x.value == "9.9.9.9")
    assert hit.source_count == 2
    assert sorted(hit.source_names) == ["A", "B"]


def test_ioc_seen_in_all_sources_is_high_confidence():
    c = ThreatIntelCorrelator(min_source_count=1, high_confidence_threshold=0.5)
    sources = [
        _src("A", _ioc("domain", "evil.com", 0.9)),
        _src("B", _ioc("domain", "evil.com", 0.9)),
        _src("C", _ioc("domain", "evil.com", 0.9)),
    ]
    report = c.correlate(sources)
    values = [x.value for x in report.high_confidence]
    assert "evil.com" in values


def test_min_source_count_filter_applied():
    c = ThreatIntelCorrelator(min_source_count=2)
    sources = [
        _src("A", _ioc("ip", "1.1.1.1")),
        _src("B", _ioc("ip", "2.2.2.2")),
    ]
    report = c.correlate(sources)
    assert report.correlated == []


def test_confidence_weighted_by_source_count():
    c = ThreatIntelCorrelator(min_source_count=1)
    sources = [
        _src("A", _ioc("ip", "1.1.1.1", 1.0)),
        _src("B", _ioc("ip", "1.1.1.1", 1.0)),
        _src("C", _ioc("ip", "2.2.2.2", 1.0)),
    ]
    report = c.correlate(sources)
    a = next(x for x in report.correlated if x.value == "1.1.1.1")
    b = next(x for x in report.correlated if x.value == "2.2.2.2")
    assert a.confidence > b.confidence


def test_cluster_ips_groups_by_slash24():
    c = ThreatIntelCorrelator()
    out = c.cluster_ips(["1.2.3.4", "1.2.3.5", "8.8.8.8"])
    assert "1.2.3.0/24" in out
    assert set(out["1.2.3.0/24"]) == {"1.2.3.4", "1.2.3.5"}
    assert "8.8.8.0/24" in out


def test_cluster_domains_groups_by_etld_plus_one():
    c = ThreatIntelCorrelator()
    out = c.cluster_domains(["sub.evil.com", "mail.evil.com", "good.org"])
    assert "evil.com" in out
    assert set(out["evil.com"]) == {"sub.evil.com", "mail.evil.com"}


def test_cluster_hashes_splits_md5_vs_sha256():
    c = ThreatIntelCorrelator()
    md5 = "a" * 32
    sha = "b" * 64
    out = c.cluster_hashes([md5, sha])
    assert "md5" in out and md5 in out["md5"]
    assert "sha256" in out and sha in out["sha256"]


def test_empty_sources_returns_empty_report():
    c = ThreatIntelCorrelator(min_source_count=1)
    report = c.correlate([])
    assert isinstance(report, CorrelationReport)
    assert report.correlated == []
    assert report.high_confidence == []
    assert report.clusters == {}


def test_duplicate_iocs_within_one_source_deduplicated():
    c = ThreatIntelCorrelator(min_source_count=1)
    sources = [
        _src("A", _ioc("ip", "1.1.1.1"), _ioc("ip", "1.1.1.1")),
        _src("B", _ioc("ip", "1.1.1.1")),
    ]
    report = c.correlate(sources)
    hit = next(x for x in report.correlated if x.value == "1.1.1.1")
    assert hit.source_count == 2


def test_determinism_alpha_order():
    c = ThreatIntelCorrelator(min_source_count=1)
    sources = [
        _src("A", _ioc("ip", "9.9.9.9"), _ioc("ip", "1.1.1.1")),
        _src("B", _ioc("domain", "z.com"), _ioc("domain", "a.com")),
    ]
    r1 = c.correlate(sources)
    r2 = c.correlate(sources)
    assert [x.value for x in r1.correlated] == [x.value for x in r2.correlated]
    types_values = [(x.type.lower(), x.value) for x in r1.correlated]
    assert types_values == sorted(types_values)


def test_case_insensitive_domain_comparison():
    c = ThreatIntelCorrelator(min_source_count=2)
    sources = [
        _src("A", _ioc("domain", "Evil.COM")),
        _src("B", _ioc("domain", "evil.com")),
    ]
    report = c.correlate(sources)
    assert any(x.value == "evil.com" and x.source_count == 2 for x in report.correlated)


def test_invalid_prefix_raises():
    c = ThreatIntelCorrelator()
    with pytest.raises(ValueError):
        c.cluster_ips(["1.2.3.4"], prefix=33)
    with pytest.raises(ValueError):
        c.cluster_ips(["1.2.3.4"], prefix=-1)


def test_unknown_ioc_type_passed_through():
    c = ThreatIntelCorrelator(min_source_count=1)
    sources = [
        _src("A", _ioc("custom-indicator", "xyz")),
        _src("B", _ioc("custom-indicator", "xyz")),
    ]
    report = c.correlate(sources)
    assert any(x.value == "xyz" for x in report.correlated)


def test_high_confidence_subset_of_correlated():
    c = ThreatIntelCorrelator(min_source_count=1, high_confidence_threshold=0.5)
    sources = [
        _src("A", _ioc("ip", "1.1.1.1", 0.9)),
        _src("B", _ioc("ip", "1.1.1.1", 0.9)),
        _src("C", _ioc("ip", "2.2.2.2", 0.1)),
    ]
    report = c.correlate(sources)
    for h in report.high_confidence:
        assert h in report.correlated


def test_clusters_present_all_types():
    c = ThreatIntelCorrelator(min_source_count=1)
    md5 = "a" * 32
    sources = [
        _src("A", _ioc("ip", "1.2.3.4"), _ioc("domain", "evil.com"), _ioc("md5", md5)),
    ]
    report = c.correlate(sources)
    keys = " ".join(report.clusters.keys())
    assert "ip:" in keys
    assert "domain:" in keys
    assert "hash:" in keys


def test_dataclass_types():
    c = ThreatIntelCorrelator(min_source_count=1)
    sources = [_src("A", _ioc("ip", "1.1.1.1"))]
    report = c.correlate(sources)
    assert isinstance(report, CorrelationReport)
    if report.correlated:
        assert isinstance(report.correlated[0], CorrelatedIOC)
