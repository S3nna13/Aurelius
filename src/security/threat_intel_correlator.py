"""Threat-intelligence correlator.

Aggregates IOCs from multiple caller-provided sources, normalizes and
deduplicates them, computes source-agreement scores, and groups related IOCs
(IPs by /N prefix, domains by effective second-level label, hashes by family).

Pure standard library only.
"""

from __future__ import annotations

import ipaddress
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ThreatIntelSource:
    """A single source feed of IOCs.

    Each IOC is a dict with at minimum ``type`` and ``value`` keys and an
    optional ``confidence`` float in ``[0, 1]``.
    """

    name: str
    iocs: list[dict]


@dataclass
class CorrelatedIOC:
    """An IOC after cross-source correlation."""

    type: str
    value: str
    source_count: int
    source_names: list[str]
    confidence: float
    first_seen: str | None = None


@dataclass
class CorrelationReport:
    """Output of a correlation run."""

    correlated: list[CorrelatedIOC]
    high_confidence: list[CorrelatedIOC]
    clusters: dict[str, list[str]] = field(default_factory=dict)


def _normalize_value(ioc_type: str, value: str) -> str:
    v = value.strip()
    t = (ioc_type or "").lower()
    if t in ("domain", "url", "email"):
        return v.lower()
    if t in ("md5", "sha1", "sha256", "sha512", "hash"):
        return v.lower()
    if t in ("ip", "ipv4", "ipv6"):
        return v
    return v


class ThreatIntelCorrelator:
    """Cross-source correlator for threat intelligence IOCs."""

    def __init__(
        self,
        min_source_count: int = 2,
        high_confidence_threshold: float = 0.8,
    ) -> None:
        if min_source_count < 1:
            raise ValueError("min_source_count must be >= 1")
        if not 0.0 <= high_confidence_threshold <= 1.0:
            raise ValueError("high_confidence_threshold must be in [0, 1]")
        self.min_source_count = min_source_count
        self.high_confidence_threshold = high_confidence_threshold

    # ------------------------------------------------------------------ #
    # Correlation                                                        #
    # ------------------------------------------------------------------ #
    def correlate(self, sources: list[ThreatIntelSource]) -> CorrelationReport:
        total_sources = len(sources)
        # key -> dict
        agg: dict[tuple[str, str], dict] = {}
        for src in sources:
            seen_in_source: set[tuple[str, str]] = set()
            for raw in src.iocs:
                if not isinstance(raw, dict):
                    continue
                itype = str(raw.get("type", "unknown"))
                ivalue = raw.get("value")
                if ivalue is None:
                    continue
                norm = _normalize_value(itype, str(ivalue))
                key = (itype.lower(), norm)
                if key in seen_in_source:
                    # dedup within a single source
                    continue
                seen_in_source.add(key)
                conf = raw.get("confidence", 0.5)
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 0.5
                first_seen = raw.get("first_seen")
                if key not in agg:
                    agg[key] = {
                        "type": itype,
                        "value": norm,
                        "source_names": [],
                        "confidences": [],
                        "first_seen": first_seen,
                    }
                agg[key]["source_names"].append(src.name)
                agg[key]["confidences"].append(conf)
                existing_fs = agg[key]["first_seen"]
                if first_seen and (existing_fs is None or first_seen < existing_fs):
                    agg[key]["first_seen"] = first_seen

        correlated: list[CorrelatedIOC] = []
        for key, data in agg.items():
            count = len(data["source_names"])
            if count < self.min_source_count:
                continue
            base = sum(data["confidences"]) / len(data["confidences"])
            # Weight: scale average confidence by agreement ratio.
            if total_sources > 0:
                agreement = count / total_sources
            else:
                agreement = 0.0
            weighted = base * agreement
            if weighted > 1.0:
                weighted = 1.0
            correlated.append(
                CorrelatedIOC(
                    type=data["type"],
                    value=data["value"],
                    source_count=count,
                    source_names=sorted(set(data["source_names"])),
                    confidence=round(weighted, 6),
                    first_seen=data["first_seen"],
                )
            )

        # Deterministic ordering: by type then value.
        correlated.sort(key=lambda c: (c.type.lower(), c.value))
        high_conf = [c for c in correlated if c.confidence >= self.high_confidence_threshold]

        # Cluster by IOC type buckets.
        ips = [c.value for c in correlated if c.type.lower() in ("ip", "ipv4")]
        domains = [c.value for c in correlated if c.type.lower() == "domain"]
        hashes = [
            c.value
            for c in correlated
            if c.type.lower() in ("md5", "sha1", "sha256", "sha512", "hash")
        ]

        clusters: dict[str, list[str]] = {}
        if ips:
            for net, members in self.cluster_ips(ips).items():
                clusters[f"ip:{net}"] = members
        if domains:
            for root, members in self.cluster_domains(domains).items():
                clusters[f"domain:{root}"] = members
        if hashes:
            for fam, members in self.cluster_hashes(hashes).items():
                clusters[f"hash:{fam}"] = members

        return CorrelationReport(
            correlated=correlated,
            high_confidence=high_conf,
            clusters=clusters,
        )

    # ------------------------------------------------------------------ #
    # Clustering primitives                                              #
    # ------------------------------------------------------------------ #
    def cluster_ips(self, ip_list: list[str], prefix: int = 24) -> dict[str, list[str]]:
        if prefix < 0 or prefix > 32:
            raise ValueError("prefix must be between 0 and 32")
        buckets: dict[str, list[str]] = defaultdict(list)
        for ip in ip_list:
            try:
                addr = ipaddress.ip_address(ip.strip())
            except ValueError:
                continue
            if isinstance(addr, ipaddress.IPv4Address):
                net = ipaddress.ip_network(f"{addr}/{prefix}", strict=False)
                buckets[str(net)].append(str(addr))
        out: dict[str, list[str]] = {}
        for k in sorted(buckets):
            out[k] = sorted(set(buckets[k]))
        return out

    def cluster_domains(self, domain_list: list[str]) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for d in domain_list:
            if not d:
                continue
            name = d.strip().lower().rstrip(".")
            parts = [p for p in name.split(".") if p]
            if len(parts) < 2:
                root = name
            else:
                # naive eTLD+1: take the last two labels.
                root = ".".join(parts[-2:])
            buckets[root].append(name)
        out: dict[str, list[str]] = {}
        for k in sorted(buckets):
            out[k] = sorted(set(buckets[k]))
        return out

    def cluster_hashes(self, hash_list: list[str]) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for h in hash_list:
            if not h:
                continue
            v = h.strip().lower()
            if not all(ch in "0123456789abcdef" for ch in v):
                family = "unknown"
            else:
                n = len(v)
                family = {
                    32: "md5",
                    40: "sha1",
                    64: "sha256",
                    128: "sha512",
                }.get(n, "unknown")
            buckets[family].append(v)
        out: dict[str, list[str]] = {}
        for k in sorted(buckets):
            out[k] = sorted(set(buckets[k]))
        return out


__all__ = [
    "ThreatIntelSource",
    "CorrelatedIOC",
    "CorrelationReport",
    "ThreatIntelCorrelator",
]
