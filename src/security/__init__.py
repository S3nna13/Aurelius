"""Aurelius security subpackage.

Re-exports defensive components for convenience.
"""

from src.security.model_stealing_defense import (
    ModelStealingDefense,
    QueryAuditEntry,
    StealingThreatReport,
)

from src.security.threat_intel_correlator import (
    CorrelatedIOC,
    CorrelationReport,
    ThreatIntelCorrelator,
    ThreatIntelSource,
)

from src.security.mitre_attack_taxonomy import (
    ATTACK_TECHNIQUES,
    MitreAttackClassifier,
    TACTIC_ORDER,
    TechniqueMatch,
)

from src.security.ioc_extractor import (
    IOC,
    IOCExtractor,
    IOCReport,
)

from src.security.yara_rule_engine import (
    YaraMatch,
    YaraRule,
    YaraRuleEngine,
    YaraRuleParser,
)

from src.security.pe_file_analyzer import (
    PEInfo,
    PESection,
    analyze_pe,
    analyze_pe_file,
)

from src.security.log_anomaly_detector import (
    LogAnomaly,
    LogAnomalyDetector,
)

__all__ = [
    "ModelStealingDefense",
    "QueryAuditEntry",
    "StealingThreatReport",
    "CorrelatedIOC",
    "CorrelationReport",
    "ThreatIntelCorrelator",
    "ThreatIntelSource",
    "ATTACK_TECHNIQUES",
    "MitreAttackClassifier",
    "TACTIC_ORDER",
    "TechniqueMatch",
    "IOC",
    "IOCExtractor",
    "IOCReport",
    "YaraMatch",
    "YaraRule",
    "YaraRuleEngine",
    "YaraRuleParser",
    "PEInfo",
    "PESection",
    "analyze_pe",
    "analyze_pe_file",
    "LogAnomaly",
    "LogAnomalyDetector",
]
