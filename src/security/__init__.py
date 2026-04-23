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

from src.security.soc_pipeline import (
    DEFAULT_SOC_PIPELINE,
    DecideStage,
    EnrichStage,
    Finding,
    IngestStage,
    NormalizeStage,
    PipelineStage,
    RouteStage,
    Rule,
    SOCPipeline,
    SecurityEvent,
    TriageDecision,
    evidence_first_gate,
)

from src.security.canary_pipeline import (
    CanaryConfig,
    CanaryPipeline,
    CanaryToken,
    CANARY_PIPELINE_REGISTRY,
    DEFAULT_CANARY_PIPELINE,
)

from src.security.safe_archive import (
    ArchiveError,
    SAFE_EXTRACTOR_REGISTRY,
    SafeTarExtractor,
    SafeZipExtractor,
    safe_extract,
)


# Additive registry keyed by name; test suites and integrators can register
# new defensive pipelines without touching downstream call sites.
DEFENSIVE_PIPELINE_REGISTRY: dict = {
    "default": DEFAULT_SOC_PIPELINE,
}

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
    "DEFAULT_SOC_PIPELINE",
    "DEFENSIVE_PIPELINE_REGISTRY",
    "DecideStage",
    "EnrichStage",
    "Finding",
    "IngestStage",
    "NormalizeStage",
    "PipelineStage",
    "RouteStage",
    "Rule",
    "SOCPipeline",
    "SecurityEvent",
    "TriageDecision",
    "evidence_first_gate",
    "CanaryConfig",
    "CanaryPipeline",
    "CanaryToken",
    "CANARY_PIPELINE_REGISTRY",
    "DEFAULT_CANARY_PIPELINE",
    "ArchiveError",
    "SAFE_EXTRACTOR_REGISTRY",
    "SafeTarExtractor",
    "SafeZipExtractor",
    "safe_extract",
]
