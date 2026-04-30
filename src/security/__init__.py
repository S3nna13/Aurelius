"""Aurelius security subpackage.

Re-exports defensive components for convenience.
"""

from src.security.audit_logger import (
    AUDIT_LOGGER,
    AuditCategory,
    AuditEvent,
    AuditLevel,
    AuditLogger,
)
from src.security.audit_trail import (
    AUDIT_TRAIL_REGISTRY,
    AuditTrail,
)
from src.security.audit_trail import (
    AuditEvent as TrailAuditEvent,
)
from src.security.canary_pipeline import (
    CANARY_PIPELINE_REGISTRY,
    DEFAULT_CANARY_PIPELINE,
    CanaryConfig,
    CanaryPipeline,
    CanaryToken,
)
from src.security.ioc_extractor import (
    IOC,
    IOCExtractor,
    IOCReport,
)
from src.security.ip_allowlist import (
    IP_ALLOWLIST_REGISTRY,
    CIDRBlock,
    IPAllowlist,
)
from src.security.log_anomaly_detector import (
    LogAnomaly,
    LogAnomalyDetector,
)
from src.security.mitre_attack_taxonomy import (
    ATTACK_TECHNIQUES,
    TACTIC_ORDER,
    MitreAttackClassifier,
    TechniqueMatch,
)
from src.security.model_stealing_defense import (
    ModelStealingDefense,
    QueryAuditEntry,
    StealingThreatReport,
)
from src.security.pe_file_analyzer import (
    PEInfo,
    PESection,
    analyze_pe,
    analyze_pe_file,
)
from src.security.policy_gate import (
    DEFAULT_POLICY_GATE,
    POLICY_GATE_REGISTRY,
    PolicyGate,
    PolicyScope,
    PolicyViolation,
)
from src.security.posture_scorer import (
    POSTURE_SCORER_REGISTRY,
    PostureScore,
    SecurityPostureScorer,
)
from src.security.rate_abuse_detector import (
    RATE_ABUSE_DETECTOR_REGISTRY,
    AbuseAlert,
    AbusePattern,
    RateAbuseDetector,
    RequestRecord,
)
from src.security.safe_archive import (
    SAFE_EXTRACTOR_REGISTRY,
    ArchiveError,
    SafeTarExtractor,
    SafeZipExtractor,
    safe_extract,
)
from src.security.sandbox_executor import (
    SANDBOX_EXECUTOR,
    SandboxConfig,
    SandboxExecutor,
    SandboxResult,
    SandboxViolation,
)
from src.security.sarif_reporter import (
    SARIF_REPORTER_REGISTRY,
    SarifLevel,
    SarifReport,
    SarifResult,
    SarifRule,
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
    SecurityEvent,
    SOCPipeline,
    TriageDecision,
    evidence_first_gate,
)
from src.security.threat_intel_correlator import (
    CorrelatedIOC,
    CorrelationReport,
    ThreatIntelCorrelator,
    ThreatIntelSource,
)
from src.security.yara_rule_engine import (
    YaraMatch,
    YaraRule,
    YaraRuleEngine,
    YaraRuleParser,
)

# Additive registry keyed by name; test suites and integrators can register
# new defensive pipelines without touching downstream call sites.
DEFENSIVE_PIPELINE_REGISTRY: dict = {
    "default": DEFAULT_SOC_PIPELINE,
}

from src.security.checkpoint_signer import (
    CHECKPOINT_SIGNER_REGISTRY,
    CheckpointIntegrityError,
    CheckpointSigner,
)
from src.security.provenance_attestation import (
    PROVENANCE_ATTESTATION_REGISTRY,
    AttestationVerificationError,
    Material,
    ProvenanceAttestation,
    ProvenanceAttestationBuilder,
    ProvenanceError,
    build_file_attestation,
)
from src.security.sbom_generator import SBOM_REGISTRY  # noqa: F401

SECURITY_REGISTRY: dict = {
    "rate_abuse_detector": RATE_ABUSE_DETECTOR_REGISTRY,
    "ip_allowlist": IP_ALLOWLIST_REGISTRY,
    "audit_trail": AUDIT_TRAIL_REGISTRY,
    "provenance_attestation": PROVENANCE_ATTESTATION_REGISTRY,
    "checkpoint_signer": CHECKPOINT_SIGNER_REGISTRY,
    "sbom_generator": SBOM_REGISTRY,
}

# --- Message content filter (cycle-197) --------------------------------------
from .message_content_filter import (  # noqa: F401
    DEFAULT_MESSAGE_FILTER,
    MESSAGE_FILTER_REGISTRY,
    FilterConfig,
    FilterMatch,
    FilterResult,
    FilterVerdict,
    MessageContentFilter,
)

SECURITY_REGISTRY["message_filter"] = MESSAGE_FILTER_REGISTRY

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
    "PolicyScope",
    "PolicyViolation",
    "PolicyGate",
    "POLICY_GATE_REGISTRY",
    "DEFAULT_POLICY_GATE",
    "SarifLevel",
    "SarifRule",
    "SarifResult",
    "SarifReport",
    "SARIF_REPORTER_REGISTRY",
    "AuditLevel",
    "AuditCategory",
    "AuditEvent",
    "AuditLogger",
    "AUDIT_LOGGER",
    "SandboxConfig",
    "SandboxViolation",
    "SandboxResult",
    "SandboxExecutor",
    "SANDBOX_EXECUTOR",
    # rate_abuse_detector
    "AbuseAlert",
    "AbusePattern",
    "RATE_ABUSE_DETECTOR_REGISTRY",
    "RateAbuseDetector",
    "RequestRecord",
    # ip_allowlist
    "CIDRBlock",
    "IP_ALLOWLIST_REGISTRY",
    "IPAllowlist",
    # audit_trail
    "TrailAuditEvent",
    "AUDIT_TRAIL_REGISTRY",
    "AuditTrail",
    # provenance_attestation
    "PROVENANCE_ATTESTATION_REGISTRY",
    "ProvenanceAttestation",
    "ProvenanceAttestationBuilder",
    "ProvenanceError",
    "AttestationVerificationError",
    "Material",
    "build_file_attestation",
    # checkpoint_signer
    "CHECKPOINT_SIGNER_REGISTRY",
    "CheckpointSigner",
    "CheckpointIntegrityError",
    # sbom_generator
    "SBOM_REGISTRY",
    # combined registry
    "SECURITY_REGISTRY",
    # posture_scorer
    "PostureScore",
    "SecurityPostureScorer",
    "POSTURE_SCORER_REGISTRY",
]
