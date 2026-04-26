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

from src.security.policy_gate import (
    DEFAULT_POLICY_GATE,
    POLICY_GATE_REGISTRY,
    PolicyGate,
    PolicyScope,
    PolicyViolation,
)

from src.security.sarif_reporter import (
    SARIF_REPORTER_REGISTRY,
    SarifLevel,
    SarifReport,
    SarifResult,
    SarifRule,
)

from src.security.audit_logger import (
    AUDIT_LOGGER,
    AuditCategory,
    AuditEvent,
    AuditLevel,
    AuditLogger,
)

from src.security.sandbox_executor import (
    SANDBOX_EXECUTOR,
    SandboxConfig,
    SandboxExecutor,
    SandboxResult,
    SandboxViolation,
)

from src.security.rate_abuse_detector import (
    AbuseAlert,
    AbusePattern,
    RATE_ABUSE_DETECTOR_REGISTRY,
    RateAbuseDetector,
    RequestRecord,
)

from src.security.ip_allowlist import (
    CIDRBlock,
    IP_ALLOWLIST_REGISTRY,
    IPAllowlist,
)

from src.security.audit_trail import (
    AuditEvent as TrailAuditEvent,
    AUDIT_TRAIL_REGISTRY,
    AuditTrail,
)


# Additive registry keyed by name; test suites and integrators can register
# new defensive pipelines without touching downstream call sites.
DEFENSIVE_PIPELINE_REGISTRY: dict = {
    "default": DEFAULT_SOC_PIPELINE,
}

SECURITY_REGISTRY: dict = {
    "rate_abuse_detector": RATE_ABUSE_DETECTOR_REGISTRY,
    "ip_allowlist": IP_ALLOWLIST_REGISTRY,
    "audit_trail": AUDIT_TRAIL_REGISTRY,
}

# --- Message content filter (cycle-197) --------------------------------------
from .message_content_filter import (  # noqa: F401
    FilterConfig,
    FilterMatch,
    FilterResult,
    FilterVerdict,
    MessageContentFilter,
    MESSAGE_FILTER_REGISTRY,
    DEFAULT_MESSAGE_FILTER,
)
SECURITY_REGISTRY["message_filter"] = MESSAGE_FILTER_REGISTRY

# --- Prompt injection corpus (cycle-202) -------------------------------------
from .prompt_injection_corpus import (  # noqa: F401
    InjectionPattern,
    PromptInjectionCorpus,
    CORPUS_REGISTRY,
    DEFAULT_PROMPT_INJECTION_CORPUS,
)
SECURITY_REGISTRY["prompt_injection_corpus"] = CORPUS_REGISTRY

# --- Tool call attestation (cycle-202) ---------------------------------------
from .tool_call_attestation import (  # noqa: F401
    AttestedToolCall,
    ToolCallAttestation,
    ATTESTATION_REGISTRY,
    DEFAULT_TOOL_CALL_ATTESTATION,
)
SECURITY_REGISTRY["tool_call_attestation"] = ATTESTATION_REGISTRY

# --- Input normalizer (cycle-202) --------------------------------------------
from .input_normalizer import (  # noqa: F401
    InputNormalizer,
    NormalizationResult,
    NORMALIZER_REGISTRY,
    DEFAULT_INPUT_NORMALIZER,
)
SECURITY_REGISTRY["input_normalizer"] = NORMALIZER_REGISTRY

# --- Agent budget enforcer (cycle-202) ---------------------------------------
from .agent_budget_enforcer import (  # noqa: F401
    AgentBudget,
    AgentBudgetEnforcer,
    BudgetExhausted,
    BudgetSnapshot,
    BUDGET_ENFORCER_REGISTRY,
    DEFAULT_BUDGET_ENFORCER,
)
SECURITY_REGISTRY["agent_budget_enforcer"] = BUDGET_ENFORCER_REGISTRY

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
    # message_content_filter
    "FilterConfig",
    "FilterMatch",
    "FilterResult",
    "FilterVerdict",
    "MessageContentFilter",
    "MESSAGE_FILTER_REGISTRY",
    "DEFAULT_MESSAGE_FILTER",
    # prompt_injection_corpus
    "InjectionPattern",
    "PromptInjectionCorpus",
    "CORPUS_REGISTRY",
    "DEFAULT_PROMPT_INJECTION_CORPUS",
    # tool_call_attestation
    "AttestedToolCall",
    "ToolCallAttestation",
    "ATTESTATION_REGISTRY",
    "DEFAULT_TOOL_CALL_ATTESTATION",
    # input_normalizer
    "InputNormalizer",
    "NormalizationResult",
    "NORMALIZER_REGISTRY",
    "DEFAULT_INPUT_NORMALIZER",
    # agent_budget_enforcer
    "AgentBudget",
    "AgentBudgetEnforcer",
    "BudgetExhausted",
    "BudgetSnapshot",
    "BUDGET_ENFORCER_REGISTRY",
    "DEFAULT_BUDGET_ENFORCER",
    # combined registry
    "SECURITY_REGISTRY",
]
