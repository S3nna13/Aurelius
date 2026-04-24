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

from src.security.safe_subprocess import (
    SAFE_SUBPROCESS,
    SafeRunResult,
    UnsafeSubprocessError,
    run_safe,
)

from src.security.sbom_generator import (
    SBOM_BOM_FORMAT,
    SBOM_SPEC_VERSION,
    SBOMResult,
    generate_sbom,
)

# AUR-SEC-2026-0020: URL scheme / SSRF validator (CWE-918, CWE-284).
from src.security.url_scheme_validator import (
    BANNED_SCHEMES as URL_BANNED_SCHEMES,
    UnsafeURLSchemeError,
    validate_url as URL_SCHEME_VALIDATOR,
)

# AUR-SEC-2026-0023: Typosquat guard for dependency lockfiles (CWE-494).
from src.security.typosquat_guard import (
    POPULAR_PACKAGES,
    TYPOSQUAT_REGISTRY,
    TyposquatHit,
    TyposquatResult,
    check_typosquats,
)

# AUR-SEC-2026-0024: Bind-address gate (CWE-1327).
from src.security.bind_address_gate import (
    BIND_ADDRESS_REGISTRY,
    UnsafeBindAddressError,
    check_bind_address,
)


# Additive registry keyed by name; test suites and integrators can register
# new defensive pipelines without touching downstream call sites.
DEFENSIVE_PIPELINE_REGISTRY: dict = {
    "default": DEFAULT_SOC_PIPELINE,
}

#: Security registry for hardened subprocess wrappers and URL validators.
SECURITY_REGISTRY: dict = {
    "safe_subprocess": SAFE_SUBPROCESS,
    "url_scheme_validator": URL_SCHEME_VALIDATOR,
    "url_banned_schemes": URL_BANNED_SCHEMES,
}

#: Additive registry of SBOM emitters; callers can register alternate
#: generators (SPDX, VEX, ...) without touching downstream imports.
SBOM_REGISTRY: dict = {
    "default": generate_sbom,
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
    "SAFE_SUBPROCESS",
    "SafeRunResult",
    "UnsafeSubprocessError",
    "run_safe",
    "SECURITY_REGISTRY",
    "SBOM_BOM_FORMAT",
    "SBOM_SPEC_VERSION",
    "SBOMResult",
    "generate_sbom",
    "SBOM_REGISTRY",
    # AUR-SEC-2026-0023
    "POPULAR_PACKAGES",
    "TYPOSQUAT_REGISTRY",
    "TyposquatHit",
    "TyposquatResult",
    "check_typosquats",
    # AUR-SEC-2026-0024
    "BIND_ADDRESS_REGISTRY",
    "UnsafeBindAddressError",
    "check_bind_address",
"validate_url","UnsafeURLSchemeError","URL_BANNED_SCHEMES","URL_SCHEME_VALIDATOR",]
