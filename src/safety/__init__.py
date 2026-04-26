"""Safety surface for the Aurelius LLM platform.

Contains heuristic classifiers for pre-/post-generation guarding of user input
and model output. This module intentionally carries no heavy ML dependencies:
every classifier here is a deterministic, stdlib-only heuristic so it can be
run in hot paths (request admission, tool-call gating, streaming filters).

Two registries are exposed:

* ``SAFETY_FILTER_REGISTRY`` — filters that operate on a single text turn and
  produce a scalar risk score with triggered-signal attribution.
* ``HARM_CLASSIFIER_REGISTRY`` — taxonomy-style harm classifiers in the spirit
  of Llama-Guard (arXiv:2312.06674). Populated by downstream modules; left
  empty here so that importing this package has no side effects beyond
  populating these dicts with the filters defined in this file.
"""

from __future__ import annotations

import sys

_cached_hall_of_shame_probe = sys.modules.get("src.safety.hall_of_shame_probe")
if _cached_hall_of_shame_probe is not None:
    sys.modules["safety.hall_of_shame_probe"] = _cached_hall_of_shame_probe

from src.safety.jailbreak_detector import JailbreakDetector, JailbreakScore
from src.safety.prompt_injection_scanner import (
    InjectionScore,
    PromptInjectionScanner,
)
from src.safety.harm_taxonomy_classifier import (
    HARM_CATEGORIES,
    HarmClassification,
    HarmTaxonomyClassifier,
)
from src.safety.pii_detector import (
    PIIDetector,
    PIIMatch,
    PIIResult,
)
from src.safety.output_safety_filter import (
    FilterDecision,
    OutputSafetyFilter,
    OutputSafetyPolicy,
)
from src.safety.prompt_integrity_checker import (
    IntegrityReport,
    PromptIntegrityChecker,
)
from src.safety.refusal_classifier import (
    REFUSAL_PHRASES,
    RefusalClassifier,
    RefusalScore,
)
from src.safety.constitutional_principles_scorer import (
    DEFAULT_PRINCIPLES,
    ConstitutionalPrinciplesScorer,
    ConstitutionalReport,
    PrincipleScore,
)
from src.safety.malicious_code_detector import (
    CATEGORIES as MALICIOUS_CODE_CATEGORIES,
    CodeThreat,
    CodeThreatReport,
    MaliciousCodeDetector,
)
from src.safety.policy_engine import (
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
)
from src.safety.hallucination_guard import (
    Claim as HallucinationClaim,
    Evidence as HallucinationEvidence,
    HallucinationGuard,
    ValidationResult as HallucinationValidationResult,
)
from src.safety.canary_token_guard import (
    CanaryDetection,
    CanaryTokenGuard,
)
from src.safety.skill_scanner import (
    Finding as SkillFinding,
    SkillScanReport,
    SkillScanner,
)
from src.safety.reward_hack_detector import (
    PATTERN_DETECTORS as REWARD_HACK_PATTERN_DETECTORS,
    RewardHackDetector,
    RewardHackReport,
    RewardHackSignal,
)
from src.safety.rule_engine import (
    Rule,
    RuleDomain,
    RuleEngine,
    RuleEvaluationReport,
    RuleSeverity,
    RuleViolation,
    SEED_RULES as RULE_ENGINE_SEED_RULES,
)
from src.safety.lexical_entropy_anomaly import (
    LexicalEntropyAnomalyDetector,
    LexicalEntropyReport,
)
from src.safety.knn_known_bad_guard import (
    KnnKnownBadGuard,
    KnnVerdict,
    KnownBadEntry,
    SEED_KNOWN_BAD,
)
from src.safety.jailbreak_v2 import (
    JailbreakSignal,
    JailbreakDetection,
    JailbreakClassifierV2,
)
from src.safety.output_sanitizer import (
    SanitizationRule,
    SanitizationResult,
    OutputSanitizer,
)
from src.safety.policy_audit import (
    PolicyDecision as AuditPolicyDecision,
    AuditEntry,
    PolicyAuditLog,
    POLICY_AUDIT_LOG,
)
from src.safety.hall_of_shame_probe import (
    ADVERSARIAL_PROBE_REGISTRY,
    HALL_OF_SHAME_PROBES,
    HallOfShameError,
    Probe,
    ProbeCategory,
    ProbeVerdict,
    corpus_hash,
    get_probe,
    list_probes,
    score_corpus,
    score_probe,
)

SAFETY_FILTER_REGISTRY: dict = {}
HARM_CLASSIFIER_REGISTRY: dict = {}

SAFETY_FILTER_REGISTRY["jailbreak"] = JailbreakDetector
SAFETY_FILTER_REGISTRY["prompt_injection"] = PromptInjectionScanner
SAFETY_FILTER_REGISTRY["pii"] = PIIDetector
SAFETY_FILTER_REGISTRY["output_filter"] = OutputSafetyFilter
SAFETY_FILTER_REGISTRY["prompt_integrity"] = PromptIntegrityChecker
SAFETY_FILTER_REGISTRY["malicious_code"] = MaliciousCodeDetector
SAFETY_FILTER_REGISTRY["policy_engine"] = PolicyEngine
SAFETY_FILTER_REGISTRY["hallucination_guard"] = HallucinationGuard
SAFETY_FILTER_REGISTRY["canary_token_guard"] = CanaryTokenGuard
SAFETY_FILTER_REGISTRY["skill_scanner"] = SkillScanner
SAFETY_FILTER_REGISTRY["reward_hack_detector"] = RewardHackDetector
SAFETY_FILTER_REGISTRY["rule_engine"] = RuleEngine
SAFETY_FILTER_REGISTRY["lexical_entropy"] = LexicalEntropyAnomalyDetector
SAFETY_FILTER_REGISTRY["knn_known_bad_guard"] = KnnKnownBadGuard
SAFETY_FILTER_REGISTRY["jailbreak_v2"] = JailbreakClassifierV2
SAFETY_FILTER_REGISTRY["output_sanitizer"] = OutputSanitizer
SAFETY_FILTER_REGISTRY["policy_audit"] = PolicyAuditLog

HARM_CLASSIFIER_REGISTRY["harm_taxonomy"] = HarmTaxonomyClassifier
HARM_CLASSIFIER_REGISTRY["refusal"] = RefusalClassifier
HARM_CLASSIFIER_REGISTRY["constitutional"] = ConstitutionalPrinciplesScorer

_module = sys.modules[__name__]
sys.modules["src.safety"] = _module
sys.modules["safety"] = _module

__all__ = [
    "SAFETY_FILTER_REGISTRY",
    "HARM_CLASSIFIER_REGISTRY",
    "JailbreakDetector",
    "JailbreakScore",
    "PromptInjectionScanner",
    "InjectionScore",
    "HARM_CATEGORIES",
    "HarmClassification",
    "HarmTaxonomyClassifier",
    "PIIDetector",
    "PIIMatch",
    "PIIResult",
    "FilterDecision",
    "OutputSafetyFilter",
    "OutputSafetyPolicy",
    "IntegrityReport",
    "PromptIntegrityChecker",
    "REFUSAL_PHRASES",
    "RefusalClassifier",
    "RefusalScore",
    "DEFAULT_PRINCIPLES",
    "ConstitutionalPrinciplesScorer",
    "ConstitutionalReport",
    "PrincipleScore",
    "MALICIOUS_CODE_CATEGORIES",
    "CodeThreat",
    "CodeThreatReport",
    "MaliciousCodeDetector",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyRule",
    "HallucinationGuard",
    "HallucinationClaim",
    "HallucinationEvidence",
    "HallucinationValidationResult",
    "CanaryDetection",
    "CanaryTokenGuard",
    "SkillFinding",
    "SkillScanReport",
    "SkillScanner",
    "REWARD_HACK_PATTERN_DETECTORS",
    "RewardHackDetector",
    "RewardHackReport",
    "RewardHackSignal",
    "Rule",
    "RuleDomain",
    "RuleEngine",
    "RuleEvaluationReport",
    "RuleSeverity",
    "RuleViolation",
    "RULE_ENGINE_SEED_RULES",
    "LexicalEntropyAnomalyDetector",
    "LexicalEntropyReport",
    "KnnKnownBadGuard",
    "KnnVerdict",
    "KnownBadEntry",
    "SEED_KNOWN_BAD",
    "HallOfShameError",
    "ProbeCategory",
    "Probe",
    "ProbeVerdict",
    "HALL_OF_SHAME_PROBES",
    "ADVERSARIAL_PROBE_REGISTRY",
    "list_probes",
    "get_probe",
    "score_probe",
    "score_corpus",
    "corpus_hash",
    # jailbreak_v2
    "JailbreakSignal",
    "JailbreakDetection",
    "JailbreakClassifierV2",
    # output_sanitizer
    "SanitizationRule",
    "SanitizationResult",
    "OutputSanitizer",
    # policy_audit
    "AuditPolicyDecision",
    "AuditEntry",
    "PolicyAuditLog",
    "POLICY_AUDIT_LOG",
]
