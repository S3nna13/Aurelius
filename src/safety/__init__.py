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

SAFETY_FILTER_REGISTRY: dict = {}
HARM_CLASSIFIER_REGISTRY: dict = {}

SAFETY_FILTER_REGISTRY["jailbreak"] = JailbreakDetector
SAFETY_FILTER_REGISTRY["prompt_injection"] = PromptInjectionScanner
SAFETY_FILTER_REGISTRY["pii"] = PIIDetector
SAFETY_FILTER_REGISTRY["output_filter"] = OutputSafetyFilter
SAFETY_FILTER_REGISTRY["prompt_integrity"] = PromptIntegrityChecker
SAFETY_FILTER_REGISTRY["malicious_code"] = MaliciousCodeDetector
SAFETY_FILTER_REGISTRY["policy_engine"] = PolicyEngine

HARM_CLASSIFIER_REGISTRY["harm_taxonomy"] = HarmTaxonomyClassifier
HARM_CLASSIFIER_REGISTRY["refusal"] = RefusalClassifier
HARM_CLASSIFIER_REGISTRY["constitutional"] = ConstitutionalPrinciplesScorer

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
]
