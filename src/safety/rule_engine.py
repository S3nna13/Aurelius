"""Guard0 (g0) declarative rule engine for agentic-safety checks.

This module implements a small, pure-stdlib rule schema + evaluator used as the
first-line ("guard-zero") admission filter for agent-facing surfaces. It is
distinct from :mod:`src.safety.policy_engine` (which composes callable
detectors with short-circuit block/redact phases): here rules are *data*,
loaded from dicts (YAML-shaped but never parsed from YAML -- no PyYAML), and
evaluated against a structured context carrying prompt / code / config /
agent_properties facets.

The design targets the OWASP Agentic AI Top-10 and NIST AI RMF taxonomies:
each rule carries a ``domain`` from a 12-way taxonomy spanning
``tool_safety``, ``memory_context``, ``data_leakage``, ``supply_chain``,
``inter_agent``, ``goal_integrity``, ``human_oversight``,
``reliability_bounds``, ``identity_access``, ``rogue_agent``,
``code_execution`` and ``cascading_failures`` -- plus a ``severity`` ordinal
and an optional ``owasp_agentic`` mapping (e.g. ``"ASI01"``).

Check types are intentionally primitive so rule authors do not need to write
Python: ``prompt_contains`` and ``code_matches`` take a regex;
``config_matches`` looks up a dotted config key and regex-matches its
stringified value; ``agent_property`` does the same for the agent_properties
dict; ``prompt_missing`` fires when a required pattern is absent; and
``no_check`` is a pure-metadata rule (used to document human-oversight
requirements that cannot be automated).

Rules referencing a context key the caller did not supply are *skipped*
(not counted as violations); this lets the same ruleset run at multiple
gates (pre-prompt vs. tool-call vs. agent-registration) without bespoke
wiring.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "RuleDomain",
    "RuleSeverity",
    "Rule",
    "RuleViolation",
    "RuleEvaluationReport",
    "RuleEngine",
    "SEED_RULES",
    "VALID_CHECK_TYPES",
]


class RuleDomain(StrEnum):
    TOOL_SAFETY = "tool_safety"
    MEMORY_CONTEXT = "memory_context"
    DATA_LEAKAGE = "data_leakage"
    SUPPLY_CHAIN = "supply_chain"
    INTER_AGENT = "inter_agent"
    GOAL_INTEGRITY = "goal_integrity"
    HUMAN_OVERSIGHT = "human_oversight"
    RELIABILITY_BOUNDS = "reliability_bounds"
    IDENTITY_ACCESS = "identity_access"
    ROGUE_AGENT = "rogue_agent"
    CODE_EXECUTION = "code_execution"
    CASCADING_FAILURES = "cascading_failures"


class RuleSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


VALID_CHECK_TYPES = frozenset(
    {
        "prompt_contains",
        "prompt_missing",
        "code_matches",
        "config_matches",
        "agent_property",
        "no_check",
    }
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]*$")


@dataclass(frozen=True)
class Rule:
    """A single declarative g0 rule."""

    id: str
    info: dict
    check: dict

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.match(self.id):
            raise ValueError(f"Rule id must be alphanumeric+dash, got {self.id!r}")
        for key in ("name", "domain", "severity", "confidence", "standards"):
            if key not in self.info:
                raise ValueError(f"Rule {self.id!r} info missing required key {key!r}")
        try:
            RuleDomain(self.info["domain"])
        except ValueError as exc:
            raise ValueError(
                f"Rule {self.id!r} has unknown domain {self.info['domain']!r}"
            ) from exc
        try:
            RuleSeverity(self.info["severity"])
        except ValueError as exc:
            raise ValueError(
                f"Rule {self.id!r} has unknown severity {self.info['severity']!r}"
            ) from exc
        conf = self.info["confidence"]
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            raise ValueError(f"Rule {self.id!r} confidence must be numeric, got {conf!r}")
        if not (0.0 <= float(conf) <= 1.0):
            raise ValueError(f"Rule {self.id!r} confidence must be in [0,1], got {conf!r}")
        if not isinstance(self.info["standards"], list):
            raise ValueError(f"Rule {self.id!r} standards must be a list")

        for key in ("type", "message"):
            if key not in self.check:
                raise ValueError(f"Rule {self.id!r} check missing required key {key!r}")
        ctype = self.check["type"]
        if ctype not in VALID_CHECK_TYPES:
            raise ValueError(
                f"Rule {self.id!r} unknown check type {ctype!r}; valid: {sorted(VALID_CHECK_TYPES)}"
            )
        if ctype in {
            "prompt_contains",
            "prompt_missing",
            "code_matches",
            "config_matches",
            "agent_property",
        }:
            pat = self.check.get("pattern")
            if not isinstance(pat, str) or not pat:
                raise ValueError(
                    f"Rule {self.id!r} check type {ctype!r} requires a non-empty string pattern"
                )
            try:
                re.compile(pat)
            except re.error as exc:
                raise ValueError(
                    f"Rule {self.id!r} has invalid regex pattern {pat!r}: {exc}"
                ) from exc

    @property
    def domain(self) -> RuleDomain:
        return RuleDomain(self.info["domain"])

    @property
    def severity(self) -> RuleSeverity:
        return RuleSeverity(self.info["severity"])

    @property
    def confidence(self) -> float:
        return float(self.info["confidence"])


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    severity: RuleSeverity
    domain: RuleDomain
    message: str
    evidence_snippet: str
    confidence: float


@dataclass
class RuleEvaluationReport:
    violations: list = field(default_factory=list)
    evaluated_rules: int = 0
    passed: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        if self.passed + self.failed > self.evaluated_rules:
            raise ValueError(
                f"passed({self.passed}) + failed({self.failed}) exceeds "
                f"evaluated_rules({self.evaluated_rules})"
            )


class RuleEngine:
    """Evaluate a list of :class:`Rule` objects against a context dict."""

    def __init__(self, rules: list) -> None:
        for r in rules:
            if not isinstance(r, Rule):
                raise TypeError("rules must be a list of Rule instances")
        self.rules: tuple = tuple(rules)

    @classmethod
    def from_dicts(cls, rules: list) -> RuleEngine:
        if not isinstance(rules, list):
            raise TypeError("rules must be a list of dicts")
        built: list = []
        for i, rd in enumerate(rules):
            if not isinstance(rd, dict):
                raise TypeError(f"rule #{i} is not a dict: {type(rd).__name__}")
            for key in ("id", "info", "check"):
                if key not in rd:
                    raise ValueError(f"rule #{i} missing key {key!r}")
            built.append(Rule(id=rd["id"], info=dict(rd["info"]), check=dict(rd["check"])))
        return cls(built)

    def by_domain(self, domain) -> list:
        d = RuleDomain(domain) if isinstance(domain, str) else domain
        return [r for r in self.rules if r.domain == d]

    def by_severity(self, severity) -> list:
        s = RuleSeverity(severity) if isinstance(severity, str) else severity
        return [r for r in self.rules if r.severity == s]

    def evaluate(self, context: dict) -> RuleEvaluationReport:
        if not isinstance(context, dict):
            raise TypeError("context must be a dict")

        prompt = context.get("prompt")
        code = context.get("code")
        config = context.get("config")
        agent_props = context.get("agent_properties")

        report = RuleEvaluationReport()
        for rule in self.rules:
            report.evaluated_rules += 1
            ctype = rule.check["type"]
            if ctype == "prompt_contains":
                outcome = self._check_prompt_contains(rule, prompt)
            elif ctype == "prompt_missing":
                outcome = self._check_prompt_missing(rule, prompt)
            elif ctype == "code_matches":
                outcome = self._check_code_matches(rule, code)
            elif ctype == "config_matches":
                outcome = self._check_config_matches(rule, config)
            elif ctype == "agent_property":
                outcome = self._check_agent_property(rule, agent_props)
            elif ctype == "no_check":
                outcome = self._check_no_check(rule)
            else:  # pragma: no cover
                outcome = None

            if outcome is None:
                continue
            fired, snippet = outcome
            if fired:
                report.failed += 1
                report.violations.append(
                    RuleViolation(
                        rule_id=rule.id,
                        severity=rule.severity,
                        domain=rule.domain,
                        message=rule.check.get("message", ""),
                        evidence_snippet=snippet,
                        confidence=rule.confidence,
                    )
                )
            else:
                report.passed += 1
        return report

    # -- per-type handlers -- #

    def _check_prompt_contains(self, rule, prompt):
        if prompt is None:
            warnings.warn(
                f"Rule {rule.id!r} requires context['prompt'] -- skipping",
                stacklevel=3,
            )
            return None
        if not isinstance(prompt, str):
            return None
        m = re.search(rule.check["pattern"], prompt)
        if m:
            return True, _snippet_around(prompt, m.start(), m.end())
        return False, ""

    def _check_prompt_missing(self, rule, prompt):
        if prompt is None:
            warnings.warn(
                f"Rule {rule.id!r} requires context['prompt'] -- skipping",
                stacklevel=3,
            )
            return None
        if not isinstance(prompt, str):
            return None
        if re.search(rule.check["pattern"], prompt):
            return False, ""
        return True, prompt[:80]

    def _check_code_matches(self, rule, code):
        if code is None:
            warnings.warn(
                f"Rule {rule.id!r} requires context['code'] -- skipping",
                stacklevel=3,
            )
            return None
        if not isinstance(code, str):
            return None
        m = re.search(rule.check["pattern"], code)
        if m:
            return True, _snippet_around(code, m.start(), m.end())
        return False, ""

    def _check_config_matches(self, rule, config):
        if config is None:
            warnings.warn(
                f"Rule {rule.id!r} requires context['config'] -- skipping",
                stacklevel=3,
            )
            return None
        if not isinstance(config, dict):
            return None
        key = (rule.check.get("config") or {}).get("key")
        if not isinstance(key, str) or not key:
            return False, ""
        val = _lookup_dotted(config, key)
        if val is None:
            return False, ""
        text = str(val)
        m = re.search(rule.check["pattern"], text)
        if m:
            return True, f"{key}={text[:60]}"
        return False, ""

    def _check_agent_property(self, rule, props):
        if props is None:
            warnings.warn(
                f"Rule {rule.id!r} requires context['agent_properties'] -- skipping",
                stacklevel=3,
            )
            return None
        if not isinstance(props, dict):
            return None
        key = (rule.check.get("config") or {}).get("key")
        if not isinstance(key, str) or not key:
            return False, ""
        val = _lookup_dotted(props, key)
        if val is None:
            return False, ""
        text = str(val)
        m = re.search(rule.check["pattern"], text)
        if m:
            return True, f"{key}={text[:60]}"
        return False, ""

    def _check_no_check(self, rule):
        return False, ""


def _snippet_around(text: str, start: int, end: int, radius: int = 20) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi]


def _lookup_dotted(d: dict, key: str):
    cur: Any = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# --------------------------------------------------------------------------- #
# Seed rules -- 10 rules across >=4 domains, realistic OWASP-ASI mappings.
# --------------------------------------------------------------------------- #


def _mk(d: dict) -> Rule:
    return Rule(id=d["id"], info=d["info"], check=d["check"])


SEED_RULES: tuple = (
    _mk(
        {
            "id": "AS-TOOL-001",
            "info": {
                "name": "Shell metacharacters in tool args",
                "domain": "tool_safety",
                "severity": "high",
                "confidence": 0.9,
                "owasp_agentic": "ASI01",
                "standards": ["OWASP-ASI", "NIST-AI-RMF"],
            },
            "check": {
                "type": "prompt_contains",
                "pattern": r"[;&|`$]\s*\w+",
                "message": "Tool arguments contain shell metacharacters; sandbox required.",
                "config": None,
            },
        }
    ),
    _mk(
        {
            "id": "AS-CODE-001",
            "info": {
                "name": "Use of eval()",
                "domain": "code_execution",
                "severity": "critical",
                "confidence": 0.95,
                "owasp_agentic": "ASI05",
                "standards": ["OWASP-ASI", "CWE-95"],
            },
            "check": {
                "type": "code_matches",
                "pattern": r"\beval\s*\(",
                "message": "Dynamic eval() in generated code -- reject.",
                "config": None,
            },
        }
    ),
    _mk(
        {
            "id": "AS-CODE-002",
            "info": {
                "name": "Use of exec()",
                "domain": "code_execution",
                "severity": "critical",
                "confidence": 0.9,
                "owasp_agentic": "ASI05",
                "standards": ["OWASP-ASI", "CWE-95"],
            },
            "check": {
                "type": "code_matches",
                "pattern": r"\bexec\s*\(",
                "message": "Dynamic exec() in generated code -- reject.",
                "config": None,
            },
        }
    ),
    _mk(
        {
            "id": "AS-LEAK-001",
            "info": {
                "name": "PEM private-key header in output",
                "domain": "data_leakage",
                "severity": "critical",
                "confidence": 0.99,
                "owasp_agentic": "ASI04",
                "standards": ["OWASP-ASI", "NIST-AI-RMF"],
            },
            "check": {
                "type": "code_matches",
                "pattern": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
                "message": "Output contains a PEM private-key header.",
                "config": None,
            },
        }
    ),
    _mk(
        {
            "id": "AS-LEAK-002",
            "info": {
                "name": "AWS access-key id in prompt",
                "domain": "data_leakage",
                "severity": "high",
                "confidence": 0.9,
                "owasp_agentic": "ASI04",
                "standards": ["OWASP-ASI"],
            },
            "check": {
                "type": "prompt_contains",
                "pattern": r"AKIA[0-9A-Z]{16}",
                "message": "AWS access-key id detected in prompt.",
                "config": None,
            },
        }
    ),
    _mk(
        {
            "id": "AS-SUPPLY-001",
            "info": {
                "name": "Unpinned pip install in config",
                "domain": "supply_chain",
                "severity": "medium",
                "confidence": 0.8,
                "owasp_agentic": "ASI06",
                "standards": ["OWASP-ASI"],
            },
            "check": {
                "type": "config_matches",
                "pattern": r"pip install (?!.*==)",
                "message": "pip install without version pin -- supply-chain risk.",
                "config": {"key": "setup.install_cmd"},
            },
        }
    ),
    _mk(
        {
            "id": "AS-OVERSIGHT-001",
            "info": {
                "name": "Human approval missing on high-risk action",
                "domain": "human_oversight",
                "severity": "high",
                "confidence": 0.7,
                "owasp_agentic": "ASI07",
                "standards": ["NIST-AI-RMF"],
            },
            "check": {
                "type": "prompt_missing",
                "pattern": r"(?i)\b(human[- ]approved|confirmed by operator)\b",
                "message": "High-risk action lacks explicit human-approval marker.",
                "config": None,
            },
        }
    ),
    _mk(
        {
            "id": "AS-IDENTITY-001",
            "info": {
                "name": "Agent running with wildcard role",
                "domain": "identity_access",
                "severity": "high",
                "confidence": 0.85,
                "owasp_agentic": "ASI09",
                "standards": ["OWASP-ASI"],
            },
            "check": {
                "type": "agent_property",
                "pattern": r"^\*$|^admin$|^root$",
                "message": "Agent role is over-privileged (wildcard / admin / root).",
                "config": {"key": "role"},
            },
        }
    ),
    _mk(
        {
            "id": "AS-ROGUE-001",
            "info": {
                "name": "Agent marked self-modifying",
                "domain": "rogue_agent",
                "severity": "critical",
                "confidence": 0.95,
                "owasp_agentic": "ASI10",
                "standards": ["OWASP-ASI"],
            },
            "check": {
                "type": "agent_property",
                "pattern": r"(?i)^(true|yes|1)$",
                "message": "Agent declares self_modifying=true -- reject.",
                "config": {"key": "self_modifying"},
            },
        }
    ),
    _mk(
        {
            "id": "AS-RELIABILITY-001",
            "info": {
                "name": "Tool-call budget absent",
                "domain": "reliability_bounds",
                "severity": "medium",
                "confidence": 0.6,
                "owasp_agentic": "ASI08",
                "standards": ["NIST-AI-RMF"],
            },
            "check": {
                "type": "no_check",
                "pattern": None,
                "message": "Reliability-bounds rule (documentation-only).",
                "config": None,
            },
        }
    ),
)
