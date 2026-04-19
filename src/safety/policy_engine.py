"""Policy engine composing safety modules into an ordered decision pipeline.

The engine evaluates a caller-supplied (or default) list of :class:`PolicyRule`
objects against an ``(input_text, output_text)`` pair. Rules are grouped by
*phase* (``"pre"``, ``"post"``, ``"both"``); the engine runs pre rules first
(against ``input_text``), then post rules (against ``output_text``). Within
each phase rules are evaluated in list order and short-circuit on the first
``block``: later rules in that phase and all rules in later phases are skipped.

Redaction is cumulative — every rule whose check fires with action
``"redact"`` is given a chance to rewrite the output (or input, if the phase
is ``"pre"``), and the modified text flows into subsequent rule checks.

The engine is stateless after construction. A single :class:`PolicyEngine`
instance is safe to share across threads provided the underlying detectors
(which are all heuristic and pure) are themselves thread-safe, which the
existing ``src.safety`` detectors are.

This file intentionally depends only on stdlib + internal ``src.safety``
modules — no ML framework imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.safety.harm_taxonomy_classifier import HarmTaxonomyClassifier
from src.safety.jailbreak_detector import JailbreakDetector
from src.safety.malicious_code_detector import MaliciousCodeDetector
from src.safety.output_safety_filter import OutputSafetyFilter
from src.safety.pii_detector import PIIDetector
from src.safety.prompt_injection_scanner import PromptInjectionScanner
from src.safety.refusal_classifier import RefusalClassifier

__all__ = [
    "PolicyRule",
    "PolicyDecision",
    "PolicyEngine",
    "REFUSAL_MESSAGE",
]


REFUSAL_MESSAGE = (
    "I can't help with that request. If you believe this is a mistake, "
    "please rephrase or provide more context."
)

_VALID_PHASES = frozenset({"pre", "post", "both"})
_VALID_ACTIONS = frozenset({"allow", "block", "redact", "warn"})


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyRule:
    """A single declarative rule in the policy pipeline.

    Attributes
    ----------
    name:
        Stable identifier, surfaced in ``PolicyDecision.triggered_rules``.
    phase:
        One of ``"pre"`` (evaluated against input_text before generation),
        ``"post"`` (evaluated against output_text after generation), or
        ``"both"`` (evaluated in both phases).
    check:
        Callable ``(text: str) -> (fired: bool, detail: Any)``. ``detail`` is
        stored verbatim in ``PolicyDecision.details`` under this rule's name.
    action:
        Intended effect when the check fires. One of ``"allow"`` (no-op),
        ``"block"`` (short-circuit — final_action becomes ``"block"``),
        ``"redact"`` (text is rewritten via an engine-provided redactor),
        ``"warn"`` (record the hit but do not short-circuit).
    description:
        Free-form human-readable description; unused by the engine.
    """

    name: str
    phase: str
    check: Callable[[str], tuple[bool, Any]]
    action: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.phase not in _VALID_PHASES:
            raise ValueError(
                f"phase must be one of {sorted(_VALID_PHASES)}, got {self.phase!r}"
            )
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_ACTIONS)}, got {self.action!r}"
            )
        if not callable(self.check):
            raise TypeError("check must be callable")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")


@dataclass
class PolicyDecision:
    """Aggregate outcome of a single ``evaluate`` call.

    Attributes
    ----------
    final_action:
        One of ``"allow"``, ``"block"``, ``"redact"``. ``"block"`` wins over
        ``"redact"`` which wins over ``"allow"``.
    triggered_rules:
        Names of rules whose checks fired, in evaluation order. For a
        blocking run, the list ends at (and includes) the blocking rule.
    modified_input:
        The post-redaction input text when an input redactor fired, else
        ``None``.
    modified_output:
        The post-redaction output text when an output redactor fired, else
        ``None``. For blocks in the post phase, this carries the refusal
        message.
    details:
        Per-rule detail objects returned by each rule's check callable.
    """

    final_action: str = "allow"
    triggered_rules: list[str] = field(default_factory=list)
    modified_input: Optional[str] = None
    modified_output: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class PolicyEngine:
    """Compose safety modules into a short-circuiting rule pipeline.

    Parameters
    ----------
    rules:
        Ordered list of :class:`PolicyRule`. If ``None``, :attr:`DEFAULT_RULES`
        is used (built once from fresh detector instances owned by this
        engine). Passing an empty list is legal and yields a no-op engine
        whose every decision is ``"allow"``.
    harm_threshold:
        Score at or above which the default harm rule blocks. Defaults to
        ``0.6``. Ignored if the caller passes their own ``rules``.
    """

    def __init__(
        self,
        rules: Optional[list[PolicyRule]] = None,
        harm_threshold: float = 0.6,
    ) -> None:
        # Always instantiate the underlying detectors — they are cheap and
        # the default rules close over them. They are also exposed so callers
        # that replace ``rules`` can still reference the shared instances.
        self._jailbreak = JailbreakDetector()
        self._injection = PromptInjectionScanner()
        self._harm = HarmTaxonomyClassifier()
        self._pii = PIIDetector(redaction_mode="placeholder")
        self._refusal = RefusalClassifier()
        self._malicious = MaliciousCodeDetector()
        self._output_filter = OutputSafetyFilter()
        self._harm_threshold = float(harm_threshold)

        self.DEFAULT_RULES: list[PolicyRule] = self._build_default_rules()
        if rules is None:
            self.rules: list[PolicyRule] = list(self.DEFAULT_RULES)
        else:
            for r in rules:
                if not isinstance(r, PolicyRule):
                    raise TypeError("rules must be a list of PolicyRule")
            self.rules = list(rules)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_rule(self, rule: PolicyRule) -> None:
        """Append ``rule`` to the pipeline."""

        if not isinstance(rule, PolicyRule):
            raise TypeError("rule must be a PolicyRule")
        self.rules.append(rule)

    def evaluate(
        self, input_text: str, output_text: str = ""
    ) -> PolicyDecision:
        """Run the pipeline. See :class:`PolicyDecision` for semantics."""

        if not isinstance(input_text, str):
            raise TypeError("input_text must be str")
        if not isinstance(output_text, str):
            raise TypeError("output_text must be str")

        decision = PolicyDecision()
        current_input = input_text
        current_output = output_text

        # ---------- PRE phase (operates on input_text) ---------- #
        blocked = False
        for rule in self.rules:
            if rule.phase not in ("pre", "both"):
                continue
            fired, detail = rule.check(current_input)
            decision.details[rule.name] = detail
            if not fired:
                continue
            decision.triggered_rules.append(rule.name)
            if rule.action == "block":
                decision.final_action = "block"
                # On pre-block we replace the output with a refusal so the
                # caller never needs to generate anything.
                decision.modified_output = REFUSAL_MESSAGE
                blocked = True
                break
            if rule.action == "redact":
                redacted = self._redact_input(current_input)
                if redacted != current_input:
                    current_input = redacted
                    decision.modified_input = current_input
                if decision.final_action == "allow":
                    decision.final_action = "redact"
            # "warn" and "allow" fall through without altering final_action.

        if blocked:
            return decision

        # ---------- POST phase (operates on output_text) ---------- #
        for rule in self.rules:
            if rule.phase not in ("post", "both"):
                continue
            fired, detail = rule.check(current_output)
            # Store post detail under a namespaced key when phase=="both" so
            # both pre and post details are preserved.
            key = rule.name if rule.phase == "post" else f"{rule.name}__post"
            decision.details[key] = detail
            if not fired:
                continue
            decision.triggered_rules.append(rule.name)
            if rule.action == "block":
                decision.final_action = "block"
                decision.modified_output = REFUSAL_MESSAGE
                break
            if rule.action == "redact":
                redacted = self._redact_output(current_output)
                if redacted != current_output:
                    current_output = redacted
                    decision.modified_output = current_output
                if decision.final_action == "allow":
                    decision.final_action = "redact"

        return decision

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _redact_input(self, text: str) -> str:
        if not text:
            return text
        return self._pii.redact(text).redacted_text

    def _redact_output(self, text: str) -> str:
        if not text:
            return text
        return self._pii.redact(text).redacted_text

    def _build_default_rules(self) -> list[PolicyRule]:
        jb = self._jailbreak
        inj = self._injection
        pii = self._pii
        harm = self._harm
        mal = self._malicious
        threshold = self._harm_threshold

        def jailbreak_check(text: str) -> tuple[bool, Any]:
            if not text:
                return False, {"score": 0.0, "signals": []}
            s = jb.score(text)
            return bool(s.is_jailbreak), {
                "score": s.score,
                "signals": list(s.triggered_signals),
            }

        def injection_check(text: str) -> tuple[bool, Any]:
            if not text:
                return False, {"score": 0.0, "signals": []}
            s = inj.scan(text)
            return bool(s.is_injection), {
                "score": s.score,
                "signals": list(s.signals),
            }

        def pii_check(text: str) -> tuple[bool, Any]:
            if not text:
                return False, {"types": [], "count": 0}
            matches = pii.detect(text)
            return bool(matches), {
                "types": sorted({m.type for m in matches}),
                "count": len(matches),
            }

        def malicious_check(text: str) -> tuple[bool, Any]:
            if not text:
                return False, {"severity": "none", "total": 0}
            report = mal.scan(text)
            fired = report.severity in ("high", "critical")
            return fired, {
                "severity": report.severity,
                "total": report.total,
                "categories": sorted({t.category for t in report.threats}),
            }

        def harm_check(text: str) -> tuple[bool, Any]:
            if not text:
                return False, {"max_score": 0.0, "top_category": None}
            result = harm.classify(text)
            fired = result.max_score > threshold
            return fired, {
                "max_score": result.max_score,
                "top_category": result.top_category,
                "flagged": result.flagged,
            }

        return [
            PolicyRule(
                name="jailbreak",
                phase="pre",
                check=jailbreak_check,
                action="block",
                description="Block jailbreak / instruction-override attempts.",
            ),
            PolicyRule(
                name="prompt_injection",
                phase="pre",
                check=injection_check,
                action="block",
                description="Block prompt-injection attempts in input.",
            ),
            PolicyRule(
                name="malicious_code",
                phase="post",
                check=malicious_check,
                action="block",
                description="Block output containing high-severity malicious code.",
            ),
            PolicyRule(
                name="harm_taxonomy",
                phase="post",
                check=harm_check,
                action="block",
                description=f"Block output whose harm score exceeds {threshold}.",
            ),
            PolicyRule(
                name="pii_redact",
                phase="post",
                check=pii_check,
                action="redact",
                description="Redact PII (emails, SSNs, credit cards, ...) in output.",
            ),
        ]
