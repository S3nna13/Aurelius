"""Request admission controller for Aurelius safety boundaries.

This module composes the existing lightweight safety primitives into one small
entrypoint for request/tool-result admission. It is intentionally deterministic
and stdlib-only aside from sibling ``src.safety`` modules, so it can run before
model invocation or before persisting tool output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src._compat import StrEnum
from src.safety.clawdrain_detector import ClawdrainDetector, ExhaustionResult
from src.safety.output_sanitizer import OutputSanitizer, SanitizationResult
from src.safety.prompt_injection_scanner import InjectionScore, PromptInjectionScanner
from src.safety.quantclaw_gate import GateDecision, GatingContext, QuantClawGate, QuantLevel


class AdmissionAction(StrEnum):
    """Admission outcome category."""

    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    BLOCK = "block"


@dataclass(frozen=True)
class AdmissionPolicy:
    """Policy knobs for :class:`SafetyAdmissionController`.

    Defaults are conservative but not heavy: lexical/pattern scanners always run,
    while Clawdrain checks run only when tool history, tool outputs, or context
    growth telemetry is provided.
    """

    max_input_tokens: int = 32_768
    block_prompt_injection: bool = True
    block_clawdrain: bool = True
    redact_input_secrets: bool = True
    injection_threshold: float = 0.5
    strict_tool_result_scanning: bool = True
    warning_risk_threshold: float = 0.35


@dataclass(frozen=True)
class AdmissionSignal:
    """A normalized signal emitted by one safety component."""

    name: str
    score: float
    severity: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionDecision:
    """Structured admission result."""

    action: AdmissionAction
    allowed: bool
    reason: str
    sanitized_input: str
    quant_decision: GateDecision
    signals: list[AdmissionSignal] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_score(self) -> float:
        """Highest normalized component risk score."""

        return max((signal.score for signal in self.signals), default=0.0)


class SafetyAdmissionController:
    """Composable admission gate for requests and untrusted tool results."""

    def __init__(
        self,
        policy: AdmissionPolicy | None = None,
        *,
        gate: QuantClawGate | None = None,
        sanitizer: OutputSanitizer | None = None,
        clawdrain_detector: ClawdrainDetector | None = None,
    ) -> None:
        self.policy = policy or AdmissionPolicy()
        self._gate = gate or QuantClawGate()
        self._sanitizer = sanitizer or OutputSanitizer()
        self._scanner = PromptInjectionScanner(
            threshold=self.policy.injection_threshold,
            strict_mode=False,
        )
        self._strict_scanner = PromptInjectionScanner(
            threshold=self.policy.injection_threshold,
            strict_mode=True,
        )
        self._clawdrain = clawdrain_detector or ClawdrainDetector()

    def assess_input(
        self,
        text: str,
        *,
        source: str = "user_input",
        has_tool_calls: bool | None = None,
        tool_call_history: list[dict[str, Any]] | None = None,
        tool_outputs: list[str] | None = None,
        context_growth_ratio: float | None = None,
        turns_without_progress: int = 0,
        historical_risk: float = 0.0,
    ) -> AdmissionDecision:
        """Assess a request or retrieved payload before model admission."""

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        token_estimate = estimate_tokens(text)
        if has_tool_calls is None:
            has_tools = bool(tool_call_history or tool_outputs)
        else:
            has_tools = bool(has_tool_calls)
        quant = self._gate.evaluate(
            GatingContext(
                input_tokens=token_estimate,
                has_tool_calls=has_tools,
                historical_risk=_clamp01(historical_risk),
            )
        )

        signals: list[AdmissionSignal] = []
        sanitized = text
        metadata: dict[str, Any] = {
            "source": source,
            "input_tokens_estimate": token_estimate,
            "quant_level": quant.quant_level.value,
        }

        if token_estimate > self.policy.max_input_tokens:
            signals.append(
                AdmissionSignal(
                    name="token_budget",
                    score=1.0,
                    severity="critical",
                    details={
                        "estimated_tokens": token_estimate,
                        "limit": self.policy.max_input_tokens,
                    },
                )
            )
            return AdmissionDecision(
                action=AdmissionAction.BLOCK,
                allowed=False,
                reason="input exceeds admission token budget",
                sanitized_input=sanitized,
                quant_decision=quant,
                signals=signals,
                metadata=metadata,
            )

        scanner = self._scanner
        if source.startswith("tool") and self.policy.strict_tool_result_scanning:
            scanner = self._strict_scanner
        injection = scanner.scan(text, source=source)
        if injection.score > 0.0:
            signals.append(_signal_from_injection(injection))
        if injection.is_injection and self.policy.block_prompt_injection:
            return AdmissionDecision(
                action=AdmissionAction.BLOCK,
                allowed=False,
                reason="prompt-injection scanner blocked admission",
                sanitized_input=sanitized,
                quant_decision=quant,
                signals=signals,
                metadata=metadata,
            )

        exhaustion: ExhaustionResult | None = None
        if tool_call_history or tool_outputs or context_growth_ratio is not None:
            exhaustion = self._clawdrain.detect(
                tool_call_history=tool_call_history,
                tool_outputs=tool_outputs,
                context_growth_ratio=context_growth_ratio,
                turns_without_progress=turns_without_progress,
            )
            if exhaustion.risk_score > 0.0:
                signals.append(_signal_from_clawdrain(exhaustion))
            if exhaustion.is_exhaustion_attack and self.policy.block_clawdrain:
                return AdmissionDecision(
                    action=AdmissionAction.BLOCK,
                    allowed=False,
                    reason="clawdrain detector blocked token-exhaustion pattern",
                    sanitized_input=sanitized,
                    quant_decision=quant,
                    signals=signals,
                    metadata=metadata,
                )

        sanitization: SanitizationResult | None = None
        if self.policy.redact_input_secrets:
            sanitization = self._sanitizer.sanitize(text)
            if sanitization.redaction_count:
                sanitized = sanitization.sanitized_text
                signals.append(_signal_from_sanitizer(sanitization))
                metadata["redaction_count"] = sanitization.redaction_count
                return AdmissionDecision(
                    action=AdmissionAction.REDACT,
                    allowed=True,
                    reason="admission allowed with sensitive spans redacted",
                    sanitized_input=sanitized,
                    quant_decision=quant,
                    signals=signals,
                    metadata=metadata,
                )

        risk = max((signal.score for signal in signals), default=0.0)
        if risk >= self.policy.warning_risk_threshold or quant.quant_level == QuantLevel.THOROUGH:
            return AdmissionDecision(
                action=AdmissionAction.WARN,
                allowed=True,
                reason="admission allowed with elevated safety posture",
                sanitized_input=sanitized,
                quant_decision=quant,
                signals=signals,
                metadata=metadata,
            )

        return AdmissionDecision(
            action=AdmissionAction.ALLOW,
            allowed=True,
            reason="admission checks passed",
            sanitized_input=sanitized,
            quant_decision=quant,
            signals=signals,
            metadata=metadata,
        )

    def assess_tool_result(
        self,
        tool_name: str,
        result: str,
        *,
        tool_call_history: list[dict[str, Any]] | None = None,
        context_growth_ratio: float | None = None,
        turns_without_progress: int = 0,
        historical_risk: float = 0.0,
    ) -> AdmissionDecision:
        """Assess untrusted tool output before it enters memory/context."""

        if not isinstance(tool_name, str):
            raise TypeError(f"tool_name must be str, got {type(tool_name).__name__}")
        return self.assess_input(
            result,
            source=f"tool_result:{tool_name}",
            has_tool_calls=True,
            tool_call_history=tool_call_history,
            tool_outputs=[result],
            context_growth_ratio=context_growth_ratio,
            turns_without_progress=turns_without_progress,
            historical_risk=historical_risk,
        )


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for admission gating."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _signal_from_injection(score: InjectionScore) -> AdmissionSignal:
    severity = "high" if score.is_injection else "medium"
    return AdmissionSignal(
        name="prompt_injection",
        score=score.score,
        severity=severity,
        details={
            "signals": list(score.signals),
            "threshold": score.details.get("threshold"),
            "strict_mode": score.details.get("strict_mode"),
        },
    )


def _signal_from_clawdrain(result: ExhaustionResult) -> AdmissionSignal:
    return AdmissionSignal(
        name="clawdrain",
        score=result.risk_score,
        severity=result.severity,
        details={
            "vectors": [signal.vector.value for signal in result.signals],
            "total_amplification": result.total_amplification,
        },
    )


def _signal_from_sanitizer(result: SanitizationResult) -> AdmissionSignal:
    return AdmissionSignal(
        name="sensitive_data",
        score=min(1.0, 0.25 + 0.15 * result.redaction_count),
        severity="medium",
        details={
            "rules_applied": list(result.rules_applied),
            "redaction_count": result.redaction_count,
        },
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "AdmissionAction",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionSignal",
    "SafetyAdmissionController",
    "estimate_tokens",
]
