"""Clawdrain token-exhaustion attack detector.

Detects tool-call chain amplification patterns identified in OpenClaw research
(2603.00902 -- Clawdrain): malicious agents crafting recursive tool calls that
amplify context by 6-9x per turn, exhausting the context window and causing
cost overruns or denial-of-service.

Attack vectors detected:
  1. Recursive tool-call loops (tool N calls tool N)
  2. SKILL.md prompt bloat (loading multiple skills to inflate context)
  3. Persistent tool-output pollution (tool returns grow unboundedly)
  4. Cron/heartbeat frequency amplification (rapid-fire tool calls)
  5. Behavioral instruction injection (tool returns that instruct the agent
     to keep calling more tools)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src._compat import StrEnum


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class AmplificationVector(StrEnum):
    """Categories of token-exhaustion attack patterns."""

    RECURSIVE_TOOL_LOOP = "recursive_tool_loop"
    SKILL_BLOAT = "skill_bloat"
    OUTPUT_POLLUTION = "output_pollution"
    FREQUENCY_SPIKE = "frequency_spike"
    BEHAVIORAL_INJECTION = "behavioral_injection"
    NESTED_TOOL_CHAIN = "nested_tool_chain"


@dataclass(frozen=True)
class ExhaustionSignal:
    """A single detection signal."""

    vector: AmplificationVector
    severity: str  # "low", "medium", "high", "critical"
    description: str
    amplification_ratio: float  # estimated tokens_out / tokens_in


@dataclass
class ExhaustionResult:
    """Aggregate result of a detection run."""

    is_exhaustion_attack: bool
    signals: list[ExhaustionSignal] = field(default_factory=list)
    total_amplification: float = 0.0
    risk_score: float = 0.0  # 0.0 - 1.0

    @property
    def severity(self) -> str:
        if self.risk_score >= 0.8:
            return "critical"
        if self.risk_score >= 0.6:
            return "high"
        if self.risk_score >= 0.3:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Behavioral instructions hidden in tool outputs that tell the agent to
# keep calling tools (a common Clawdrain technique).
_BEHAVIORAL_PATTERNS = [
    re.compile(
        r"(?:continue|again|repeat|next|keep|more)\s+"
        r"(?:calling|invoking|using)\s+(?:tools?|functions?)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:don't|do not)\s+(?:stop|halt|quit|end|terminate)", re.IGNORECASE),
    re.compile(
        r"(?:call|invoke)\s+(?:tool|function|api)\s+"
        r"(?:again|repeatedly|in a loop)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:repeat|loop)\s+this\s+(?:process|procedure|action)", re.IGNORECASE),
    re.compile(r"for\s+each\s+(?:result|item|entry)\s*,\s*(?:call|invoke)", re.IGNORECASE),
]

# SKILL.md loading patterns that bloat context.
_SKILL_BLOPATTERNS = [
    re.compile(
        r"(?:load|import)\s+(?:skill|plugin|module).{0,100}"
        r"(?:skill|plugin|module)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"SKILL\.md.*SKILL\.md", re.IGNORECASE | re.DOTALL),
]

# Instruction injection in tool outputs.
_INJECTION_PATTERNS = [
    re.compile(
        r"(?:ignore|disregard|forget)\s+(?:previous|above|prior|all)\s+"
        r"(?:instructions|commands|rules|constraints)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:new|updated|revised)\s+instructions?\s*:\s*", re.IGNORECASE),
    re.compile(r"SYSTEM:\s*.*(?:continue|proceed)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ClawdrainDetector:
    """Detects Clawdrain-style token exhaustion attacks."""

    def __init__(
        self,
        *,
        max_tool_calls_per_turn: int = 10,
        max_observation_tokens: int = 4096,
        max_total_amplification: float = 5.0,
        recursive_loop_depth_limit: int = 3,
    ) -> None:
        self._max_tool_calls = max_tool_calls_per_turn
        self._max_obs_tokens = max_observation_tokens
        self._max_amplification = max_total_amplification
        self._loop_limit = recursive_loop_depth_limit

    def detect(
        self,
        *,
        tool_call_history: list[dict[str, Any]] | None = None,
        tool_outputs: list[str] | None = None,
        context_growth_ratio: float | None = None,
        turns_without_progress: int = 0,
    ) -> ExhaustionResult:
        """Run all Clawdrain detection checks.

        Parameters
        ----------
        tool_call_history:
            List of dicts with ``tool_name`` and ``args`` keys.
        tool_outputs:
            List of raw tool output strings.
        context_growth_ratio:
            Estimated ratio of current context size vs. initial size.
        turns_without_progress:
            Number of consecutive turns that didn't advance the task.

        Returns
        -------
        ExhaustionResult with signals and aggregate risk score.
        """
        signals: list[ExhaustionSignal] = []

        # Check 1: Recursive tool-call loop detection
        if tool_call_history:
            loop_signals = self._detect_recursive_loops(tool_call_history)
            signals.extend(loop_signals)

        # Check 2: SKILL.md prompt bloat
        if tool_outputs:
            bloat_signals = self._detect_skill_bloat(tool_outputs)
            signals.extend(bloat_signals)

            # Check 3: Tool-output pollution
            pollution_signals = self._detect_output_pollution(tool_outputs)
            signals.extend(pollution_signals)

            # Check 5: Behavioral instruction injection
            injection_signals = self._detect_behavioral_injection(tool_outputs)
            signals.extend(injection_signals)

        # Check 4: Frequency amplification
        if tool_call_history and turns_without_progress > 0:
            freq_signals = self._detect_frequency_spike(tool_call_history, turns_without_progress)
            signals.extend(freq_signals)

        # Check 6: Context growth
        if context_growth_ratio is not None and context_growth_ratio > self._max_amplification:
            signals.append(
                ExhaustionSignal(
                    vector=AmplificationVector.OUTPUT_POLLUTION,
                    severity="critical" if context_growth_ratio >= 8.0 else "high",
                    description=(
                        f"Context grew {context_growth_ratio:.1f}x beyond baseline "
                        f"(threshold {self._max_amplification}x)"
                    ),
                    amplification_ratio=context_growth_ratio,
                )
            )

        # Aggregate risk score
        risk = self._compute_risk(signals)
        total_amp = max((s.amplification_ratio for s in signals), default=0.0)

        return ExhaustionResult(
            is_exhaustion_attack=risk >= 0.5,
            signals=signals,
            total_amplification=total_amp,
            risk_score=risk,
        )

    # --------------- internal detection methods ---------------

    @staticmethod
    def _detect_recursive_loops(history: list[dict[str, Any]]) -> list[ExhaustionSignal]:
        """Detect when tools call themselves recursively in a chain."""
        signals: list[ExhaustionSignal] = []
        if len(history) < 3:
            return signals

        # Check for repeated tool sequences
        tool_names = [h.get("tool_name", "") for h in history]

        # Pattern: same tool called N times in a row
        for i in range(len(tool_names) - 2):
            if tool_names[i] == tool_names[i + 1] == tool_names[i + 2]:
                signals.append(
                    ExhaustionSignal(
                        vector=AmplificationVector.RECURSIVE_TOOL_LOOP,
                        severity="high",
                        description=(
                            f"Tool '{tool_names[i]}' called {len(tool_names)} times consecutively"
                        ),
                        amplification_ratio=float(len(tool_names)),
                    )
                )
                break

        # Pattern: tool calls itself within its own output
        for i, name in enumerate(tool_names[:-1]):
            if name in str(history[i].get("args", "")):
                signals.append(
                    ExhaustionSignal(
                        vector=AmplificationVector.RECURSIVE_TOOL_LOOP,
                        severity="medium",
                        description=f"Tool '{name}' references itself in arguments",
                        amplification_ratio=2.0,
                    )
                )

        return signals

    @staticmethod
    def _detect_skill_bloat(outputs: list[str]) -> list[ExhaustionSignal]:
        """Detect SKILL.md bloat patterns in tool outputs."""
        signals: list[ExhaustionSignal] = []
        for output in outputs:
            for pattern in _SKILL_BLOPATTERNS:
                if pattern.search(output):
                    ratio = len(output) / max(min(len(o) for o in outputs if o), 1)
                    signals.append(
                        ExhaustionSignal(
                            vector=AmplificationVector.SKILL_BLOAT,
                            severity="medium" if ratio < 5.0 else "high",
                            description="SKILL.md / plugin loading pattern detected in tool output",
                            amplification_ratio=ratio,
                        )
                    )
                    break
        return signals

    def _detect_output_pollution(self, outputs: list[str]) -> list[ExhaustionSignal]:
        """Detect outputs that exceed the observation token limit."""
        signals: list[ExhaustionSignal] = []
        for i, output in enumerate(outputs):
            token_estimate = len(output) // 4  # rough estimate
            if token_estimate > self._max_obs_tokens:
                signals.append(
                    ExhaustionSignal(
                        vector=AmplificationVector.OUTPUT_POLLUTION,
                        severity="high",
                        description=(
                            f"Tool output {i} estimated at {token_estimate} tokens "
                            f"(limit {self._max_obs_tokens})"
                        ),
                        amplification_ratio=token_estimate / max(self._max_obs_tokens, 1),
                    )
                )
        return signals

    def _detect_behavioral_injection(self, outputs: list[str]) -> list[ExhaustionSignal]:
        """Detect instructions hidden in tool outputs that tell agent to keep going."""
        signals: list[ExhaustionSignal] = []
        all_text = " ".join(outputs)

        for patterns, vector, description in [
            (
                _BEHAVIORAL_PATTERNS,
                AmplificationVector.BEHAVIORAL_INJECTION,
                "Behavioral instruction to continue tool calls detected in output",
            ),
            (
                _INJECTION_PATTERNS,
                AmplificationVector.BEHAVIORAL_INJECTION,
                "System instruction override pattern detected in output",
            ),
        ]:
            for pattern in patterns:
                if pattern.search(all_text):
                    signals.append(
                        ExhaustionSignal(
                            vector=vector,
                            severity="high",
                            description=description,
                            amplification_ratio=3.0,
                        )
                    )
                    break
        return signals

    def _detect_frequency_spike(
        self, history: list[dict[str, Any]], turns_without_progress: int
    ) -> list[ExhaustionSignal]:
        """Detect rapid-fire tool calls without task progress."""
        signals: list[ExhaustionSignal] = []
        if len(history) > self._max_tool_calls:
            signals.append(
                ExhaustionSignal(
                    vector=AmplificationVector.FREQUENCY_SPIKE,
                    severity="high" if turns_without_progress > 3 else "medium",
                    description=(
                        f"{len(history)} tool calls in "
                        f"{turns_without_progress} non-progressing turns"
                    ),
                    amplification_ratio=float(len(history)) / max(self._max_tool_calls, 1),
                )
            )
        return signals

    @staticmethod
    def _compute_risk(signals: list[ExhaustionSignal]) -> float:
        """Compute aggregate risk score from detection signals."""
        if not signals:
            return 0.0

        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.5, "critical": 0.8}
        vector_diversity = len({s.vector for s in signals}) / len(AmplificationVector)

        # Weight by severity and diversity
        max_severity = max(severity_weights.get(s.severity, 0.1) for s in signals)
        avg_severity = sum(severity_weights.get(s.severity, 0.1) for s in signals) / len(signals)

        # Composite: max_severity alone can trigger (0.8 -> 0.64, 0.5 -> 0.4).
        # Add amplification bonus: signals with amplification > 3x get a boost.
        max_amp = max((s.amplification_ratio for s in signals), default=0.0)
        amp_bonus = min((max_amp - 2.0) / 8.0, 0.3) if max_amp > 2.0 else 0.0

        risk = (
            0.5 * max_severity
            + 0.3 * (min(vector_diversity, 1.0) * avg_severity)
            + 0.2 * avg_severity
            + amp_bonus
        )
        return min(max(risk, 0.0), 1.0)
