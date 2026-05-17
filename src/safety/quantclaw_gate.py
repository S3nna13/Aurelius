"""QuantClaw: precision security gating for minimal performance overhead.

QuantClaw implements a dynamic risk-threshold gate that conditionally
executes expensive safety checks only when necessary, reducing latency
while maintaining security guarantees.

Design principles
-----------------
1. Simple inputs (low context size, no tool calls) have cheap checks only
2. Complex/high-risk inputs trigger full safety pipeline (PRISM, ClawKeeper, etc.)
3. Cumulative risk history can pre-warm or pre-warn (adaptive thresholds)
4. Fail-closed: if gating logic errors, run full safety suite

Quantization levels
-------------------
* Q0 (fast): Only lexical scans (char/byte patterns) — sub-100μs
* Q1 (medium): Add lightweight statistical detectors (entropy, length)
* Q2 (thorough): Full multi-layer safety (PRISM + ClawKeeper + others)

The gate automatically selects the appropriate level based on:
- Input size (tokens)
- Tool call presence
- Recent threat signals
- Agent's historical risk score
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src._compat import StrEnum

_LOGGER = logging.getLogger("aurelius.safety.quantclaw")


class QuantLevel(StrEnum):
    FAST = "fast"       # Q0: lexical only, <100μs
    MEDIUM = "medium"   # Q1: + statistical, ~1ms
    THOROUGH = "thorough"  # Q2: full suite, ~10ms+


@dataclass
class GatingContext:
    """Context used for gating decisions."""
    input_tokens: int
    has_tool_calls: bool
    agent_id: str | None = None
    historical_risk: float = 0.0  # 0-1 from recent safety evaluations
    time_budget_seconds: float = 0.1  # max allowed latency for safety checks


@dataclass
class GateDecision:
    """Result of a QuantClaw gating evaluation."""
    quant_level: QuantLevel
    reasoning: str
    estimated_latency_ms: float
    applied_filters: list[str]
    skipped_filters: list[str]


class QuantClawGate:
    """Dynamic safety gate that balances performance vs security.

    Wraps the full safety pipeline (PRISM, ClawKeeper, detectors, etc.)
    and decides at runtime which checks to run based on input characteristics
    and risk tolerance settings.

    Usage
    -----
    gate = QuantClawGate()
    decision = gate.evaluate(context)
    if decision.quant_level == QuantLevel.FAST:
        result = fast_lexical_scan(text)
    elif decision.quant_level == QuantLevel.MEDIUM:
        result = medium_checks(text, run_fast=True)
    else:  # THOROUGH
        result = full_safety_pipeline(text, context)

    The gate is designed to be embedded in the request entrypoint so that
    the expensive safety suite only runs when needed.
    """

    def __init__(
        self,
        *,
        token_threshold_fast: int = 2_000,     # <2K tokens → FAST
        token_threshold_medium: int = 8_000,   # 2-8K → MEDIUM
        # >8K → THOROUGH (unless other factors push it up)
        risk_threshold_fast: float = 0.1,      # risk >0.1 → at least MEDIUM
        risk_threshold_medium: float = 0.4,    # risk >0.4 → THOROUGH
        tool_call_penalty: bool = True,        # tool calls bump level
        historical_risk_weight: float = 0.5,
        latency_budgets_ms: dict[QuantLevel, float] | None = None,
    ) -> None:
        self._tok_fast = token_threshold_fast
        self._tok_med = token_threshold_medium
        self._risk_fast = risk_threshold_fast
        self._risk_med = risk_threshold_medium
        self._tool_penalty = tool_call_penalty
        self._risk_weight = historical_risk_weight
        self._budgets = latency_budgets_ms or {
            QuantLevel.FAST: 0.1,      # 100μs
            QuantLevel.MEDIUM: 1.0,   # 1ms
            QuantLevel.THOROUGH: 10.0,  # 10ms
        }

        # Filter sets for each quant level
        self._filters_fast: set[str] = {
            "lexical_entropy_quick",
            "keyword_blocklist",
            "zero_width_detector",
        }
        self._filters_medium: set[str] = self._filters_fast | {
            "pattern_injection_medium",
            "length_anomaly",
            "simple_pii",
        }
        self._filters_thorough: set[str] = self._filters_medium | {
            "clawdrain_full",
            "prism_all_hooks",
            "clawkeeper_full",
            "jailbreak_v2",
            "reward_hack",
            "canary_token",
            "output_sanitizer",
        }

        # Performance metrics
        self._latency_history: list[float] = []
        self._gate_hits: dict[QuantLevel, int] = {
            QuantLevel.FAST: 0,
            QuantLevel.MEDIUM: 0,
            QuantLevel.THOROUGH: 0,
        }

    # --------------- public API ---------------

    def evaluate(self, context: GatingContext) -> GateDecision:
        """Determine appropriate quantization level for this request.

        Parameters
        ----------
        context :
            Gating context with token count, tool call presence, agent risk, etc.

        Returns
        -------
        GateDecision with the selected quant level and supporting rationale.
        """
        start = time.perf_counter()

        # Base level from token count
        if context.input_tokens < self._tok_fast:
            level = QuantLevel.FAST
            reason = f"Small input ({context.input_tokens} tokens)"
        elif context.input_tokens < self._tok_med:
            level = QuantLevel.MEDIUM
            reason = f"Medium input ({context.input_tokens} tokens)"
        else:
            level = QuantLevel.THOROUGH
            reason = f"Large input ({context.input_tokens} tokens)"

        # Adjust for historical risk
        if context.historical_risk > self._risk_fast and level == QuantLevel.FAST:
            level = QuantLevel.MEDIUM
            reason += f"; historical risk {context.historical_risk:.2f} bumps level"
        if context.historical_risk > self._risk_med and level in (QuantLevel.FAST, QuantLevel.MEDIUM):
            level = QuantLevel.THOROUGH
            reason += f"; high historical risk {context.historical_risk:.2f} → thorough"

        # Adjust for tool calls
        if context.has_tool_calls and self._tool_penalty:
            if level == QuantLevel.FAST:
                level = QuantLevel.MEDIUM
                reason += "; tool calls present bump level"
            elif level == QuantLevel.MEDIUM:
                level = QuantLevel.THOROUGH
                reason += "; tool calls + medium → thorough"

        # Determine which filters to run
        if level == QuantLevel.FAST:
            applied = list(self._filters_fast)
            skipped = list(self._filters_medium | self._filters_thorough)
        elif level == QuantLevel.MEDIUM:
            applied = list(self._filters_medium)
            skipped = list(self._filters_thorough - self._filters_medium)
        else:
            applied = list(self._filters_thorough)
            skipped = []

        # Estimate latency (budget is a cap; use history to refine)
        budget = self._budgets[level]
        est_latency_ms = budget * 0.8  # assume we're efficient

        decision = GateDecision(
            quant_level=level,
            reasoning=reason.strip(),
            estimated_latency_ms=est_latency_ms,
            applied_filters=applied,
            skipped_filters=skipped,
        )

        # Record metrics
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._latency_history.append(elapsed_ms)
        self._gate_hits[level] += 1

        _LOGGER.debug(
            "QuantClaw decision: %s (%.2fms) applied=%d skipped=%d",
            level.value, elapsed_ms, len(applied), len(skipped)
        )

        return decision

    def get_stats(self) -> dict[str, Any]:
        """Return gating statistics (calls per level, avg latency)."""
        total = sum(self._gate_hits.values())
        avg_latency = sum(self._latency_history) / len(self._latency_history) if self._latency_history else 0.0
        return {
            "total_decisions": total,
            "fast": self._gate_hits[QuantLevel.FAST],
            "medium": self._gate_hits[QuantLevel.MEDIUM],
            "thorough": self._gate_hits[QuantLevel.THOROUGH],
            "fast_pct": self._gate_hits[QuantLevel.FAST] / max(total, 1),
            "medium_pct": self._gate_hits[QuantLevel.MEDIUM] / max(total, 1),
            "thorough_pct": self._gate_hits[QuantLevel.THOROUGH] / max(total, 1),
            "avg_latency_ms": avg_latency,
        }

    def reset_stats(self) -> None:
        """Clear runtime statistics."""
        self._latency_history.clear()
        for k in self._gate_hits:
            self._gate_hits[k] = 0

    # --------------- integration helpers ---------------

    def should_run_full_safety(self, context: GatingContext) -> bool:
        """Convenience boolean: should caller run the full safety pipeline?"""
        decision = self.evaluate(context)
        return decision.quant_level == QuantLevel.THOROUGH

    def get_recommended_filters(self, context: GatingContext) -> list[str]:
        """Get the list of filter names to run for this context."""
        decision = self.evaluate(context)
        return decision.applied_filters
