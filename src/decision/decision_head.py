"""Aurelius v2 DecisionHead — decisive action routing without loops.

Implements the DAIES decision scoring system:
  score = utility + information_gain - risk - cost

Where cost includes: latency, tokens, dollars, watts, memory pressure,
user interruption, side-effect risk, context pollution risk.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class DecisiveAction(enum.StrEnum):
    ANSWER_NOW = "answer_now"
    USE_TOOL = "use_tool"
    USE_SKILL = "use_skill"
    USE_COMPUTER = "use_computer"
    SEARCH_MEMORY = "search_memory"
    ASK_USER = "ask_user"
    DELEGATE = "delegate"
    ESCALATE_MODEL = "escalate_model"
    REFUSE_SAFE = "refuse_safe"
    FINALIZE_PARTIAL = "finalize_partial"


@dataclass
class DecisionScore:
    """Score for a potential action."""
    action: DecisiveAction
    utility: float = 0.0
    information_gain: float = 0.0
    risk: float = 0.0
    cost: float = 0.0
    confidence: float = 0.0
    estimated_latency_ms: int = 0
    token_cost: int = 0

    @property
    def total(self) -> float:
        return self.utility + self.information_gain - self.risk - self.cost


@dataclass
class DecisionOutput:
    """Result of making a decision."""
    action: DecisiveAction
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    fallback_action: DecisiveAction | None = None


class DecisionHead:
    """Makes decisive routing decisions without looping.

    Scoring formula:
        score = utility + information_gain - risk - cost

    Cost factors:
        - latency
        - tokens
        - dollars/cloud cost
        - watts (energy)
        - memory pressure
        - user interruption cost
        - side-effect risk
        - context pollution risk
    """

    def __init__(self, max_loop_penalty: float = 1.0) -> None:
        self._loop_count = 0
        self._max_loop_penalty = max_loop_penalty

    def decide(
        self,
        candidate_scores: list[DecisionScore],
        context: dict[str, Any] | None = None,
    ) -> DecisionOutput:
        """Select the best action from scored candidates.

        Runtime rules:
        - If confidence high: answer or act
        - If native skill fits better than raw tool use: use skill
        - If evidence needed and safe tool exists: use tool
        - If ambiguity changes intent: ask user
        - If local model/hardware insufficient: escalate
        - If unsafe: refuse or ask for confirmation
        - If repeated no-progress: finalize partial with next step
        """
        if not candidate_scores:
            return DecisionOutput(
                action=DecisiveAction.REFUSE_SAFE,
                reasoning="No viable action candidates",
                confidence=0.0,
            )

        # Apply loop penalty
        loop_penalty = min(self._loop_count * 0.2, self._max_loop_penalty)
        for cs in candidate_scores:
            cs.total -= loop_penalty

        best = max(candidate_scores, key=lambda s: s.total)

        # Minimum confidence threshold — ask user if too uncertain
        if best.confidence < 0.3:
            return DecisionOutput(
                action=DecisiveAction.ASK_USER,
                details={"original_action": best.action.value, "confidence": best.confidence},
                confidence=best.confidence,
                reasoning=f"Confidence too low ({best.confidence:.2f}) for {best.action.value}",
            )

        # High confidence: act directly
        self._loop_count += 1
        return DecisionOutput(
            action=best.action,
            details={"score": best.total},
            confidence=best.confidence,
        )

    def reset(self) -> None:
        """Reset loop count for new conversation."""
        self._loop_count = 0

    def compute_scores(
        self,
        request: dict[str, Any],
        available_tools: list[str],
        available_skills: list[str],
        memory_available: bool = True,
        cua_available: bool = False,
        hardware_adequate: bool = True,
    ) -> list[DecisionScore]:
        """Compute scores for all possible actions given current context.

        Returns a list of DecisionScore objects for each viable candidate action.
        """
        scores: list[DecisionScore] = []

        # Answer now — default candidate
        has_query = bool(request.get("prompt", "").strip())
        scores.append(DecisionScore(
            action=DecisiveAction.ANSWER_NOW,
            utility=0.6 if has_query else 0.1,
            information_gain=0.3,
            risk=0.0,
            cost=0.2,
            confidence=0.7 if has_query else 0.2,
            estimated_latency_ms=500,
            token_cost=512,
        ))

        # Use tool
        if available_tools:
            scores.append(DecisionScore(
                action=DecisiveAction.USE_TOOL,
                utility=0.8,
                information_gain=0.7,
                risk=0.2,
                cost=0.3,
                confidence=0.6,
                estimated_latency_ms=1000,
                token_cost=1024,
            ))

        # Use skill
        if available_skills:
            scores.append(DecisionScore(
                action=DecisiveAction.USE_SKILL,
                utility=0.9,
                information_gain=0.6,
                risk=0.1,
                cost=0.2,
                confidence=0.7,
                estimated_latency_ms=800,
                token_cost=768,
            ))

        # Use computer (CUA)
        if cua_available:
            scores.append(DecisionScore(
                action=DecisiveAction.USE_COMPUTER,
                utility=0.9,
                information_gain=0.8,
                risk=0.5,
                cost=0.6,
                confidence=0.5,
                estimated_latency_ms=3000,
                token_cost=2048,
            ))

        # Search memory
        if memory_available:
            scores.append(DecisionScore(
                action=DecisiveAction.SEARCH_MEMORY,
                utility=0.5,
                information_gain=0.5,
                risk=0.0,
                cost=0.1,
                confidence=0.6,
                estimated_latency_ms=200,
                token_cost=256,
            ))

        # Ask user
        scores.append(DecisionScore(
            action=DecisiveAction.ASK_USER,
            utility=0.3,
            information_gain=0.9,
            risk=0.0,
            cost=0.0,
            confidence=1.0,
            estimated_latency_ms=0,
            token_cost=0,
        ))

        # Escalate model
        if not hardware_adequate:
            scores.append(DecisionScore(
                action=DecisiveAction.ESCALATE_MODEL,
                utility=0.7,
                information_gain=0.4,
                risk=0.3,
                cost=0.8,
                confidence=0.5,
                estimated_latency_ms=5000,
                token_cost=4096,
            ))

        return scores
