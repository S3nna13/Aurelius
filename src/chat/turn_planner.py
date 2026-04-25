"""Turn planner — decide what action to take given the conversation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TurnAction(str, Enum):
    RESPOND = "respond"
    USE_TOOL = "use_tool"
    ASK_CLARIFICATION = "ask_clarification"
    SUMMARIZE = "summarize"
    DEFER = "defer"


@dataclass
class TurnPlan:
    action: TurnAction
    rationale: str
    confidence: float
    metadata: dict = field(default_factory=dict)


# Keywords that trigger each non-default action
_TOOL_KEYWORDS = ("search", "calculate", "run", "execute")
_DEFER_KEYWORDS = ("later", "remind")
_HISTORY_SUMMARIZE_THRESHOLD = 20


class TurnPlanner:
    """Lightweight heuristic turn planner."""

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        conversation_history: list[dict],
        last_user_msg: str,
    ) -> TurnPlan:
        """Return a TurnPlan for the current turn.

        Priority order:
        1. SUMMARIZE  — if history is long
        2. DEFER      — if message mentions deferral
        3. USE_TOOL   — if message contains tool-trigger keywords
        4. ASK_CLARIFICATION — if message is very short or a bare question
        5. RESPOND    — default
        """
        msg_lower = last_user_msg.lower().strip()

        # 1. Long history
        if len(conversation_history) > _HISTORY_SUMMARIZE_THRESHOLD:
            return TurnPlan(
                action=TurnAction.SUMMARIZE,
                rationale=(
                    f"Conversation has {len(conversation_history)} turns, "
                    "exceeding the summarization threshold."
                ),
                confidence=0.85,
                metadata={"history_len": len(conversation_history)},
            )

        # 2. Defer keywords
        if any(kw in msg_lower for kw in _DEFER_KEYWORDS):
            return TurnPlan(
                action=TurnAction.DEFER,
                rationale="User message contains a deferral keyword.",
                confidence=0.80,
                metadata={"matched_keyword": next(
                    kw for kw in _DEFER_KEYWORDS if kw in msg_lower
                )},
            )

        # 3. Tool keywords
        if any(kw in msg_lower for kw in _TOOL_KEYWORDS):
            matched = next(kw for kw in _TOOL_KEYWORDS if kw in msg_lower)
            return TurnPlan(
                action=TurnAction.USE_TOOL,
                rationale=f"User message contains tool-trigger keyword: '{matched}'.",
                confidence=0.90,
                metadata={"trigger_keyword": matched},
            )

        # 4. Clarification — very short OR ends with "?"
        if len(last_user_msg.strip()) < 10 or last_user_msg.strip().endswith("?"):
            return TurnPlan(
                action=TurnAction.ASK_CLARIFICATION,
                rationale=(
                    "User message is ambiguous (too short or an open question)."
                ),
                confidence=0.75,
                metadata={"msg_len": len(last_user_msg.strip())},
            )

        # 5. Default
        return TurnPlan(
            action=TurnAction.RESPOND,
            rationale="No special condition matched; generating a direct response.",
            confidence=0.95,
            metadata={},
        )

    # ------------------------------------------------------------------
    # Response-length estimation
    # ------------------------------------------------------------------

    def estimate_response_length(
        self,
        plan: TurnPlan,
        context_tokens: int,
    ) -> int:
        """Estimate the expected response length in tokens.

        Heuristic scales:
        - RESPOND       → moderate reply relative to context
        - USE_TOOL      → short (tool handles the heavy lifting)
        - ASK_CLARIFICATION → short (one or two sentences)
        - SUMMARIZE     → large (summarising many turns)
        - DEFER         → very short (acknowledgement only)
        """
        base_estimates: dict[TurnAction, int] = {
            TurnAction.RESPOND: max(64, min(context_tokens // 4, 512)),
            TurnAction.USE_TOOL: 64,
            TurnAction.ASK_CLARIFICATION: 32,
            TurnAction.SUMMARIZE: max(128, min(context_tokens // 2, 1024)),
            TurnAction.DEFER: 24,
        }
        return base_estimates.get(plan.action, 128)
