"""Action scoring module for the Aurelius agent surface.

Scores candidate actions using a weighted linear combination:
    score = λu·U + λi·I + λr·R + λc·C

where U=utility, I=info_gain, R=risk, C=cost and the lambdas can be
negative to penalise undesirable dimensions (risk, cost by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScoreLambdas:
    """Weighting coefficients for each scoring dimension."""

    utility: float = 1.0
    info_gain: float = 0.5
    risk: float = -1.5  # negative: penalizes risk
    cost: float = -0.2  # negative: penalizes cost


@dataclass
class ActionScore:
    """Raw dimension scores plus the weighted total."""

    utility: float
    info_gain: float
    risk: float
    cost: float
    total: float


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ActionScorer:
    """Score and rank candidate actions against the current agent state."""

    HIGH_RISK = frozenset({"delete", "write_file", "execute", "shell", "deploy", "drop", "rm"})
    MED_RISK = frozenset({"read_file", "api_call", "network", "http"})
    INFO_ACTIONS = frozenset({"search", "retrieve", "query", "fetch", "lookup"})

    # ------------------------------------------------------------------
    # Estimation helpers
    # ------------------------------------------------------------------

    def estimate_utility(self, action: dict, state: dict) -> float:
        """Return 1.0 if action type matches a goal, 0.5 if partial, 0.1 default."""
        action_type: str = str(action.get("type", "")).lower()
        goals: list[str] = [str(g).lower() for g in state.get("goals", [])]

        if not goals:
            return 0.1

        # Full match: action type is exactly one of the goals
        if action_type in goals:
            return 1.0

        # Partial match: action type appears as a substring of any goal or vice-versa
        for goal in goals:
            if action_type in goal or goal in action_type:
                return 0.5

        return 0.1

    def estimate_info_gain(self, action: dict, state: dict) -> float:  # noqa: ARG002
        """Return information-gain score based on action type."""
        action_type: str = str(action.get("type", "")).lower()

        if action_type in self.INFO_ACTIONS:
            return 1.0
        if action_type in {"write", "create"}:
            return 0.3
        if action_type == "wait":
            return 0.05
        return 0.2

    def estimate_risk(
        self,
        action: dict,
        state: dict,  # noqa: ARG002
        policy: dict | None = None,
    ) -> float:
        """Return risk score; if action type is in policy deny_list return 1.0."""
        action_type: str = str(action.get("type", "")).lower()

        if policy is not None:
            deny_list: list[str] = [str(d).lower() for d in policy.get("deny_list", [])]
            if action_type in deny_list:
                return 1.0

        if action_type in self.HIGH_RISK:
            return 0.8
        if action_type in self.MED_RISK:
            return 0.4
        return 0.1

    def estimate_cost(self, action: dict) -> float:
        """Return cost score based on estimated_tokens, clamped to [0, 1]."""
        tokens: float = float(action.get("estimated_tokens", 0))
        cost = tokens / 1000.0
        return max(0.0, min(1.0, cost))

    # ------------------------------------------------------------------
    # Scoring & ranking
    # ------------------------------------------------------------------

    def score(
        self,
        action: dict,
        state: dict,
        lambdas: ScoreLambdas | None = None,
        policy: dict | None = None,
    ) -> ActionScore:
        """Compute a weighted ActionScore for a single action."""
        if lambdas is None:
            lambdas = ScoreLambdas()

        u = self.estimate_utility(action, state)
        i = self.estimate_info_gain(action, state)
        r = self.estimate_risk(action, state, policy)
        c = self.estimate_cost(action)

        total = lambdas.utility * u + lambdas.info_gain * i + lambdas.risk * r + lambdas.cost * c

        return ActionScore(utility=u, info_gain=i, risk=r, cost=c, total=total)

    def rank_actions(
        self,
        actions: list[dict],
        state: dict,
        lambdas: ScoreLambdas | None = None,
    ) -> list[tuple[dict, ActionScore]]:
        """Return actions sorted by descending total score."""
        scored = [(a, self.score(a, state, lambdas)) for a in actions]
        scored.sort(key=lambda pair: pair[1].total, reverse=True)
        return scored


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, Any] = {
    "action_scorer": ActionScorer(),
}
