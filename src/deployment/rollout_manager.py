"""Canary and blue/green rollout manager: traffic splitting, health gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RolloutStrategy(str, Enum):
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"


@dataclass
class RolloutStage:
    name: str
    traffic_pct: float
    min_healthy_pct: float = 95.0
    duration_seconds: int = 300


@dataclass
class RolloutPlan:
    strategy: RolloutStrategy
    stages: list[RolloutStage]
    rollback_on_error: bool = True


class RolloutManager:
    def __init__(self, plan: RolloutPlan) -> None:
        self._plan = plan
        self._stage_index: int = 0

    def current_stage(self) -> RolloutStage | None:
        if not self._plan.stages:
            return None
        if self._stage_index >= len(self._plan.stages):
            return None
        return self._plan.stages[self._stage_index]

    def advance(self) -> bool:
        if self._stage_index >= len(self._plan.stages) - 1:
            return False
        self._stage_index += 1
        return True

    def rollback(self) -> None:
        self._stage_index = 0

    def traffic_split(self) -> dict[str, float]:
        stage = self.current_stage()
        if stage is None:
            return {}

        strategy = self._plan.strategy

        if strategy == RolloutStrategy.BLUE_GREEN:
            is_final = self._stage_index == len(self._plan.stages) - 1
            if is_final:
                return {"blue": 0.0, "green": 1.0}
            else:
                return {"blue": 1.0, "green": 0.0}

        # CANARY and ROLLING
        canary_fraction = stage.traffic_pct / 100.0
        stable_fraction = 1.0 - canary_fraction
        return {"stable": stable_fraction, "canary": canary_fraction}

    def progress(self) -> dict:
        stage = self.current_stage()
        return {
            "stage_index": self._stage_index,
            "total_stages": len(self._plan.stages),
            "strategy": self._plan.strategy.value,
            "current_stage_name": stage.name if stage is not None else "",
        }


# Pre-built plans
CANARY_3_STAGE = RolloutPlan(
    strategy=RolloutStrategy.CANARY,
    stages=[
        RolloutStage("5pct", 5.0),
        RolloutStage("25pct", 25.0),
        RolloutStage("100pct", 100.0),
    ],
)

ROLLOUT_REGISTRY: dict[str, RolloutPlan] = {
    "canary_3stage": CANARY_3_STAGE,
    "blue_green": RolloutPlan(
        strategy=RolloutStrategy.BLUE_GREEN,
        stages=[RolloutStage("cutover", 100.0)],
    ),
}
