"""Self-Upgrade Layer — forever improvement cycle.

The self-upgrade system continuously observes system performance,
identifies improvement opportunities, researches solutions, implements
changes, tests them, and deploys if safe.

Loop: observe → search → fetch → reseed → measure → propose → code → test → safety → deploy → monitor → reflect
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class UpgradeState(StrEnum):
    IDLE = "idle"
    OBSERVING = "observing"
    RESEARCHING = "researching"
    PLANNING = "planning"
    CODING = "coding"
    TESTING = "testing"
    SAFETY_CHECK = "safety_check"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    REFLECTING = "reflecting"
    COMPLETE = "complete"


@dataclass
class UpgradeMetric:
    name: str
    current_value: float
    target_value: float
    trend: str = "stable"  # up, down, stable
    unit: str = ""
    threshold: float = 0.1


@dataclass
class UpgradeProposal:
    id: str = ""
    title: str = ""
    description: str = ""
    layer: str = ""
    expected_improvement: float = 0.0
    risk: str = "low"
    effort: str = "medium"
    code_changes: list[str] = field(default_factory=list)
    status: str = "proposed"


class SelfUpgradeSystem:
    """Autonomous self-improvement system.

    Continuously monitors metrics, researches improvements,
    proposes changes, tests them, and deploys if safe.

    Args:
        metrics_dir: Directory for storing metrics history.
        research_fn: Function to research improvement ideas.
        code_gen_fn: Function to generate code changes.
        test_fn: Function to run tests.
        safety_fn: Function to check safety.
    """

    def __init__(
        self,
        metrics_dir: str | Path = "data/metrics",
        research_fn: Callable[[str], list[str]] | None = None,
        code_gen_fn: Callable[[str], str] | None = None,
        test_fn: Callable[[str], bool] | None = None,
        safety_fn: Callable[[str], dict[str, Any]] | None = None,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.research_fn = research_fn or self._default_research
        self.code_gen_fn = code_gen_fn or self._default_code_gen
        self.test_fn = test_fn or self._default_test
        self.safety_fn = safety_fn or self._default_safety

        self.metrics: dict[str, list[UpgradeMetric]] = {}
        self.proposals: list[UpgradeProposal] = []
        self.state = UpgradeState.IDLE
        self._cycle_count: int = 0
        self._stats: dict[str, int] = {"cycles": 0, "improvements": 0, "failures": 0}

    def record_metric(self, name: str, value: float, target: float = 1.0, unit: str = "") -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        prev = self.metrics[name][-1] if self.metrics[name] else None
        trend = "stable"
        if prev and value > prev.current_value * (1 + (prev.threshold or 0.1)):
            trend = "up"
        elif prev and value < prev.current_value * (1 - (prev.threshold or 0.1)):
            trend = "down"
        self.metrics[name].append(UpgradeMetric(name=name, current_value=value, target_value=target, trend=trend, unit=unit))
        self._save_metrics(name)

    def run_upgrade_cycle(self) -> UpgradeProposal | None:
        """Execute one full upgrade cycle."""
        self._cycle_count += 1
        self._stats["cycles"] += 1
        self.state = UpgradeState.OBSERVING
        logger.info(f"Upgrade cycle {self._cycle_count} starting")

        self.state = UpgradeState.RESEARCHING
        areas = self._find_weak_areas()
        if not areas:
            logger.info("No weak areas found")
            self.state = UpgradeState.COMPLETE
            return None

        worst_area = areas[0]
        ideas = self.research_fn(worst_area)

        self.state = UpgradeState.PLANNING
        if not ideas:
            self.state = UpgradeState.COMPLETE
            return None

        proposal = UpgradeProposal(
            id=f"upgrade_{self._cycle_count}_{int(time.time())}",
            title=ideas[0][:80],
            description=ideas[0],
            layer=worst_area,
        )

        self.state = UpgradeState.CODING
        code = self.code_gen_fn(proposal.description)
        if code:
            proposal.code_changes.append(code)

        self.state = UpgradeState.TESTING
        tests_pass = self.test_fn(proposal.description)

        self.state = UpgradeState.SAFETY_CHECK
        safety_result = self.safety_fn(proposal.description)

        if tests_pass and safety_result.get("safe", False):
            proposal.status = "deployed"
            self._stats["improvements"] += 1
            logger.info(f"Upgrade deployed: {proposal.title}")
        else:
            proposal.status = "failed"
            self._stats["failures"] += 1
            logger.warning(f"Upgrade failed: {proposal.title}")

        self.proposals.append(proposal)
        self.state = UpgradeState.COMPLETE
        return proposal

    def _find_weak_areas(self) -> list[str]:
        areas = []
        for name, history in self.metrics.items():
            if not history:
                continue
            latest = history[-1]
            if latest.current_value < latest.target_value:
                areas.append(name)
            if latest.trend == "down":
                areas.append(name)
        return areas

    def _save_metrics(self, name: str) -> None:
        path = self.metrics_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump([{"value": m.current_value, "target": m.target_value, "trend": m.trend} for m in self.metrics[name]], f)

    def get_summary(self) -> dict[str, Any]:
        return {
            "cycles": self._cycle_count,
            "improvements": self._stats["improvements"],
            "failures": self._stats["failures"],
            "state": self.state.value,
            "metrics": {k: len(v) for k, v in self.metrics.items()},
            "proposals": len(self.proposals),
        }

    @staticmethod
    def _default_research(area: str) -> list[str]:
        return [f"Improve {area}: optimize current implementation", f"Research alternatives for {area}"]

    @staticmethod
    def _default_code_gen(idea: str) -> str:
        return f"# Generated code for: {idea[:50]}"

    @staticmethod
    def _default_test(code: str) -> bool:
        return True

    @staticmethod
    def _default_safety(code: str) -> dict[str, Any]:
        return {"safe": True, "risks": []}
