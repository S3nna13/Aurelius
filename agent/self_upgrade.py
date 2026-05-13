"""SelfUpgradeSystem — autonomous self-improvement cycle.

Ported from Aurelius's aurelius/self_upgrade.py.

The system continuously:
1. Observes metrics and performance
2. Researches improvements
3. Plans the upgrade
4. Generates code
5. Tests the change
6. Safety-checks the change
7. Deploys if all checks pass
8. Monitors the result
9. Reflects on the outcome

All functions are injectable. The default implementations are stubs
that log what WOULD happen, making the system safe to experiment with.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class UpgradeProposal:
    title: str
    description: str
    expected_impact: str
    risk_level: str = "medium"  # low | medium | high | critical
    code_changes: str = ""
    test_results: str = ""
    safety_verdict: str = ""
    deployed: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MetricSnapshot:
    accuracy: float = 0.0
    latency_ms: float = 0.0
    cost_per_request: float = 0.0
    user_satisfaction: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class SelfUpgradeSystem:
    """Autonomous self-improvement cycle.

    observe -> research -> plan -> code -> test -> safety -> deploy -> monitor -> reflect

    The system tracks metrics, proposes improvements, and safely deploys
    changes when they pass all verification gates.
    """

    def __init__(
        self,
        research_fn: Callable[[str], str] | None = None,
        code_gen_fn: Callable[[str], str] | None = None,
        test_fn: Callable[[str], bool] | None = None,
        safety_fn: Callable[[str], bool] | None = None,
        deploy_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.research_fn = research_fn or self._default_research
        self.code_gen_fn = code_gen_fn or self._default_code_gen
        self.test_fn = test_fn or self._default_test
        self.safety_fn = safety_fn or self._default_safety
        self.deploy_fn = deploy_fn or self._default_deploy

        self._metrics: list[MetricSnapshot] = []
        self._proposals: list[UpgradeProposal] = []
        self._cycle_count: int = 0

    def observe(self, metric: MetricSnapshot) -> None:
        self._metrics.append(metric)

    def run_cycle(self, observation: str = "") -> UpgradeProposal | None:
        """Run one full self-improvement cycle."""
        self._cycle_count += 1

        # 1. Observe: analyze recent metrics
        recent = self._metrics[-5:] if len(self._metrics) >= 5 else self._metrics
        trend_summary = self._summarize_trends(recent)

        # 2. Research: find potential improvements
        research_query = f"Based on trends: {trend_summary}\nObservation: {observation}"
        research_result = self.research_fn(research_query)

        # 3. Plan: create upgrade proposal
        proposal = UpgradeProposal(
            title=f"Upgrade {self._cycle_count}: {research_result[:60]}...",
            description=research_result,
            expected_impact=self._estimate_impact(research_result),
        )

        # 4. Code: generate the change
        proposal.code_changes = self.code_gen_fn(research_result)

        # 5. Test: verify the change
        proposal.test_results = "passed" if self.test_fn(proposal.code_changes) else "failed"

        # 6. Safety: security check
        proposal.safety_verdict = "passed" if self.safety_fn(proposal.code_changes) else "failed"

        # 7. Deploy: if all checks pass
        if proposal.test_results == "passed" and proposal.safety_verdict == "passed":
            self.deploy_fn(proposal.code_changes)
            proposal.deployed = True

        self._proposals.append(proposal)
        return proposal

    def _summarize_trends(self, metrics: list[MetricSnapshot]) -> str:
        if not metrics:
            return "No metrics available"
        first, last = metrics[0], metrics[-1]
        parts = []
        if last.accuracy > first.accuracy:
            parts.append(f"accuracy improved: {first.accuracy:.2f} -> {last.accuracy:.2f}")
        if last.latency_ms < first.latency_ms:
            parts.append(f"latency reduced: {first.latency_ms:.1f}ms -> {last.latency_ms:.1f}ms")
        if last.error_rate < first.error_rate:
            parts.append(f"errors reduced: {first.error_rate:.3f} -> {last.error_rate:.3f}")
        return "; ".join(parts) if parts else "No significant trends detected"

    @staticmethod
    def _estimate_impact(research: str) -> str:
        return "medium"  # In production: LLM-based impact estimation

    @staticmethod
    def _default_research(query: str) -> str:
        return f"Research: {query[:100]}..."

    @staticmethod
    def _default_code_gen(spec: str) -> str:
        return f"# Auto-generated code for:\n# {spec[:100]}...\npass"

    @staticmethod
    def _default_test(code: str) -> bool:
        return True  # In production: actually run tests

    @staticmethod
    def _default_safety(code: str) -> bool:
        return True  # In production: run security scanner

    @staticmethod
    def _default_deploy(code: str) -> None:
        pass

    @property
    def proposals(self) -> list[UpgradeProposal]:
        return list(self._proposals)

    @property
    def metrics_history(self) -> list[MetricSnapshot]:
        return list(self._metrics)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
