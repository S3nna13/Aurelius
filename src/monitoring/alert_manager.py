"""Alert manager: threshold-based rules, firing/resolution, alert history."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class AlertSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AlertRule:
    name: str
    metric_name: str
    threshold: float
    severity: AlertSeverity
    comparison: str = ">"
    message_template: str = ""


def _auto_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Alert:
    rule_name: str
    severity: AlertSeverity
    metric_value: float
    message: str
    fired_at: str
    resolved: bool = False
    id: str = field(default_factory=_auto_id)


def _compare(value: float, threshold: float, op: str) -> bool:
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    return False


class AlertManager:
    def __init__(self) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._history: list[Alert] = []
        # track which rules are currently firing (have an unresolved alert)
        self._firing: set[str] = set()

    def add_rule(self, rule: AlertRule) -> None:
        self._rules[rule.name] = rule

    def evaluate(self, metric_name: str, value: float) -> list[Alert]:
        newly_fired: list[Alert] = []
        for rule in self._rules.values():
            if rule.metric_name != metric_name:
                continue
            if rule.name in self._firing:
                # already active, don't re-fire
                continue
            if _compare(value, rule.threshold, rule.comparison):
                msg = rule.message_template or (
                    f"{metric_name} {rule.comparison} {rule.threshold} (value={value})"
                )
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    metric_value=value,
                    message=msg,
                    fired_at=datetime.now(UTC).isoformat(),
                )
                self._history.append(alert)
                self._firing.add(rule.name)
                newly_fired.append(alert)
        return newly_fired

    def resolve(self, rule_name: str) -> bool:
        # Find the most recent unresolved alert for this rule
        for alert in reversed(self._history):
            if alert.rule_name == rule_name and not alert.resolved:
                alert.resolved = True
                self._firing.discard(rule_name)
                return True
        return False

    def active_alerts(self) -> list[Alert]:
        return [a for a in self._history if not a.resolved]

    def alert_history(self) -> list[Alert]:
        return list(self._history)

    def rules(self) -> list[AlertRule]:
        return list(self._rules.values())


ALERT_MANAGER = AlertManager()
