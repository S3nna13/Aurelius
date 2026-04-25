from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class AlertSeverity(Enum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


class AlertStatus(Enum):
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    status: AlertStatus = AlertStatus.FIRING
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    value: float | None = None

    def resolve(self) -> None:
        self.status = AlertStatus.RESOLVED
        self.timestamp = time.time()


ConditionFn = Callable[[], tuple[bool, str, float | None]]


@dataclass
class AlertRule:
    name: str
    severity: AlertSeverity
    condition: ConditionFn
    cooldown_seconds: float = 300.0
    _last_fired: float = 0.0

    def evaluate(self) -> Alert | None:
        now = time.time()
        if now - self._last_fired < self.cooldown_seconds:
            return None
        triggered, message, value = self.condition()
        if triggered:
            self._last_fired = now
            return Alert(
                name=self.name,
                severity=self.severity,
                message=message,
                value=value,
            )
        return None


class AlertManager:
    def __init__(self) -> None:
        self._rules: list[AlertRule] = []
        self._alerts: list[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def evaluate_all(self) -> list[Alert]:
        new_alerts: list[Alert] = []
        for rule in self._rules:
            alert = rule.evaluate()
            if alert is not None:
                self._alerts.append(alert)
                new_alerts.append(alert)
        return new_alerts

    def active_alerts(self) -> list[Alert]:
        return [a for a in self._alerts if a.status == AlertStatus.FIRING]

    def resolve_by_name(self, name: str) -> None:
        for alert in self._alerts:
            if alert.name == name and alert.status == AlertStatus.FIRING:
                alert.resolve()

    def clear_resolved(self) -> None:
        self._alerts = [a for a in self._alerts if a.status == AlertStatus.FIRING]


ALERT_MANAGER = AlertManager()
