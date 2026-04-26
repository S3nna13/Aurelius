from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MetricSnapshot = dict[str, float]
ConditionFn = Callable[[MetricSnapshot], tuple[bool, str, float | None]]


@dataclass
class Rule:
    name: str
    condition: ConditionFn
    severity: str = "info"
    cooldown_seconds: float = 300.0
    _last_fired: float = field(default=0.0, repr=False)

    def evaluate(self, snapshot: MetricSnapshot) -> Alert | None:
        now = time.time()
        if now - self._last_fired < self.cooldown_seconds:
            return None
        triggered, message, value = self.condition(snapshot)
        if triggered:
            self._last_fired = now
            return Alert(
                name=self.name,
                severity=self.severity,
                message=message,
                value=value,
            )
        return None


@dataclass
class Alert:
    name: str
    severity: str
    message: str
    timestamp: float = field(default_factory=time.time)
    value: float | None = None
    labels: dict[str, str] = field(default_factory=dict)


class RuleEngine:
    def __init__(self) -> None:
        self._rules: list[Rule] = []
        self._alerts: list[Alert] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: Rule) -> None:
        with self._lock:
            self._rules.append(rule)

    def evaluate_all(self, snapshot: MetricSnapshot) -> list[Alert]:
        new_alerts: list[Alert] = []
        with self._lock:
            for rule in self._rules:
                alert = rule.evaluate(snapshot)
                if alert is not None:
                    self._alerts.append(alert)
                    new_alerts.append(alert)
        return new_alerts

    def active_alerts(self) -> list[Alert]:
        with self._lock:
            return list(self._alerts)

    def clear_alerts(self) -> None:
        with self._lock:
            self._alerts.clear()

    @property
    def rules(self) -> list[Rule]:
        with self._lock:
            return list(self._rules)


ALERT_ENGINE = RuleEngine()
