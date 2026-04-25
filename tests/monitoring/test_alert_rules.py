from __future__ import annotations

from src.monitoring.alert_rules import (
    ALERT_MANAGER,
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
)


class TestAlert:
    def test_default_status_firing(self):
        alert = Alert(name="test", severity=AlertSeverity.WARNING, message="test msg")
        assert alert.status == AlertStatus.FIRING

    def test_resolve_changes_status(self):
        alert = Alert(name="test", severity=AlertSeverity.CRITICAL, message="msg")
        alert.resolve()
        assert alert.status == AlertStatus.RESOLVED


class TestAlertRule:
    def test_evaluate_triggered(self):
        rule = AlertRule(
            name="high_cpu",
            severity=AlertSeverity.WARNING,
            condition=lambda: (True, "CPU > 90%", 95.0),
        )
        alert = rule.evaluate()
        assert alert is not None
        assert alert.name == "high_cpu"
        assert alert.value == 95.0

    def test_evaluate_not_triggered(self):
        rule = AlertRule(
            name="low_cpu",
            severity=AlertSeverity.INFO,
            condition=lambda: (False, "", None),
        )
        alert = rule.evaluate()
        assert alert is None

    def test_cooldown_suppresses_rapid_firing(self):
        rule = AlertRule(
            name="rapid",
            severity=AlertSeverity.WARNING,
            condition=lambda: (True, "boom", None),
            cooldown_seconds=999999,
        )
        alert1 = rule.evaluate()
        alert2 = rule.evaluate()
        assert alert1 is not None
        assert alert2 is None


class TestAlertManager:
    def test_add_and_evaluate_all(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="cpu",
            severity=AlertSeverity.CRITICAL,
            condition=lambda: (True, "high cpu", 99.0),
            cooldown_seconds=0,
        ))
        mgr.add_rule(AlertRule(
            name="mem",
            severity=AlertSeverity.WARNING,
            condition=lambda: (False, "", None),
        ))
        alerts = mgr.evaluate_all()
        assert len(alerts) == 1
        assert alerts[0].name == "cpu"

    def test_active_alerts(self):
        mgr = AlertManager()
        alert = Alert(name="test", severity=AlertSeverity.INFO, message="msg")
        mgr._alerts.append(alert)
        assert len(mgr.active_alerts()) == 1
        alert.resolve()
        assert len(mgr.active_alerts()) == 0

    def test_resolve_by_name(self):
        mgr = AlertManager()
        mgr._alerts.append(Alert(name="a1", severity=AlertSeverity.WARNING, message="m"))
        mgr.resolve_by_name("a1")
        assert mgr.active_alerts() == []

    def test_clear_resolved(self):
        mgr = AlertManager()
        a1 = Alert(name="a1", severity=AlertSeverity.INFO, message="m")
        a2 = Alert(name="a2", severity=AlertSeverity.INFO, message="m")
        mgr._alerts.append(a1)
        mgr._alerts.append(a2)
        a1.resolve()
        mgr.clear_resolved()
        assert len(mgr._alerts) == 1
        assert mgr._alerts[0].name == "a2"

    def test_module_instance(self):
        assert ALERT_MANAGER is not None
        assert hasattr(ALERT_MANAGER, "add_rule")
