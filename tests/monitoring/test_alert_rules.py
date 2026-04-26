from __future__ import annotations

import time

from src.monitoring.alert_rules import (
    ALERT_ENGINE,
    Alert,
    Rule,
    RuleEngine,
)


class TestRule:
    def test_basic_fields(self):
        rule = Rule(
            name="high_cpu",
            condition=lambda s: (True, "cpu > 90", 95.0),
            severity="critical",
            cooldown_seconds=60.0,
        )
        assert rule.name == "high_cpu"
        assert rule.severity == "critical"
        assert rule.cooldown_seconds == 60.0

    def test_default_severity(self):
        rule = Rule(name="r", condition=lambda s: (True, "", None))
        assert rule.severity == "info"

    def test_default_cooldown(self):
        rule = Rule(name="r", condition=lambda s: (True, "", None))
        assert rule.cooldown_seconds == 300.0

    def test_evaluate_triggers_alert(self):
        rule = Rule(
            name="cpu",
            condition=lambda s: (True, "high cpu", 99.0),
            cooldown_seconds=0,
        )
        alert = rule.evaluate({"cpu": 99.0})
        assert alert is not None
        assert alert.name == "cpu"
        assert alert.message == "high cpu"
        assert alert.value == 99.0

    def test_evaluate_not_triggered(self):
        rule = Rule(
            name="cpu",
            condition=lambda s: (False, "", None),
            cooldown_seconds=0,
        )
        assert rule.evaluate({"cpu": 50.0}) is None

    def test_cooldown_suppresses_duplicates(self):
        rule = Rule(
            name="rapid",
            condition=lambda s: (True, "boom", None),
            cooldown_seconds=999999,
        )
        assert rule.evaluate({}) is not None
        assert rule.evaluate({}) is None

    def test_cooldown_expires(self):
        rule = Rule(
            name="slow",
            condition=lambda s: (True, "ok", None),
            cooldown_seconds=0.01,
        )
        assert rule.evaluate({}) is not None
        time.sleep(0.02)
        assert rule.evaluate({}) is not None

    def test_condition_receives_snapshot(self):
        captured: dict | None = None

        def check(snapshot):
            nonlocal captured
            captured = snapshot
            return (True, "", None)

        rule = Rule(name="test", condition=check, cooldown_seconds=0)
        snapshot = {"cpu": 85.0, "mem": 60.0}
        rule.evaluate(snapshot)
        assert captured == snapshot

    def test_condition_can_use_snapshot_values(self):
        def check(snapshot):
            cpu = snapshot.get("cpu", 0)
            return (cpu > 80, f"cpu={cpu}", cpu)

        rule = Rule(name="cpu_high", condition=check, cooldown_seconds=0)
        alert = rule.evaluate({"cpu": 95.0})
        assert alert is not None
        assert alert.value == 95.0

    def test_condition_no_fire_below_threshold(self):
        def check(snapshot):
            return (snapshot.get("cpu", 0) > 80, "", snapshot.get("cpu"))

        rule = Rule(name="cpu_high", condition=check, cooldown_seconds=0)
        assert rule.evaluate({"cpu": 50.0}) is None


class TestAlert:
    def test_fields(self):
        alert = Alert(
            name="test",
            severity="warning",
            message="something happened",
            value=42.0,
            labels={"host": "localhost"},
        )
        assert alert.name == "test"
        assert alert.severity == "warning"
        assert alert.message == "something happened"
        assert alert.value == 42.0
        assert alert.labels == {"host": "localhost"}

    def test_default_timestamp(self):
        alert = Alert(name="t", severity="info", message="m")
        assert alert.timestamp > 0

    def test_value_none_by_default(self):
        alert = Alert(name="t", severity="info", message="m")
        assert alert.value is None

    def test_labels_empty_by_default(self):
        alert = Alert(name="t", severity="info", message="m")
        assert alert.labels == {}


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_add_rule(self):
        rule = Rule(name="r", condition=lambda s: (True, "", None), cooldown_seconds=0)
        self.engine.add_rule(rule)
        assert len(self.engine.rules) == 1

    def test_add_multiple(self):
        for i in range(3):
            self.engine.add_rule(Rule(name=f"r{i}", condition=lambda s: (True, "", None), cooldown_seconds=0))
        assert len(self.engine.rules) == 3

    def test_evaluate_all_returns_fired(self):
        self.engine.add_rule(Rule(name="a", condition=lambda s: (True, "a", 1.0), cooldown_seconds=0))
        self.engine.add_rule(Rule(name="b", condition=lambda s: (False, "", None), cooldown_seconds=0))
        alerts = self.engine.evaluate_all({})
        assert len(alerts) == 1
        assert alerts[0].name == "a"

    def test_evaluate_all_no_trigger(self):
        self.engine.add_rule(Rule(name="a", condition=lambda s: (False, "", None), cooldown_seconds=0))
        assert self.engine.evaluate_all({}) == []

    def test_active_alerts_after_evaluate(self):
        self.engine.add_rule(Rule(name="a", condition=lambda s: (True, "", None), cooldown_seconds=0))
        self.engine.evaluate_all({})
        assert len(self.engine.active_alerts()) == 1

    def test_active_alerts_empty_initially(self):
        assert self.engine.active_alerts() == []

    def test_clear_alerts(self):
        self.engine.add_rule(Rule(name="a", condition=lambda s: (True, "", None), cooldown_seconds=0))
        self.engine.evaluate_all({})
        self.engine.clear_alerts()
        assert self.engine.active_alerts() == []

    def test_cooldown_within_engine(self):
        rule = Rule(name="slow", condition=lambda s: (True, "ok", None), cooldown_seconds=999999)
        self.engine.add_rule(rule)
        assert len(self.engine.evaluate_all({})) == 1
        assert len(self.engine.evaluate_all({})) == 0

    def test_evaluate_all_passes_snapshot(self):
        captured: dict | None = None

        def check(snapshot):
            nonlocal captured
            captured = snapshot
            return (True, "", None)

        self.engine.add_rule(Rule(name="t", condition=check, cooldown_seconds=0))
        self.engine.evaluate_all({"cpu": 90.0})
        assert captured == {"cpu": 90.0}


class TestSingleton:
    def test_engine_exists(self):
        assert ALERT_ENGINE is not None
        assert isinstance(ALERT_ENGINE, RuleEngine)
