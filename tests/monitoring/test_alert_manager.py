"""Tests for AlertManager (~50 tests)."""

from __future__ import annotations

from src.monitoring.alert_manager import (
    ALERT_MANAGER,
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
)

# ---------------------------------------------------------------------------
# AlertSeverity enum
# ---------------------------------------------------------------------------


class TestAlertSeverityEnum:
    def test_critical(self):
        assert AlertSeverity.CRITICAL == "critical"

    def test_high(self):
        assert AlertSeverity.HIGH == "high"

    def test_medium(self):
        assert AlertSeverity.MEDIUM == "medium"

    def test_low(self):
        assert AlertSeverity.LOW == "low"

    def test_info(self):
        assert AlertSeverity.INFO == "info"

    def test_five_members(self):
        assert len(AlertSeverity) == 5

    def test_is_str(self):
        assert isinstance(AlertSeverity.HIGH, str)


# ---------------------------------------------------------------------------
# AlertRule dataclass
# ---------------------------------------------------------------------------


class TestAlertRule:
    def test_basic_fields(self):
        rule = AlertRule(
            name="cpu_high",
            metric_name="cpu",
            threshold=90.0,
            severity=AlertSeverity.HIGH,
        )
        assert rule.name == "cpu_high"
        assert rule.metric_name == "cpu"
        assert rule.threshold == 90.0
        assert rule.severity == AlertSeverity.HIGH

    def test_comparison_default(self):
        rule = AlertRule(name="r", metric_name="m", threshold=1.0, severity=AlertSeverity.INFO)
        assert rule.comparison == ">"

    def test_custom_comparison(self):
        rule = AlertRule(
            name="r", metric_name="m", threshold=1.0, severity=AlertSeverity.INFO, comparison="<"
        )
        assert rule.comparison == "<"

    def test_message_template_default_empty(self):
        rule = AlertRule(name="r", metric_name="m", threshold=1.0, severity=AlertSeverity.INFO)
        assert rule.message_template == ""

    def test_message_template_set(self):
        rule = AlertRule(
            name="r",
            metric_name="m",
            threshold=1.0,
            severity=AlertSeverity.INFO,
            message_template="cpu too high",
        )
        assert rule.message_template == "cpu too high"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


class TestAlert:
    def test_auto_id_generated(self):
        a = Alert(
            rule_name="r",
            severity=AlertSeverity.HIGH,
            metric_value=1.0,
            message="msg",
            fired_at="2026-01-01T00:00:00+00:00",
        )
        assert isinstance(a.id, str)
        assert len(a.id) == 8

    def test_unique_ids(self):
        a1 = Alert(
            rule_name="r",
            severity=AlertSeverity.HIGH,
            metric_value=1.0,
            message="m",
            fired_at="2026-01-01T00:00:00+00:00",
        )
        a2 = Alert(
            rule_name="r",
            severity=AlertSeverity.HIGH,
            metric_value=1.0,
            message="m",
            fired_at="2026-01-01T00:00:00+00:00",
        )
        assert a1.id != a2.id

    def test_resolved_default_false(self):
        a = Alert(
            rule_name="r",
            severity=AlertSeverity.LOW,
            metric_value=0.0,
            message="m",
            fired_at="2026-01-01T00:00:00+00:00",
        )
        assert a.resolved is False

    def test_fields_stored(self):
        a = Alert(
            rule_name="my_rule",
            severity=AlertSeverity.CRITICAL,
            metric_value=99.5,
            message="boom",
            fired_at="ts",
        )
        assert a.rule_name == "my_rule"
        assert a.severity == AlertSeverity.CRITICAL
        assert a.metric_value == 99.5
        assert a.message == "boom"
        assert a.fired_at == "ts"


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    def setup_method(self):
        self.am = AlertManager()

    def _rule(
        self,
        name="cpu_high",
        metric="cpu",
        threshold=80.0,
        severity=AlertSeverity.HIGH,
        comparison=">",
    ):
        return AlertRule(
            name=name,
            metric_name=metric,
            threshold=threshold,
            severity=severity,
            comparison=comparison,
        )

    # add_rule
    def test_add_rule_stores(self):
        rule = self._rule()
        self.am.add_rule(rule)
        assert rule in self.am.rules()

    def test_add_multiple_rules(self):
        r1 = self._rule("r1", "cpu")
        r2 = self._rule("r2", "mem")
        self.am.add_rule(r1)
        self.am.add_rule(r2)
        assert len(self.am.rules()) == 2

    # evaluate — fires
    def test_evaluate_fires_alert_gt(self):
        self.am.add_rule(self._rule())
        alerts = self.am.evaluate("cpu", 95.0)
        assert len(alerts) == 1

    def test_evaluate_alert_has_correct_rule(self):
        self.am.add_rule(self._rule())
        alerts = self.am.evaluate("cpu", 95.0)
        assert alerts[0].rule_name == "cpu_high"

    def test_evaluate_alert_has_correct_severity(self):
        self.am.add_rule(self._rule())
        alerts = self.am.evaluate("cpu", 95.0)
        assert alerts[0].severity == AlertSeverity.HIGH

    def test_evaluate_alert_has_metric_value(self):
        self.am.add_rule(self._rule())
        alerts = self.am.evaluate("cpu", 95.0)
        assert alerts[0].metric_value == 95.0

    def test_evaluate_no_fire_when_below(self):
        self.am.add_rule(self._rule(threshold=80.0))
        alerts = self.am.evaluate("cpu", 50.0)
        assert alerts == []

    def test_evaluate_does_not_refire_active_rule(self):
        self.am.add_rule(self._rule())
        self.am.evaluate("cpu", 95.0)
        alerts2 = self.am.evaluate("cpu", 96.0)
        assert alerts2 == []

    def test_evaluate_fires_for_lt_comparison(self):
        rule = self._rule(name="mem_low", metric="mem", threshold=10.0, comparison="<")
        self.am.add_rule(rule)
        alerts = self.am.evaluate("mem", 5.0)
        assert len(alerts) == 1

    def test_evaluate_no_fire_for_lt_when_above(self):
        rule = self._rule(name="mem_low", metric="mem", threshold=10.0, comparison="<")
        self.am.add_rule(rule)
        alerts = self.am.evaluate("mem", 50.0)
        assert alerts == []

    def test_evaluate_fires_for_gte(self):
        rule = self._rule(comparison=">=", threshold=80.0)
        self.am.add_rule(rule)
        assert len(self.am.evaluate("cpu", 80.0)) == 1

    def test_evaluate_fires_for_lte(self):
        rule = AlertRule(
            name="r",
            metric_name="temp",
            threshold=0.0,
            severity=AlertSeverity.INFO,
            comparison="<=",
        )
        self.am.add_rule(rule)
        assert len(self.am.evaluate("temp", 0.0)) == 1

    def test_evaluate_fires_for_eq(self):
        rule = AlertRule(
            name="r", metric_name="q", threshold=42.0, severity=AlertSeverity.INFO, comparison="=="
        )
        self.am.add_rule(rule)
        assert len(self.am.evaluate("q", 42.0)) == 1

    def test_evaluate_only_matching_metric(self):
        self.am.add_rule(self._rule(metric="cpu"))
        alerts = self.am.evaluate("mem", 999.0)
        assert alerts == []

    def test_evaluate_fired_at_is_iso8601(self):
        self.am.add_rule(self._rule())
        alerts = self.am.evaluate("cpu", 99.0)
        fired_at = alerts[0].fired_at
        # Should contain 'T' separating date and time
        assert "T" in fired_at

    # resolve
    def test_resolve_returns_true(self):
        self.am.add_rule(self._rule())
        self.am.evaluate("cpu", 95.0)
        assert self.am.resolve("cpu_high") is True

    def test_resolve_marks_alert_resolved(self):
        self.am.add_rule(self._rule())
        self.am.evaluate("cpu", 95.0)
        self.am.resolve("cpu_high")
        assert self.am.active_alerts() == []

    def test_resolve_unknown_returns_false(self):
        assert self.am.resolve("no_such_rule") is False

    def test_resolve_allows_refire(self):
        self.am.add_rule(self._rule())
        self.am.evaluate("cpu", 95.0)
        self.am.resolve("cpu_high")
        alerts = self.am.evaluate("cpu", 97.0)
        assert len(alerts) == 1

    # active_alerts
    def test_active_alerts_empty_initially(self):
        assert self.am.active_alerts() == []

    def test_active_alerts_excludes_resolved(self):
        self.am.add_rule(self._rule())
        self.am.evaluate("cpu", 95.0)
        self.am.resolve("cpu_high")
        assert self.am.active_alerts() == []

    def test_active_alerts_includes_unresolved(self):
        self.am.add_rule(self._rule())
        self.am.evaluate("cpu", 95.0)
        assert len(self.am.active_alerts()) == 1

    # alert_history
    def test_alert_history_includes_all(self):
        self.am.add_rule(self._rule())
        self.am.evaluate("cpu", 95.0)
        self.am.resolve("cpu_high")
        self.am.evaluate("cpu", 97.0)
        assert len(self.am.alert_history()) == 2

    def test_alert_history_empty_initially(self):
        assert self.am.alert_history() == []

    # rules
    def test_rules_returns_list(self):
        assert isinstance(self.am.rules(), list)

    def test_rules_empty_initially(self):
        assert self.am.rules() == []

    # message_template
    def test_custom_message_used(self):
        rule = AlertRule(
            name="r",
            metric_name="cpu",
            threshold=80.0,
            severity=AlertSeverity.INFO,
            message_template="custom msg",
        )
        self.am.add_rule(rule)
        alerts = self.am.evaluate("cpu", 99.0)
        assert alerts[0].message == "custom msg"

    def test_default_message_generated(self):
        self.am.add_rule(self._rule())
        alerts = self.am.evaluate("cpu", 99.0)
        assert len(alerts[0].message) > 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_alert_manager_exists(self):
        assert ALERT_MANAGER is not None
        assert isinstance(ALERT_MANAGER, AlertManager)
