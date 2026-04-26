"""Tests for src/deployment/config_drift_detector.py — ≥28 test cases."""

from __future__ import annotations

import dataclasses

import pytest

from src.deployment.config_drift_detector import (
    CONFIG_DRIFT_DETECTOR_REGISTRY,
    ConfigDriftDetector,
    ConfigField,
    DriftReport,
)

# ---------------------------------------------------------------------------
# ConfigField frozen dataclass
# ---------------------------------------------------------------------------


class TestConfigField:
    def test_fields_stored(self):
        cf = ConfigField("a.b", "expected", "actual", True)
        assert cf.path == "a.b"
        assert cf.expected == "expected"
        assert cf.actual == "actual"
        assert cf.drifted is True

    def test_not_drifted_when_equal(self):
        cf = ConfigField("x", 1, 1, False)
        assert cf.drifted is False

    def test_frozen_path(self):
        cf = ConfigField("p", 1, 2, True)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cf.path = "other"  # type: ignore[misc]

    def test_frozen_drifted(self):
        cf = ConfigField("p", 1, 1, False)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cf.drifted = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DriftReport frozen dataclass
# ---------------------------------------------------------------------------


class TestDriftReport:
    def test_fields_stored(self):
        report = DriftReport(total_fields=5, drifted_count=2, drifted_fields=[], drift_pct=40.0)
        assert report.total_fields == 5
        assert report.drifted_count == 2
        assert report.drift_pct == 40.0

    def test_frozen(self):
        report = DriftReport(3, 1, [], 33.33)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            report.drifted_count = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConfigDriftDetector.compare — identical configs
# ---------------------------------------------------------------------------


class TestCompareIdentical:
    def setup_method(self):
        self.detector = ConfigDriftDetector()

    def test_identical_flat_no_drift(self):
        cfg = {"a": 1, "b": "hello"}
        report = self.detector.compare(cfg, cfg)
        assert report.drifted_count == 0

    def test_identical_nested_no_drift(self):
        cfg = {"model": {"layers": 12, "heads": 8}}
        report = self.detector.compare(cfg, cfg)
        assert report.drifted_count == 0

    def test_empty_dicts_no_drift(self):
        report = self.detector.compare({}, {})
        assert report.total_fields == 0
        assert report.drifted_count == 0

    def test_is_clean_for_identical(self):
        cfg = {"x": 99}
        report = self.detector.compare(cfg, cfg)
        assert self.detector.is_clean(report) is True


# ---------------------------------------------------------------------------
# ConfigDriftDetector.compare — single field drifted
# ---------------------------------------------------------------------------


class TestCompareSingleDrift:
    def setup_method(self):
        self.detector = ConfigDriftDetector()

    def test_single_leaf_diff(self):
        report = self.detector.compare({"a": 1}, {"a": 2})
        assert report.drifted_count == 1

    def test_drifted_field_path(self):
        report = self.detector.compare({"key": "old"}, {"key": "new"})
        assert report.drifted_fields[0].path == "key"

    def test_drifted_field_expected_value(self):
        report = self.detector.compare({"k": 10}, {"k": 20})
        assert report.drifted_fields[0].expected == 10

    def test_drifted_field_actual_value(self):
        report = self.detector.compare({"k": 10}, {"k": 20})
        assert report.drifted_fields[0].actual == 20

    def test_total_fields_single(self):
        report = self.detector.compare({"a": 1}, {"a": 2})
        assert report.total_fields == 1


# ---------------------------------------------------------------------------
# ConfigDriftDetector.compare — nested dict drift
# ---------------------------------------------------------------------------


class TestCompareNested:
    def setup_method(self):
        self.detector = ConfigDriftDetector()

    def test_nested_path_dot_notation(self):
        exp = {"model": {"layers": 12}}
        act = {"model": {"layers": 6}}
        report = self.detector.compare(exp, act)
        assert report.drifted_fields[0].path == "model.layers"

    def test_nested_no_drift(self):
        cfg = {"model": {"layers": 12, "heads": 8}}
        report = self.detector.compare(cfg, dict(cfg))
        # Must recurse into nested dict — compare a copy with identical values
        exp = {"model": {"layers": 12, "heads": 8}}
        act = {"model": {"layers": 12, "heads": 8}}
        report = self.detector.compare(exp, act)
        assert report.drifted_count == 0

    def test_nested_partial_drift(self):
        exp = {"model": {"layers": 12, "heads": 8}}
        act = {"model": {"layers": 6, "heads": 8}}
        report = self.detector.compare(exp, act)
        assert report.drifted_count == 1
        assert report.total_fields == 2

    def test_deeply_nested_path(self):
        exp = {"a": {"b": {"c": 1}}}
        act = {"a": {"b": {"c": 2}}}
        report = self.detector.compare(exp, act)
        assert report.drifted_fields[0].path == "a.b.c"

    def test_prefix_prepended_to_paths(self):
        exp = {"layer": 1}
        act = {"layer": 2}
        report = self.detector.compare(exp, act, prefix="model")
        assert report.drifted_fields[0].path == "model.layer"

    def test_missing_key_in_actual_detected(self):
        exp = {"a": 1, "b": 2}
        act = {"a": 1}
        report = self.detector.compare(exp, act)
        # "b" is in expected but not actual → drifted
        drifted_paths = {f.path for f in report.drifted_fields}
        assert "b" in drifted_paths

    def test_extra_key_in_actual_detected(self):
        exp = {"a": 1}
        act = {"a": 1, "b": 2}
        report = self.detector.compare(exp, act)
        drifted_paths = {f.path for f in report.drifted_fields}
        assert "b" in drifted_paths


# ---------------------------------------------------------------------------
# Drift percentage calculation
# ---------------------------------------------------------------------------


class TestDriftPct:
    def setup_method(self):
        self.detector = ConfigDriftDetector()

    def test_drift_pct_zero_when_clean(self):
        cfg = {"a": 1, "b": 2}
        report = self.detector.compare(cfg, cfg)
        assert report.drift_pct == 0.0

    def test_drift_pct_100_when_all_drifted(self):
        exp = {"a": 1}
        act = {"a": 99}
        report = self.detector.compare(exp, act)
        assert report.drift_pct == 100.0

    def test_drift_pct_50_when_half_drifted(self):
        exp = {"a": 1, "b": 2}
        act = {"a": 99, "b": 2}
        report = self.detector.compare(exp, act)
        assert abs(report.drift_pct - 50.0) < 1e-9

    def test_drift_pct_zero_for_empty(self):
        report = self.detector.compare({}, {})
        assert report.drift_pct == 0.0


# ---------------------------------------------------------------------------
# is_clean
# ---------------------------------------------------------------------------


class TestIsClean:
    def setup_method(self):
        self.detector = ConfigDriftDetector()

    def test_is_clean_true(self):
        report = DriftReport(total_fields=3, drifted_count=0, drifted_fields=[], drift_pct=0.0)
        assert self.detector.is_clean(report) is True

    def test_is_clean_false(self):
        report = DriftReport(total_fields=3, drifted_count=1, drifted_fields=[], drift_pct=33.33)
        assert self.detector.is_clean(report) is False


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def setup_method(self):
        self.detector = ConfigDriftDetector()

    def test_summary_format_no_drift(self):
        report = DriftReport(total_fields=4, drifted_count=0, drifted_fields=[], drift_pct=0.0)
        s = self.detector.summary(report)
        assert s == "0/4 fields drifted (0%)"

    def test_summary_format_all_drifted(self):
        report = DriftReport(total_fields=2, drifted_count=2, drifted_fields=[], drift_pct=100.0)
        s = self.detector.summary(report)
        assert s == "2/2 fields drifted (100%)"

    def test_summary_format_partial(self):
        report = DriftReport(total_fields=2, drifted_count=1, drifted_fields=[], drift_pct=50.0)
        s = self.detector.summary(report)
        assert s == "1/2 fields drifted (50%)"

    def test_summary_is_string(self):
        report = DriftReport(total_fields=1, drifted_count=0, drifted_fields=[], drift_pct=0.0)
        assert isinstance(self.detector.summary(report), str)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in CONFIG_DRIFT_DETECTOR_REGISTRY

    def test_registry_default_is_class(self):
        assert CONFIG_DRIFT_DETECTOR_REGISTRY["default"] is ConfigDriftDetector

    def test_registry_default_is_instantiable(self):
        cls = CONFIG_DRIFT_DETECTOR_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, ConfigDriftDetector)
