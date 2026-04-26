"""Tests for AnomalyDetector."""

from __future__ import annotations

import pytest

from src.monitoring.anomaly_detector import (
    ANOMALY_DETECTOR_REGISTRY,
    Anomaly,
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyMethod,
)

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class TestAnomalyMethodEnum:
    def test_zscore(self):
        assert AnomalyMethod.ZSCORE == "zscore"

    def test_iqr(self):
        assert AnomalyMethod.IQR == "iqr"

    def test_rolling(self):
        assert AnomalyMethod.ROLLING_STDDEV == "rolling_stddev"

    def test_threshold(self):
        assert AnomalyMethod.THRESHOLD == "threshold"

    def test_four_members(self):
        assert len(AnomalyMethod) == 4


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


class TestAnomalyDataclass:
    def test_fields(self):
        a = Anomaly(
            metric_name="m",
            timestamp=1.0,
            value=2.0,
            method=AnomalyMethod.ZSCORE,
            score=0.5,
            is_anomaly=False,
        )
        assert a.metric_name == "m"
        assert a.value == 2.0
        assert a.is_anomaly is False

    def test_frozen(self):
        a = Anomaly(
            metric_name="m",
            timestamp=1.0,
            value=2.0,
            method=AnomalyMethod.ZSCORE,
            score=0.5,
            is_anomaly=False,
        )
        with pytest.raises(Exception):
            a.value = 3.0  # type: ignore


class TestConfig:
    def test_defaults(self):
        c = AnomalyDetectorConfig()
        assert c.zscore_threshold == 3.0
        assert c.iqr_multiplier == 1.5
        assert c.rolling_window == 20
        assert c.static_threshold == {}

    def test_override(self):
        c = AnomalyDetectorConfig(zscore_threshold=2.0)
        assert c.zscore_threshold == 2.0


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


class TestZscore:
    def setup_method(self):
        self.d = AnomalyDetector()

    def test_empty_returns_empty(self):
        assert self.d.detect_zscore("m", []) == []

    def test_returns_one_per_point(self):
        res = self.d.detect_zscore("m", [1.0, 2.0, 3.0])
        assert len(res) == 3

    def test_outlier_detected(self):
        values = [1.0] * 20 + [100.0]
        res = self.d.detect_zscore("m", values)
        assert res[-1].is_anomaly is True

    def test_uniform_no_anomaly(self):
        res = self.d.detect_zscore("m", [5.0] * 20)
        assert all(not r.is_anomaly for r in res)

    def test_normal_data_no_anomaly(self):
        values = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        res = self.d.detect_zscore("m", values)
        assert not any(r.is_anomaly for r in res)

    def test_method_tag(self):
        res = self.d.detect_zscore("m", [1.0, 2.0])
        assert all(r.method == AnomalyMethod.ZSCORE for r in res)

    def test_custom_threshold(self):
        d = AnomalyDetector(AnomalyDetectorConfig(zscore_threshold=1.0))
        values = [1.0] * 10 + [5.0]
        res = d.detect_zscore("m", values)
        assert res[-1].is_anomaly is True


# ---------------------------------------------------------------------------
# IQR
# ---------------------------------------------------------------------------


class TestIQR:
    def setup_method(self):
        self.d = AnomalyDetector()

    def test_empty(self):
        assert self.d.detect_iqr("m", []) == []

    def test_outlier_detected(self):
        values = list(range(1, 21)) + [1000]
        res = self.d.detect_iqr("m", values)
        assert res[-1].is_anomaly is True

    def test_no_outlier_uniform(self):
        res = self.d.detect_iqr("m", [5.0] * 10)
        assert not any(r.is_anomaly for r in res)

    def test_method_tag(self):
        res = self.d.detect_iqr("m", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert all(r.method == AnomalyMethod.IQR for r in res)

    def test_low_outlier_detected(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, -500.0]
        res = self.d.detect_iqr("m", values)
        assert any(r.is_anomaly and r.value == -500.0 for r in res)


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------


class TestRolling:
    def setup_method(self):
        self.d = AnomalyDetector(AnomalyDetectorConfig(rolling_window=5))

    def test_empty(self):
        assert self.d.detect_rolling("m", []) == []

    def test_outlier_detected(self):
        values = [1.0, 1.0, 1.01, 0.99, 1.0, 1.0, 1.0, 1.0, 100.0]
        res = self.d.detect_rolling("m", values)
        assert res[-1].is_anomaly is True

    def test_uniform_no_anomaly(self):
        res = self.d.detect_rolling("m", [3.0] * 10)
        assert not any(r.is_anomaly for r in res)

    def test_method_tag(self):
        res = self.d.detect_rolling("m", [1.0, 2.0, 3.0, 4.0])
        assert all(r.method == AnomalyMethod.ROLLING_STDDEV for r in res)

    def test_timestamps_applied(self):
        ts = [10.0, 20.0, 30.0]
        res = self.d.detect_rolling("m", [1.0, 2.0, 3.0], timestamps=ts)
        assert [r.timestamp for r in res] == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def setup_method(self):
        self.d = AnomalyDetector()

    def test_within_bounds(self):
        res = self.d.detect_threshold("m", [5.0, 6.0, 7.0], 0.0, 10.0)
        assert not any(r.is_anomaly for r in res)

    def test_above_high(self):
        res = self.d.detect_threshold("m", [5.0, 15.0], 0.0, 10.0)
        assert res[1].is_anomaly is True

    def test_below_low(self):
        res = self.d.detect_threshold("m", [-5.0, 5.0], 0.0, 10.0)
        assert res[0].is_anomaly is True

    def test_empty(self):
        assert self.d.detect_threshold("m", [], 0.0, 1.0) == []

    def test_method_tag(self):
        res = self.d.detect_threshold("m", [1.0], 0.0, 10.0)
        assert res[0].method == AnomalyMethod.THRESHOLD


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDetect:
    def setup_method(self):
        self.d = AnomalyDetector()

    def test_default_zscore(self):
        res = self.d.detect("m", [1.0, 2.0, 3.0])
        assert all(r.method == AnomalyMethod.ZSCORE for r in res)

    def test_iqr(self):
        res = self.d.detect("m", [1.0, 2.0, 3.0, 4.0], method=AnomalyMethod.IQR)
        assert all(r.method == AnomalyMethod.IQR for r in res)

    def test_rolling(self):
        res = self.d.detect("m", [1.0, 2.0], method=AnomalyMethod.ROLLING_STDDEV)
        assert all(r.method == AnomalyMethod.ROLLING_STDDEV for r in res)

    def test_threshold_kwargs(self):
        res = self.d.detect("m", [5.0, 20.0], method=AnomalyMethod.THRESHOLD, low=0.0, high=10.0)
        assert res[1].is_anomaly is True

    def test_threshold_from_config(self):
        d = AnomalyDetector(AnomalyDetectorConfig(static_threshold={"m": (0.0, 10.0)}))
        res = d.detect("m", [5.0, 20.0], method=AnomalyMethod.THRESHOLD)
        assert res[1].is_anomaly is True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def setup_method(self):
        self.d = AnomalyDetector()

    def test_empty(self):
        s = self.d.summary([])
        assert s == {"count": 0, "anomaly_rate": 0.0, "worst_score": 0.0}

    def test_counts_only_flagged(self):
        anomalies = [
            Anomaly("m", 0, 1, AnomalyMethod.ZSCORE, 0.5, False),
            Anomaly("m", 1, 2, AnomalyMethod.ZSCORE, 4.0, True),
            Anomaly("m", 2, 3, AnomalyMethod.ZSCORE, 0.2, False),
        ]
        s = self.d.summary(anomalies)
        assert s["count"] == 1

    def test_anomaly_rate(self):
        anomalies = [
            Anomaly("m", 0, 1, AnomalyMethod.ZSCORE, 4.0, True),
            Anomaly("m", 1, 2, AnomalyMethod.ZSCORE, 0.2, False),
        ]
        s = self.d.summary(anomalies)
        assert s["anomaly_rate"] == pytest.approx(50.0)

    def test_worst_score(self):
        anomalies = [
            Anomaly("m", 0, 1, AnomalyMethod.ZSCORE, 1.0, False),
            Anomaly("m", 1, 2, AnomalyMethod.ZSCORE, 4.5, True),
        ]
        s = self.d.summary(anomalies)
        assert s["worst_score"] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default(self):
        assert "default" in ANOMALY_DETECTOR_REGISTRY
        assert ANOMALY_DETECTOR_REGISTRY["default"] is AnomalyDetector
