"""Tests for src/security/rate_abuse_detector.py — ≥28 tests."""

from __future__ import annotations

import pytest

from src.security.rate_abuse_detector import (
    RATE_ABUSE_DETECTOR_REGISTRY,
    AbuseAlert,
    AbusePattern,
    RateAbuseDetector,
    RequestRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_records(client_id: str, n: int, start: float = 0.0, interval: float = 0.05) -> list:
    return [
        RequestRecord(client_id=client_id, timestamp=start + i * interval, endpoint="/api")
        for i in range(n)
    ]


def make_detector(**kwargs) -> RateAbuseDetector:
    defaults = dict(
        burst_threshold=10,
        burst_window_s=5.0,
        sustained_threshold=50,
        sustained_window_s=60.0,
    )
    defaults.update(kwargs)
    return RateAbuseDetector(**defaults)


# ---------------------------------------------------------------------------
# RequestRecord
# ---------------------------------------------------------------------------


class TestRequestRecord:
    def test_fields_stored(self):
        r = RequestRecord("alice", 1.0, "/api/v1", bytes_sent=512)
        assert r.client_id == "alice"
        assert r.timestamp == 1.0
        assert r.endpoint == "/api/v1"
        assert r.bytes_sent == 512

    def test_default_bytes_sent(self):
        r = RequestRecord("bob", 2.0, "/health")
        assert r.bytes_sent == 0

    def test_frozen(self):
        r = RequestRecord("carol", 3.0, "/login")
        with pytest.raises((AttributeError, TypeError)):
            r.client_id = "dave"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AbusePattern enum
# ---------------------------------------------------------------------------


class TestAbusePattern:
    def test_members_exist(self):
        assert AbusePattern.BURST
        assert AbusePattern.SUSTAINED
        assert AbusePattern.DISTRIBUTED
        assert AbusePattern.CREDENTIAL_STUFFING

    def test_values(self):
        assert AbusePattern.BURST.value == "BURST"
        assert AbusePattern.DISTRIBUTED.value == "DISTRIBUTED"


# ---------------------------------------------------------------------------
# AbuseAlert
# ---------------------------------------------------------------------------


class TestAbuseAlert:
    def test_fields(self):
        alert = AbuseAlert(
            client_id="x",
            pattern=AbusePattern.BURST,
            severity="high",
            request_count=20,
            window_s=5.0,
            detail="too fast",
        )
        assert alert.severity == "high"
        assert alert.pattern == AbusePattern.BURST

    def test_frozen(self):
        alert = AbuseAlert("x", AbusePattern.BURST, "high", 20, 5.0, "detail")
        with pytest.raises((AttributeError, TypeError)):
            alert.severity = "low"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RateAbuseDetector — no records
# ---------------------------------------------------------------------------


class TestNoRecords:
    def test_empty_list_no_alerts(self):
        det = make_detector()
        assert det.analyze([]) == []

    def test_single_record_no_alerts(self):
        det = make_detector()
        records = [RequestRecord("alice", 0.0, "/api")]
        assert det.analyze(records) == []


# ---------------------------------------------------------------------------
# BURST detection
# ---------------------------------------------------------------------------


class TestBurstDetection:
    def test_burst_triggers_alert(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        # 11 records within 1 second → burst
        records = make_records("attacker", n=11, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        burst = [a for a in alerts if a.pattern == AbusePattern.BURST]
        assert len(burst) == 1

    def test_burst_alert_severity_is_high(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        records = make_records("bad", n=15, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        burst = [a for a in alerts if a.pattern == AbusePattern.BURST]
        assert burst[0].severity == "high"

    def test_burst_alert_client_id(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        records = make_records("client_x", n=11, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        burst = [a for a in alerts if a.pattern == AbusePattern.BURST]
        assert burst[0].client_id == "client_x"

    def test_burst_no_trigger_below_threshold(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        # Only 10 records — threshold is 10, need *more than* 10
        records = make_records("normal", n=10, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        burst = [a for a in alerts if a.pattern == AbusePattern.BURST]
        assert len(burst) == 0

    def test_burst_spread_out_no_trigger(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        # 11 records but spread over 100 seconds
        records = make_records("slow", n=11, start=0.0, interval=10.0)
        alerts = det.analyze(records)
        burst = [a for a in alerts if a.pattern == AbusePattern.BURST]
        assert len(burst) == 0


# ---------------------------------------------------------------------------
# SUSTAINED detection
# ---------------------------------------------------------------------------


class TestSustainedDetection:
    def test_sustained_triggers_alert(self):
        det = make_detector(sustained_threshold=50, sustained_window_s=60.0)
        # 51 requests spread over time
        records = make_records("slow_attacker", n=51, start=0.0, interval=1.0)
        alerts = det.analyze(records)
        sustained = [a for a in alerts if a.pattern == AbusePattern.SUSTAINED]
        assert len(sustained) == 1

    def test_sustained_severity_is_medium(self):
        det = make_detector(sustained_threshold=50, sustained_window_s=60.0)
        records = make_records("m_client", n=51, start=0.0, interval=1.0)
        alerts = det.analyze(records)
        sustained = [a for a in alerts if a.pattern == AbusePattern.SUSTAINED]
        assert sustained[0].severity == "medium"

    def test_sustained_no_trigger_at_threshold(self):
        det = make_detector(sustained_threshold=50, sustained_window_s=60.0)
        records = make_records("ok", n=50, start=0.0, interval=1.0)
        alerts = det.analyze(records)
        sustained = [a for a in alerts if a.pattern == AbusePattern.SUSTAINED]
        assert len(sustained) == 0


# ---------------------------------------------------------------------------
# DISTRIBUTED detection
# ---------------------------------------------------------------------------


class TestDistributedDetection:
    def test_distributed_11_clients(self):
        det = make_detector()
        records = []
        for i in range(11):
            records.extend(make_records(f"client_{i}", n=51, start=0.0, interval=0.01))
        alerts = det.analyze(records)
        distributed = [a for a in alerts if a.pattern == AbusePattern.DISTRIBUTED]
        assert len(distributed) == 1

    def test_distributed_client_id_is_star(self):
        det = make_detector()
        records = []
        for i in range(11):
            records.extend(make_records(f"c{i}", n=51, start=0.0, interval=0.01))
        alerts = det.analyze(records)
        distributed = [a for a in alerts if a.pattern == AbusePattern.DISTRIBUTED]
        assert distributed[0].client_id == "*distributed*"

    def test_distributed_10_clients_no_trigger(self):
        det = make_detector()
        records = []
        for i in range(10):
            records.extend(make_records(f"c{i}", n=51, start=0.0, interval=0.01))
        alerts = det.analyze(records)
        distributed = [a for a in alerts if a.pattern == AbusePattern.DISTRIBUTED]
        assert len(distributed) == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_one_burst_alert_per_client(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        # 20 records — there should still be only ONE BURST alert for this client
        records = make_records("dup_client", n=20, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        burst = [
            a for a in alerts if a.pattern == AbusePattern.BURST and a.client_id == "dup_client"
        ]
        assert len(burst) == 1

    def test_one_sustained_alert_per_client(self):
        det = make_detector(sustained_threshold=50, sustained_window_s=60.0)
        records = make_records("heavy", n=100, start=0.0, interval=0.5)
        alerts = det.analyze(records)
        sustained = [
            a for a in alerts if a.pattern == AbusePattern.SUSTAINED and a.client_id == "heavy"
        ]
        assert len(sustained) == 1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_empty(self):
        det = make_detector()
        s = det.summary([])
        assert s["total_alerts"] == 0
        assert s["by_pattern"] == {}
        assert s["high_severity"] == 0

    def test_summary_total_alerts(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        records = make_records("a", n=15, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        s = det.summary(alerts)
        assert s["total_alerts"] == len(alerts)

    def test_summary_by_pattern(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        records = make_records("b", n=15, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        s = det.summary(alerts)
        assert "BURST" in s["by_pattern"]

    def test_summary_high_severity(self):
        det = make_detector(burst_threshold=10, burst_window_s=5.0)
        records = make_records("c", n=15, start=0.0, interval=0.1)
        alerts = det.analyze(records)
        s = det.summary(alerts)
        assert s["high_severity"] >= 1


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in RATE_ABUSE_DETECTOR_REGISTRY

    def test_registry_default_is_class(self):
        cls = RATE_ABUSE_DETECTOR_REGISTRY["default"]
        det = cls()
        assert isinstance(det, RateAbuseDetector)
