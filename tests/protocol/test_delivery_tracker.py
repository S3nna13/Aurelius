"""Tests for delivery_tracker — message reliability state machine."""
from __future__ import annotations

import time

import pytest

from src.protocol.delivery_tracker import (
    DeliveryConfig,
    DeliveryRecord,
    DeliveryStatus,
    DeliveryTracker,
    DELIVERY_TRACKER_REGISTRY,
    DEFAULT_DELIVERY_TRACKER,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_creates_pending_record():
    dt = DeliveryTracker()
    record = dt.register("msg-1", "alice")
    assert record.message_id == "msg-1"
    assert record.recipient == "alice"
    assert record.status == DeliveryStatus.PENDING
    assert record.attempts == 0


def test_register_stores_record():
    dt = DeliveryTracker()
    dt.register("msg-1", "alice")
    assert dt.get_record("msg-1") is not None


# ---------------------------------------------------------------------------
# Attempt tracking
# ---------------------------------------------------------------------------


def test_attempt_increments_counter():
    dt = DeliveryTracker()
    dt.register("msg-1", "alice")
    dt.attempt("msg-1")
    assert dt.get_record("msg-1").attempts == 1


def test_attempt_returns_none_for_unknown():
    dt = DeliveryTracker()
    assert dt.attempt("unknown") is None


# ---------------------------------------------------------------------------
# Acknowledgment
# ---------------------------------------------------------------------------


def test_ack_transitions_to_acked():
    dt = DeliveryTracker()
    dt.register("msg-1", "alice")
    assert dt.ack("msg-1") is True
    assert dt.get_record("msg-1").status == DeliveryStatus.ACKED


def test_ack_returns_false_for_unknown():
    dt = DeliveryTracker()
    assert dt.ack("unknown") is False


# ---------------------------------------------------------------------------
# Expiration / retry logic
# ---------------------------------------------------------------------------


def test_is_expired_false_immediately():
    dt = DeliveryTracker(config=DeliveryConfig(timeout_seconds=1.0))
    dt.register("msg-1", "alice")
    dt.attempt("msg-1")
    assert dt.is_expired("msg-1") is False


def test_is_expired_true_after_timeout():
    dt = DeliveryTracker(config=DeliveryConfig(timeout_seconds=0.05))
    dt.register("msg-1", "alice")
    dt.attempt("msg-1")
    time.sleep(0.1)
    assert dt.is_expired("msg-1") is True


def test_should_retry_when_expired_and_under_budget():
    dt = DeliveryTracker(config=DeliveryConfig(timeout_seconds=0.05, max_retries=2))
    dt.register("msg-1", "alice")
    dt.attempt("msg-1")
    time.sleep(0.1)
    assert dt.should_retry("msg-1") is True


def test_should_retry_false_when_already_acked():
    dt = DeliveryTracker(config=DeliveryConfig(timeout_seconds=0.05))
    dt.register("msg-1", "alice")
    dt.ack("msg-1")
    time.sleep(0.1)
    assert dt.should_retry("msg-1") is False


def test_should_retry_false_when_exhausted():
    dt = DeliveryTracker(config=DeliveryConfig(timeout_seconds=0.05, max_retries=1))
    dt.register("msg-1", "alice")
    dt.attempt("msg-1")
    dt.attempt("msg-1")  # attempts == 2 > max_retries 1
    time.sleep(0.1)
    assert dt.should_retry("msg-1") is False


# ---------------------------------------------------------------------------
# Mark failed / expired
# ---------------------------------------------------------------------------


def test_mark_failed_sets_status():
    dt = DeliveryTracker()
    dt.register("msg-1", "alice")
    assert dt.mark_failed("msg-1") is True
    assert dt.get_record("msg-1").status == DeliveryStatus.FAILED


def test_mark_expired_sets_status():
    dt = DeliveryTracker()
    dt.register("msg-1", "alice")
    assert dt.mark_expired("msg-1") is True
    assert dt.get_record("msg-1").status == DeliveryStatus.EXPIRED


def test_mark_failed_returns_false_for_unknown():
    dt = DeliveryTracker()
    assert dt.mark_failed("unknown") is False


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_pending_ids_returns_only_pending():
    dt = DeliveryTracker()
    dt.register("a", "alice")
    dt.register("b", "bob")
    dt.ack("b")
    assert dt.pending_ids() == ["a"]


def test_failed_ids_includes_failed_and_expired():
    dt = DeliveryTracker()
    dt.register("a", "alice")
    dt.register("b", "bob")
    dt.mark_failed("a")
    dt.mark_expired("b")
    failed = dt.failed_ids()
    assert "a" in failed
    assert "b" in failed


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_aggregation():
    dt = DeliveryTracker()
    dt.register("a", "alice")
    dt.register("b", "bob")
    dt.ack("b")
    dt.mark_failed("c",)
    # c not registered, so only a and b count
    stats = dt.stats()
    assert stats["pending"] == 1
    assert stats["acked"] == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_all_records():
    dt = DeliveryTracker()
    dt.register("a", "alice")
    dt.reset()
    assert dt.get_record("a") is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in DELIVERY_TRACKER_REGISTRY
    assert isinstance(DELIVERY_TRACKER_REGISTRY["default"], DeliveryTracker)


def test_default_is_delivery_tracker():
    assert isinstance(DEFAULT_DELIVERY_TRACKER, DeliveryTracker)
