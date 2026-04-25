"""Tests for src.security.audit_logger."""

from __future__ import annotations

import json
import logging

import pytest

from src.security.audit_logger import (
    AUDIT_LOGGER_NAME,
    AuditCategory,
    AuditEvent,
    AuditLevel,
    AuditLogger,
)


def _make_logger() -> AuditLogger:
    return AuditLogger()


def test_log_creates_event():
    logger = _make_logger()
    event = AuditEvent.create(
        category=AuditCategory.AUTH,
        level=AuditLevel.INFO,
        principal="alice",
        action="login",
        target="/session",
        outcome="allowed",
        detail="interactive",
    )
    logger.log(event)
    recent = logger.recent_events(10)
    assert len(recent) == 1
    assert recent[0].action == "login"
    assert recent[0].category is AuditCategory.AUTH
    assert recent[0].level is AuditLevel.INFO
    assert recent[0].event_id  # uuid populated


def test_recent_events_capped_at_limit():
    logger = AuditLogger(buffer_size=10_000)
    for i in range(10_001):
        logger.log(
            AuditEvent.create(
                category=AuditCategory.TOOL_CALL,
                level=AuditLevel.INFO,
                principal=f"user{i}",
                action="call",
                target="tool",
                outcome="allowed",
            )
        )
    # Request more than cap -- should be clamped to buffer size.
    assert len(logger.recent_events(20_000)) <= 10_000
    assert len(logger.recent_events(10_000)) == 10_000


def test_redact_bearer_token():
    logger = _make_logger()
    redacted = logger._redact("Authorization: Bearer abc123secrettoken")
    assert "abc123secrettoken" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_api_key():
    logger = _make_logger()
    text = "api_key=supersecret123 and API-KEY:other456 and sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    redacted = logger._redact(text)
    assert "supersecret123" not in redacted
    assert "other456" not in redacted
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_does_not_raise_on_empty():
    logger = _make_logger()
    assert logger._redact("") == ""
    # Also verify None-ish inputs do not propagate.
    assert logger._redact(None) == ""  # type: ignore[arg-type]
    # Non-string inputs must also be handled.
    assert isinstance(logger._redact(1234), str)  # type: ignore[arg-type]


def test_json_output_structure(caplog: pytest.LogCaptureFixture):
    logger = _make_logger()
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        event = AuditEvent.create(
            category=AuditCategory.POLICY,
            level=AuditLevel.WARNING,
            principal="svc-account",
            action="write",
            target="/etc/policy",
            outcome="denied",
            detail="Bearer abc123secret",
        )
        logger.log(event)

    assert caplog.records, "expected at least one log record"
    record = caplog.records[-1]
    payload = json.loads(record.getMessage())
    for key in (
        "event_id",
        "category",
        "level",
        "principal",
        "action",
        "target",
        "outcome",
        "detail",
        "timestamp",
    ):
        assert key in payload
    assert payload["category"] == "POLICY"
    assert payload["level"] == "WARNING"
    assert payload["outcome"] == "denied"
    assert "abc123secret" not in payload["detail"]
    assert "[REDACTED]" in payload["detail"]
