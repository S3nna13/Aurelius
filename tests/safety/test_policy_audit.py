"""Tests for policy_audit.py — policy audit log, decisions, stats, export."""

from __future__ import annotations

import json

from src.safety.policy_audit import (
    POLICY_AUDIT_LOG,
    AuditEntry,
    PolicyAuditLog,
    PolicyDecision,
)

# ---------------------------------------------------------------------------
# PolicyDecision enum values (4)
# ---------------------------------------------------------------------------


def test_decision_allow_value():
    assert PolicyDecision.ALLOW == "allow"


def test_decision_block_value():
    assert PolicyDecision.BLOCK == "block"


def test_decision_warn_value():
    assert PolicyDecision.WARN == "warn"


def test_decision_redact_value():
    assert PolicyDecision.REDACT == "redact"


def test_decision_has_four_members():
    assert len(PolicyDecision) == 4


# ---------------------------------------------------------------------------
# AuditEntry auto-generates id
# ---------------------------------------------------------------------------


def test_audit_entry_id_auto_generated():
    e = AuditEntry.create("policy_a", PolicyDecision.ALLOW)
    assert e.id is not None
    assert len(e.id) == 8


def test_audit_entry_ids_are_unique():
    e1 = AuditEntry.create("policy_a", PolicyDecision.ALLOW)
    e2 = AuditEntry.create("policy_a", PolicyDecision.ALLOW)
    assert e1.id != e2.id


def test_audit_entry_id_is_hex():
    e = AuditEntry.create("policy_a", PolicyDecision.ALLOW)
    int(e.id, 16)  # should not raise


# ---------------------------------------------------------------------------
# AuditEntry timestamp
# ---------------------------------------------------------------------------


def test_audit_entry_timestamp_is_string():
    e = AuditEntry.create("policy_a", PolicyDecision.ALLOW)
    assert isinstance(e.timestamp, str)


def test_audit_entry_timestamp_not_empty():
    e = AuditEntry.create("policy_a", PolicyDecision.ALLOW)
    assert len(e.timestamp) > 0


# ---------------------------------------------------------------------------
# AuditEntry input_hash
# ---------------------------------------------------------------------------


def test_audit_entry_input_hash_16_chars_when_provided():
    e = AuditEntry.create("policy_a", PolicyDecision.BLOCK, input_text="some input text")
    assert len(e.input_hash) == 16


def test_audit_entry_input_hash_is_hex():
    e = AuditEntry.create("policy_a", PolicyDecision.BLOCK, input_text="hello")
    int(e.input_hash, 16)  # should not raise


def test_audit_entry_input_hash_empty_when_no_input():
    e = AuditEntry.create("policy_a", PolicyDecision.ALLOW)
    assert e.input_hash == ""


def test_audit_entry_same_input_same_hash():
    e1 = AuditEntry.create("p", PolicyDecision.ALLOW, input_text="hello")
    e2 = AuditEntry.create("p", PolicyDecision.ALLOW, input_text="hello")
    assert e1.input_hash == e2.input_hash


def test_audit_entry_different_input_different_hash():
    e1 = AuditEntry.create("p", PolicyDecision.ALLOW, input_text="hello")
    e2 = AuditEntry.create("p", PolicyDecision.ALLOW, input_text="world")
    assert e1.input_hash != e2.input_hash


# ---------------------------------------------------------------------------
# PolicyAuditLog.record()
# ---------------------------------------------------------------------------


def test_record_returns_audit_entry():
    log = PolicyAuditLog()
    entry = log.record("test_policy", PolicyDecision.ALLOW)
    assert isinstance(entry, AuditEntry)


def test_record_increments_total():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.BLOCK)
    assert log.stats()["total"] == 2


def test_record_stores_policy_name():
    log = PolicyAuditLog()
    entry = log.record("my_policy", PolicyDecision.WARN)
    assert entry.policy_name == "my_policy"


def test_record_stores_decision():
    log = PolicyAuditLog()
    entry = log.record("my_policy", PolicyDecision.REDACT)
    assert entry.decision == PolicyDecision.REDACT


def test_record_stores_reason():
    log = PolicyAuditLog()
    entry = log.record("p", PolicyDecision.BLOCK, reason="too risky")
    assert entry.reason == "too risky"


def test_record_with_input_text_creates_hash():
    log = PolicyAuditLog()
    entry = log.record("p", PolicyDecision.ALLOW, input_text="some input")
    assert len(entry.input_hash) == 16


# ---------------------------------------------------------------------------
# query() — filtering
# ---------------------------------------------------------------------------


def test_query_no_args_returns_all():
    log = PolicyAuditLog()
    log.record("p1", PolicyDecision.ALLOW)
    log.record("p2", PolicyDecision.BLOCK)
    log.record("p3", PolicyDecision.WARN)
    results = log.query()
    assert len(results) == 3


def test_query_by_policy_name():
    log = PolicyAuditLog()
    log.record("policy_a", PolicyDecision.ALLOW)
    log.record("policy_b", PolicyDecision.ALLOW)
    log.record("policy_a", PolicyDecision.BLOCK)
    results = log.query(policy_name="policy_a")
    assert len(results) == 2
    assert all(e.policy_name == "policy_a" for e in results)


def test_query_by_decision():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.BLOCK)
    log.record("p", PolicyDecision.ALLOW)
    results = log.query(decision=PolicyDecision.ALLOW)
    assert len(results) == 2
    assert all(e.decision == PolicyDecision.ALLOW for e in results)


def test_query_by_both_filters():
    log = PolicyAuditLog()
    log.record("policy_a", PolicyDecision.ALLOW)
    log.record("policy_a", PolicyDecision.BLOCK)
    log.record("policy_b", PolicyDecision.ALLOW)
    results = log.query(policy_name="policy_a", decision=PolicyDecision.BLOCK)
    assert len(results) == 1


def test_query_no_match_returns_empty():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    results = log.query(policy_name="nonexistent")
    assert results == []


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


def test_stats_returns_dict_with_total():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    stats = log.stats()
    assert "total" in stats


def test_stats_returns_dict_with_by_decision():
    log = PolicyAuditLog()
    stats = log.stats()
    assert "by_decision" in stats


def test_stats_returns_dict_with_by_policy():
    log = PolicyAuditLog()
    stats = log.stats()
    assert "by_policy" in stats


def test_stats_total_correct():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.BLOCK)
    assert log.stats()["total"] == 2


def test_stats_by_decision_correct():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.BLOCK)
    stats = log.stats()
    assert stats["by_decision"]["allow"] == 2
    assert stats["by_decision"]["block"] == 1


def test_stats_by_policy_correct():
    log = PolicyAuditLog()
    log.record("policy_a", PolicyDecision.ALLOW)
    log.record("policy_b", PolicyDecision.ALLOW)
    log.record("policy_a", PolicyDecision.BLOCK)
    stats = log.stats()
    assert stats["by_policy"]["policy_a"] == 2
    assert stats["by_policy"]["policy_b"] == 1


# ---------------------------------------------------------------------------
# export_jsonl()
# ---------------------------------------------------------------------------


def test_export_jsonl_returns_string():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    assert isinstance(log.export_jsonl(), str)


def test_export_jsonl_valid_json_lines():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW, reason="ok")
    log.record("p", PolicyDecision.BLOCK, reason="bad")
    lines = log.export_jsonl().strip().split("\n")
    for line in lines:
        obj = json.loads(line)
        assert isinstance(obj, dict)


def test_export_jsonl_line_count():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.BLOCK)
    lines = log.export_jsonl().strip().split("\n")
    assert len(lines) == 2


def test_export_jsonl_contains_policy_name():
    log = PolicyAuditLog()
    log.record("my_policy", PolicyDecision.ALLOW)
    data = json.loads(log.export_jsonl().strip())
    assert data["policy_name"] == "my_policy"


def test_export_jsonl_empty_log():
    log = PolicyAuditLog()
    assert log.export_jsonl() == ""


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_removes_all():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.BLOCK)
    log.clear()
    assert log.stats()["total"] == 0


def test_clear_returns_count():
    log = PolicyAuditLog()
    log.record("p", PolicyDecision.ALLOW)
    log.record("p", PolicyDecision.BLOCK)
    count = log.clear()
    assert count == 2


def test_clear_empty_log_returns_zero():
    log = PolicyAuditLog()
    assert log.clear() == 0


# ---------------------------------------------------------------------------
# POLICY_AUDIT_LOG global instance
# ---------------------------------------------------------------------------


def test_policy_audit_log_exists():
    assert POLICY_AUDIT_LOG is not None


def test_policy_audit_log_is_instance():
    assert isinstance(POLICY_AUDIT_LOG, PolicyAuditLog)


def test_policy_audit_log_can_record():
    # use a fresh log to avoid cross-test state
    POLICY_AUDIT_LOG.clear()
    entry = POLICY_AUDIT_LOG.record("global_policy", PolicyDecision.ALLOW)
    assert entry is not None
    POLICY_AUDIT_LOG.clear()
