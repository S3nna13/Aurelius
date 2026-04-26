"""Policy audit log: record policy decisions, aggregate stats, export."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"


@dataclass
class AuditEntry:
    id: str
    timestamp: str
    policy_name: str
    decision: PolicyDecision
    reason: str = ""
    input_hash: str = ""

    @classmethod
    def create(
        cls,
        policy_name: str,
        decision: PolicyDecision,
        reason: str = "",
        input_text: str = "",
    ) -> AuditEntry:
        entry_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now(UTC).isoformat()
        input_hash = ""
        if input_text:
            input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
        return cls(
            id=entry_id,
            timestamp=timestamp,
            policy_name=policy_name,
            decision=decision,
            reason=reason,
            input_hash=input_hash,
        )


class PolicyAuditLog:
    def __init__(self, max_entries: int = 10000) -> None:
        self._max_entries = max_entries
        self._entries: list[AuditEntry] = []

    def record(
        self,
        policy_name: str,
        decision: PolicyDecision,
        reason: str = "",
        input_text: str = "",
    ) -> AuditEntry:
        entry = AuditEntry.create(
            policy_name=policy_name,
            decision=decision,
            reason=reason,
            input_text=input_text,
        )
        if len(self._entries) >= self._max_entries:
            self._entries.pop(0)
        self._entries.append(entry)
        return entry

    def query(
        self,
        policy_name: str | None = None,
        decision: PolicyDecision | None = None,
    ) -> list[AuditEntry]:
        results = self._entries
        if policy_name is not None:
            results = [e for e in results if e.policy_name == policy_name]
        if decision is not None:
            results = [e for e in results if e.decision == decision]
        return list(results)

    def stats(self) -> dict:
        by_decision: dict[str, int] = {}
        by_policy: dict[str, int] = {}
        for entry in self._entries:
            by_decision[entry.decision.value] = by_decision.get(entry.decision.value, 0) + 1
            by_policy[entry.policy_name] = by_policy.get(entry.policy_name, 0) + 1
        return {
            "total": len(self._entries),
            "by_decision": by_decision,
            "by_policy": by_policy,
        }

    def export_jsonl(self) -> str:
        lines = [json.dumps(dataclasses.asdict(e)) for e in self._entries]
        return "\n".join(lines)

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count


POLICY_AUDIT_LOG = PolicyAuditLog()
