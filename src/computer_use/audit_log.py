"""Aurelius v2 CUA Audit Log — immutable audit trail of all CUA actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CUAEntry:
    """A single entry in the CUA audit log."""
    timestamp: str
    action_type: str
    action_details: dict[str, Any]
    before_screenshot: str = ""
    after_screenshot: str = ""
    verification_result: str = "unknown"
    blocked: bool = False
    block_reason: str = ""
    task_id: str = ""


class CUAuditLog:
    """Immutable audit log of CUA actions.

    Every CUA execution produces a timestamped entry with:
    - Before/after screenshots
    - Action details
    - Verification result
    - Any blocks and reasons
    """

    def __init__(self) -> None:
        self._entries: list[CUAEntry] = []

    def add_entry(self, entry: CUAEntry) -> None:
        self._entries.append(entry)

    def get_entries(self, task_id: str = "") -> list[CUAEntry]:
        if task_id:
            return [e for e in self._entries if e.task_id == task_id]
        return list(self._entries)

    def get_blocked_actions(self) -> list[CUAEntry]:
        return [e for e in self._entries if e.blocked]

    def to_csv(self) -> str:
        lines = ["timestamp,action_type,verification,blocked,block_reason,task_id"]
        for e in self._entries:
            lines.append(f"{e.timestamp},{e.action_type},{e.verification_result},{e.blocked},{e.block_reason},{e.task_id}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)
