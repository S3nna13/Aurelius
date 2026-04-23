"""Append-only session journal with branch and compaction semantics.

The journal is local-first and JSON-safe. It is intentionally small but
captures the main OpenClaw/IronClaw style primitives Aurelius already
supports elsewhere: persistent event history, named branches, and explicit
compaction summaries that preserve lineage instead of silently dropping data.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from src.longcontext.context_compaction import ContextCompactor, Turn
from src.model.interface_framework import InterfaceFrameworkError

__all__ = [
    "SessionJournalEntry",
    "SessionJournalBranch",
    "SessionJournalCompaction",
    "SessionJournal",
]


_VALID_SEVERITIES = frozenset({"info", "notice", "warning", "error", "critical"})
_VALID_COMPACTION_POLICIES = frozenset(
    {"oldest_first", "middle_biased", "tool_output_aggregated"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
    return value


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _dedupe_strings(values: tuple[str, ...] | list[str] | tuple[Any, ...] | list[Any]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise InterfaceFrameworkError("expected a sequence of strings, got bare str")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise InterfaceFrameworkError("sequence entries must be non-empty strings")
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _entry_text(entry: "SessionJournalEntry") -> str:
    payload = _json_safe(entry.payload)
    metadata = _json_safe(entry.metadata)
    parts = [
        f"{entry.kind}: {entry.summary}",
    ]
    if entry.thread_id is not None:
        parts.append(f"thread={entry.thread_id}")
    if entry.workstream_id is not None:
        parts.append(f"workstream={entry.workstream_id}")
    if entry.parent_entry_id is not None:
        parts.append(f"parent={entry.parent_entry_id}")
    if entry.tags:
        parts.append(f"tags={', '.join(entry.tags)}")
    if payload:
        parts.append(f"payload={json.dumps(payload, sort_keys=True)}")
    if metadata:
        parts.append(f"metadata={json.dumps(metadata, sort_keys=True)}")
    return " | ".join(parts)


def _turn_for_entry(entry: "SessionJournalEntry") -> Turn:
    kind = "tool_result" if entry.kind in {"tool_result", "tool_call"} else "message"
    return Turn(role=entry.kind, content=_entry_text(entry), kind=kind)


def _summarize_turns(turns: list[Turn]) -> str:
    if not turns:
        return "No journal entries."
    lines = []
    for turn in turns:
        lines.append(turn.content)
    return "\n".join(lines)


def _count_tokens(text: str) -> int:
    if not isinstance(text, str):
        raise InterfaceFrameworkError("tokenized text must be a string")
    return len(text.split())


@dataclass(frozen=True)
class SessionJournalEntry:
    """One append-only journal entry."""

    entry_id: str
    session_id: str
    kind: str
    summary: str
    created_at: str
    branch_id: str = "main"
    thread_id: str | None = None
    workstream_id: str | None = None
    parent_entry_id: str | None = None
    severity: str = "info"
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("entry_id", "session_id", "kind", "summary", "created_at", "branch_id"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.severity not in _VALID_SEVERITIES:
            raise InterfaceFrameworkError(
                f"severity must be one of {sorted(_VALID_SEVERITIES)}, got {self.severity!r}"
            )
        if self.thread_id is not None and not isinstance(self.thread_id, str):
            raise InterfaceFrameworkError("thread_id must be str or None")
        if self.workstream_id is not None and not isinstance(self.workstream_id, str):
            raise InterfaceFrameworkError("workstream_id must be str or None")
        if self.parent_entry_id is not None and not isinstance(self.parent_entry_id, str):
            raise InterfaceFrameworkError("parent_entry_id must be str or None")
        if not isinstance(self.payload, dict):
            raise InterfaceFrameworkError("payload must be a dict")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")
        if not isinstance(self.tags, tuple):
            raise InterfaceFrameworkError("tags must be a tuple")
        if not all(isinstance(item, str) and item for item in self.tags):
            raise InterfaceFrameworkError("tags entries must be non-empty strings")
        try:
            json.dumps(self.payload, sort_keys=True)
            json.dumps(self.metadata, sort_keys=True)
        except TypeError as exc:  # pragma: no cover - defensive
            raise InterfaceFrameworkError("payload and metadata must be JSON serializable") from exc

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionJournalEntry":
        return cls(
            entry_id=payload["entry_id"],
            session_id=payload["session_id"],
            kind=payload["kind"],
            summary=payload["summary"],
            created_at=payload["created_at"],
            branch_id=payload.get("branch_id", "main"),
            thread_id=payload.get("thread_id"),
            workstream_id=payload.get("workstream_id"),
            parent_entry_id=payload.get("parent_entry_id"),
            severity=payload.get("severity", "info"),
            payload=dict(payload.get("payload", {})),
            metadata=dict(payload.get("metadata", {})),
            tags=tuple(payload.get("tags", ())),
        )


@dataclass(frozen=True)
class SessionJournalBranch:
    """A named branch of journal history."""

    branch_id: str
    session_id: str
    name: str
    base_entry_id: str | None
    head_entry_id: str | None
    entry_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("branch_id", "session_id", "name", "created_at", "updated_at"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.base_entry_id is not None and not isinstance(self.base_entry_id, str):
            raise InterfaceFrameworkError("base_entry_id must be str or None")
        if self.head_entry_id is not None and not isinstance(self.head_entry_id, str):
            raise InterfaceFrameworkError("head_entry_id must be str or None")
        if not isinstance(self.entry_ids, tuple):
            raise InterfaceFrameworkError("entry_ids must be a tuple")
        if not all(isinstance(item, str) and item for item in self.entry_ids):
            raise InterfaceFrameworkError("entry_ids entries must be non-empty strings")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")
        if self.head_entry_id is not None and self.head_entry_id not in self.entry_ids:
            raise InterfaceFrameworkError("head_entry_id must reference an entry in entry_ids")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionJournalBranch":
        return cls(
            branch_id=payload["branch_id"],
            session_id=payload["session_id"],
            name=payload["name"],
            base_entry_id=payload.get("base_entry_id"),
            head_entry_id=payload.get("head_entry_id"),
            entry_ids=tuple(payload.get("entry_ids", ())),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SessionJournalCompaction:
    """Record of a compaction pass and its summary entry."""

    compaction_id: str
    session_id: str
    branch_id: str
    created_at: str
    policy: str
    keep_last_n: int
    dropped_entry_ids: tuple[str, ...]
    retained_entry_ids: tuple[str, ...]
    summary_entry_id: str | None = None
    summary_text: str = ""
    facts: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("compaction_id", "session_id", "branch_id", "created_at", "policy"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.policy not in _VALID_COMPACTION_POLICIES:
            raise InterfaceFrameworkError(
                f"policy must be one of {sorted(_VALID_COMPACTION_POLICIES)}, got {self.policy!r}"
            )
        if not isinstance(self.keep_last_n, int) or self.keep_last_n < 0:
            raise InterfaceFrameworkError("keep_last_n must be a non-negative int")
        for tuple_name in ("dropped_entry_ids", "retained_entry_ids", "facts"):
            value = getattr(self, tuple_name)
            if not isinstance(value, tuple):
                raise InterfaceFrameworkError(f"{tuple_name} must be a tuple")
            if tuple_name == "facts":
                if not all(isinstance(item, str) and item for item in value):
                    raise InterfaceFrameworkError("facts entries must be non-empty strings")
                continue
            if not all(isinstance(item, str) and item for item in value):
                raise InterfaceFrameworkError(f"{tuple_name} entries must be non-empty strings")
        if self.summary_entry_id is not None and not isinstance(self.summary_entry_id, str):
            raise InterfaceFrameworkError("summary_entry_id must be str or None")
        if not isinstance(self.summary_text, str):
            raise InterfaceFrameworkError("summary_text must be a string")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionJournalCompaction":
        return cls(
            compaction_id=payload["compaction_id"],
            session_id=payload["session_id"],
            branch_id=payload["branch_id"],
            created_at=payload["created_at"],
            policy=payload["policy"],
            keep_last_n=payload["keep_last_n"],
            dropped_entry_ids=tuple(payload.get("dropped_entry_ids", ())),
            retained_entry_ids=tuple(payload.get("retained_entry_ids", ())),
            summary_entry_id=payload.get("summary_entry_id"),
            summary_text=payload.get("summary_text", ""),
            facts=tuple(payload.get("facts", ())),
            metadata=dict(payload.get("metadata", {})),
        )


def _default_main_branch(session_id: str, created_at: str) -> SessionJournalBranch:
    return SessionJournalBranch(
        branch_id="main",
        session_id=session_id,
        name="main",
        base_entry_id=None,
        head_entry_id=None,
        entry_ids=tuple(),
        created_at=created_at,
        updated_at=created_at,
        metadata={},
    )


@dataclass
class SessionJournal:
    """Persistent journal with named branches and compaction summaries."""

    journal_id: str
    session_id: str
    created_at: str
    updated_at: str
    entries: tuple[SessionJournalEntry, ...] = field(default_factory=tuple)
    branches: dict[str, SessionJournalBranch] = field(default_factory=dict)
    compactions: dict[str, SessionJournalCompaction] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("journal_id", "session_id", "created_at", "updated_at"):
            _require_non_empty(getattr(self, field_name), field_name)
        if not isinstance(self.entries, tuple):
            raise InterfaceFrameworkError("entries must be a tuple")
        if not all(isinstance(item, SessionJournalEntry) for item in self.entries):
            raise InterfaceFrameworkError("entries entries must be SessionJournalEntry instances")
        if not isinstance(self.branches, dict):
            raise InterfaceFrameworkError("branches must be a dict")
        if "main" not in self.branches:
            raise InterfaceFrameworkError("branches must include a 'main' branch")
        if not all(isinstance(item, SessionJournalBranch) for item in self.branches.values()):
            raise InterfaceFrameworkError("branches values must be SessionJournalBranch instances")
        if not isinstance(self.compactions, dict):
            raise InterfaceFrameworkError("compactions must be a dict")
        if not all(
            isinstance(item, SessionJournalCompaction) for item in self.compactions.values()
        ):
            raise InterfaceFrameworkError(
                "compactions values must be SessionJournalCompaction instances"
            )
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")
        entry_ids = {entry.entry_id for entry in self.entries}
        for branch in self.branches.values():
            for entry_id in branch.entry_ids:
                if entry_id not in entry_ids:
                    raise InterfaceFrameworkError(
                        f"branch {branch.branch_id!r} references unknown entry {entry_id!r}"
                    )
            if branch.base_entry_id is not None and branch.base_entry_id not in entry_ids:
                raise InterfaceFrameworkError(
                    f"branch {branch.branch_id!r} references unknown base entry {branch.base_entry_id!r}"
                )
            if branch.head_entry_id is not None and branch.head_entry_id not in entry_ids:
                raise InterfaceFrameworkError(
                    f"branch {branch.branch_id!r} references unknown head entry {branch.head_entry_id!r}"
                )
        for compaction in self.compactions.values():
            if compaction.summary_entry_id is not None and compaction.summary_entry_id not in entry_ids:
                raise InterfaceFrameworkError(
                    f"compaction {compaction.compaction_id!r} references unknown summary entry"
                )
            for entry_id in compaction.dropped_entry_ids + compaction.retained_entry_ids:
                if entry_id not in entry_ids:
                    raise InterfaceFrameworkError(
                        f"compaction {compaction.compaction_id!r} references unknown entry {entry_id!r}"
                    )

    @classmethod
    def create(
        cls,
        session_id: str,
        *,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        journal_id: str | None = None,
    ) -> "SessionJournal":
        now = created_at or _utc_now()
        return cls(
            journal_id=journal_id or f"journal-{uuid.uuid4()}",
            session_id=session_id,
            created_at=now,
            updated_at=now,
            entries=tuple(),
            branches={"main": _default_main_branch(session_id, now)},
            compactions={},
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionJournal":
        branches = {
            branch_id: SessionJournalBranch.from_dict(branch_payload)
            for branch_id, branch_payload in dict(payload.get("branches", {})).items()
        }
        entries = tuple(
            SessionJournalEntry.from_dict(entry_payload)
            for entry_payload in payload.get("entries", ())
        )
        compactions = {
            compaction_id: SessionJournalCompaction.from_dict(compaction_payload)
            for compaction_id, compaction_payload in dict(payload.get("compactions", {})).items()
        }
        return cls(
            journal_id=payload["journal_id"],
            session_id=payload["session_id"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            entries=entries,
            branches=branches,
            compactions=compactions,
            metadata=dict(payload.get("metadata", {})),
        )

    def snapshot(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def describe(self) -> dict[str, Any]:
        counts = Counter(entry.kind for entry in self.entries)
        latest_entry = self.entries[-1] if self.entries else None
        latest_compaction = (
            max(self.compactions.values(), key=lambda item: (item.created_at, item.compaction_id))
            if self.compactions
            else None
        )
        return {
            "journal_id": self.journal_id,
            "session_id": self.session_id,
            "entries": len(self.entries),
            "branches": len(self.branches),
            "branch_names": [branch.name for branch in self.list_branches()],
            "compactions": len(self.compactions),
            "latest_entry_id": latest_entry.entry_id if latest_entry is not None else None,
            "latest_entry_kind": latest_entry.kind if latest_entry is not None else None,
            "latest_compaction_id": (
                latest_compaction.compaction_id if latest_compaction is not None else None
            ),
            "entry_kinds": dict(sorted(counts.items())),
        }

    def get_entry(self, entry_id: str) -> SessionJournalEntry | None:
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise InterfaceFrameworkError("entry_id must be a non-empty string")
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_branch(self, branch_id: str) -> SessionJournalBranch | None:
        if not isinstance(branch_id, str) or not branch_id.strip():
            raise InterfaceFrameworkError("branch_id must be a non-empty string")
        return self.branches.get(branch_id)

    def list_branches(self) -> tuple[SessionJournalBranch, ...]:
        return tuple(
            sorted(
                self.branches.values(),
                key=lambda item: (item.updated_at, item.name, item.branch_id),
            )
        )

    def entries_for_branch(self, branch_id: str | None = None) -> tuple[SessionJournalEntry, ...]:
        if branch_id is None:
            return self.entries
        branch = self.get_branch(branch_id)
        if branch is None:
            raise InterfaceFrameworkError(f"unknown journal branch: {branch_id!r}")
        by_id = {entry.entry_id: entry for entry in self.entries}
        return tuple(by_id[entry_id] for entry_id in branch.entry_ids if entry_id in by_id)

    def append(
        self,
        *,
        kind: str,
        summary: str,
        branch_id: str = "main",
        thread_id: str | None = None,
        workstream_id: str | None = None,
        parent_entry_id: str | None = None,
        severity: str = "info",
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: tuple[str, ...] | list[str] | tuple[Any, ...] | list[Any] = (),
    ) -> SessionJournalEntry:
        if not isinstance(kind, str) or not kind.strip():
            raise InterfaceFrameworkError("kind must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise InterfaceFrameworkError("summary must be a non-empty string")
        if branch_id not in self.branches:
            raise InterfaceFrameworkError(f"unknown journal branch: {branch_id!r}")
        branch = self.branches[branch_id]
        if severity not in _VALID_SEVERITIES:
            raise InterfaceFrameworkError(
                f"severity must be one of {sorted(_VALID_SEVERITIES)}, got {severity!r}"
            )
        if parent_entry_id is None:
            parent_entry_id = branch.head_entry_id
        elif branch.head_entry_id is not None and parent_entry_id != branch.head_entry_id:
            raise InterfaceFrameworkError(
                "parent_entry_id must match the current branch head when provided"
            )
        if parent_entry_id is not None and self.get_entry(parent_entry_id) is None:
            raise InterfaceFrameworkError(f"unknown parent journal entry: {parent_entry_id!r}")
        now = _utc_now()
        entry = SessionJournalEntry(
            entry_id=f"journal-entry-{uuid.uuid4()}",
            session_id=self.session_id,
            kind=kind,
            summary=summary,
            created_at=now,
            branch_id=branch_id,
            thread_id=thread_id,
            workstream_id=workstream_id,
            parent_entry_id=parent_entry_id,
            severity=severity,
            payload=dict(payload or {}),
            metadata=dict(metadata or {}),
            tags=_dedupe_strings(tags),
        )
        self.entries = self.entries + (entry,)
        updated_branch = replace(
            branch,
            base_entry_id=branch.base_entry_id if branch.base_entry_id is not None else parent_entry_id,
            head_entry_id=entry.entry_id,
            entry_ids=tuple(dict.fromkeys(branch.entry_ids + (entry.entry_id,))),
            updated_at=now,
        )
        self.branches[branch_id] = updated_branch
        self.updated_at = now
        return entry

    def branch(
        self,
        name: str,
        *,
        from_entry_id: str | None = None,
        source_branch_id: str = "main",
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionJournalBranch:
        if not isinstance(name, str) or not name.strip():
            raise InterfaceFrameworkError("name must be a non-empty string")
        if source_branch_id not in self.branches:
            raise InterfaceFrameworkError(f"unknown source journal branch: {source_branch_id!r}")
        source_branch = self.branches[source_branch_id]
        base_entry_id = from_entry_id if from_entry_id is not None else source_branch.head_entry_id
        if base_entry_id is not None and self.get_entry(base_entry_id) is None:
            raise InterfaceFrameworkError(f"unknown journal entry: {base_entry_id!r}")
        branch_id = f"journal-branch-{uuid.uuid4()}"
        now = _utc_now()
        self.branches[branch_id] = SessionJournalBranch(
            branch_id=branch_id,
            session_id=self.session_id,
            name=name,
            base_entry_id=None,
            head_entry_id=None,
            entry_ids=tuple(),
            created_at=now,
            updated_at=now,
            metadata={
                **dict(metadata or {}),
                "source_branch_id": source_branch_id,
                "source_entry_id": base_entry_id,
            },
        )
        self.append(
            kind="journal.branch.created",
            summary=f"Created journal branch {name}",
            branch_id=branch_id,
            parent_entry_id=base_entry_id,
            severity="info",
            payload={
                "branch_id": branch_id,
                "branch_name": name,
                "source_branch_id": source_branch_id,
                "source_entry_id": base_entry_id,
            },
            metadata=dict(metadata or {}),
            tags=("branch", "history"),
        )
        return self.branches[branch_id]

    def compact(
        self,
        *,
        branch_id: str = "main",
        keep_last_n: int = 4,
        policy: str = "oldest_first",
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionJournalCompaction:
        if not isinstance(keep_last_n, int) or keep_last_n < 0:
            raise InterfaceFrameworkError("keep_last_n must be a non-negative int")
        if policy not in _VALID_COMPACTION_POLICIES:
            raise InterfaceFrameworkError(
                f"policy must be one of {sorted(_VALID_COMPACTION_POLICIES)}, got {policy!r}"
            )
        entries = list(self.entries_for_branch(branch_id))
        if not entries:
            if branch_id not in self.branches:
                raise InterfaceFrameworkError(f"unknown journal branch: {branch_id!r}")
            now = _utc_now()
            compaction = SessionJournalCompaction(
                compaction_id=f"journal-compaction-{uuid.uuid4()}",
                session_id=self.session_id,
                branch_id=branch_id,
                created_at=now,
                policy=policy,
                keep_last_n=keep_last_n,
                dropped_entry_ids=tuple(),
                retained_entry_ids=tuple(),
                summary_entry_id=None,
                summary_text="",
                facts=tuple(),
                metadata=dict(metadata or {}),
            )
            self.compactions[compaction.compaction_id] = compaction
            self.updated_at = now
            return compaction
        if len(entries) <= keep_last_n:
            now = _utc_now()
            compaction = SessionJournalCompaction(
                compaction_id=f"journal-compaction-{uuid.uuid4()}",
                session_id=self.session_id,
                branch_id=branch_id,
                created_at=now,
                policy=policy,
                keep_last_n=keep_last_n,
                dropped_entry_ids=tuple(),
                retained_entry_ids=tuple(entry.entry_id for entry in entries),
                summary_entry_id=None,
                summary_text="",
                facts=tuple(),
                metadata=dict(metadata or {}),
            )
            self.compactions[compaction.compaction_id] = compaction
            self.updated_at = now
            return compaction

        turns = [_turn_for_entry(entry) for entry in entries]
        tail_turns = turns[-keep_last_n:] if keep_last_n > 0 else []
        tail_tokens = sum(_count_tokens(turn.content) for turn in tail_turns)
        compactor = ContextCompactor(
            summarize_fn=_summarize_turns,
            token_counter=_count_tokens,
            target_tokens=max(1, tail_tokens + 1),
            keep_last_n=keep_last_n,
            policy=policy,
        )
        compacted_turns = compactor.compact(turns)
        summary_turn = next(
            (
                turn
                for turn in compacted_turns
                if turn.kind == "system" and turn.content.startswith("[CONTEXT SUMMARY")
            ),
            None,
        )
        dropped_turns = turns[:-keep_last_n] if keep_last_n > 0 else list(turns)
        facts = tuple(compactor.extract_facts(dropped_turns))
        retained_entry_ids = tuple(
            entry.entry_id
            for entry, turn in zip(entries, turns)
            if any(compacted is turn for compacted in compacted_turns)
        )
        dropped_entry_ids = tuple(
            entry.entry_id for entry in entries if entry.entry_id not in retained_entry_ids
        )
        summary_text = summary_turn.content if summary_turn is not None else ""
        summary_entry = self.append(
            kind="journal.compaction",
            summary=f"Compacted {len(dropped_entry_ids)} journal entries on branch {branch_id}",
            branch_id=branch_id,
            parent_entry_id=self.branches[branch_id].head_entry_id,
            severity="info",
            payload={
                "branch_id": branch_id,
                "policy": policy,
                "keep_last_n": keep_last_n,
                "dropped_entry_ids": list(dropped_entry_ids),
                "retained_entry_ids": list(retained_entry_ids),
                "summary_text": summary_text,
                "facts": list(facts),
            },
            metadata=dict(metadata or {}),
            tags=("compaction", "history"),
        )
        now = _utc_now()
        compaction = SessionJournalCompaction(
            compaction_id=f"journal-compaction-{uuid.uuid4()}",
            session_id=self.session_id,
            branch_id=branch_id,
            created_at=now,
            policy=policy,
            keep_last_n=keep_last_n,
            dropped_entry_ids=dropped_entry_ids,
            retained_entry_ids=retained_entry_ids,
            summary_entry_id=summary_entry.entry_id,
            summary_text=summary_text,
            facts=facts,
            metadata=dict(metadata or {}),
        )
        self.compactions[compaction.compaction_id] = compaction
        self.updated_at = now
        return compaction
