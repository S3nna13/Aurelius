"""Local-first persistent sessions and named workstreams for Aurelius.

The session manager is deliberately small: it persists JSON-safe session
snapshots to disk, keeps an in-memory index for the current process, and
provides explicit helpers for workstreams, queued work items, threads,
approvals, checkpoints, background jobs, messages, and tool-call audit
records. It does not launch network services or background daemons.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.model.interface_framework import (
    ApprovalRequest,
    BackgroundJob,
    Checkpoint,
    InterfaceFrameworkError,
    MessageEnvelope,
    SkillBundle,
    TaskThread,
    Workstream,
)
from .session_journal import (
    SessionJournal,
    SessionJournalBranch,
    SessionJournalCompaction,
    SessionJournalEntry,
)

__all__ = [
    "WorkItem",
    "SessionRecord",
    "SessionManager",
]


_SESSION_STATUSES = frozenset({"active", "paused", "completed", "canceled", "failed"})
_WORKSTREAM_STATUSES = frozenset({"draft", "active", "blocked", "completed", "failed", "canceled"})
_WORK_ITEM_STATUSES = frozenset({"queued", "running", "completed", "failed", "canceled"})
_BACKGROUND_JOB_STATUSES = frozenset({"pending", "running", "completed", "failed", "canceled"})
_SESSION_EXPORT_FORMAT = "aurelius.session.export"
_SESSION_EXPORT_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
    return value


def _coerce_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError(f"{field_name} must be a non-empty string or None")
    return value


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _skill_from_payload(payload: Mapping[str, Any]) -> SkillBundle:
    return SkillBundle(
        skill_id=_require_non_empty(str(payload["skill_id"]), "skill_id"),
        name=_require_non_empty(str(payload.get("name", payload["skill_id"])), "name"),
        description=str(payload.get("description", "")),
        scope=str(payload.get("scope", "thread")),
        instructions=str(payload.get("instructions", "")),
        scripts=tuple(payload.get("scripts", ())),
        resources=tuple(payload.get("resources", ())),
        entrypoints=tuple(payload.get("entrypoints", ())),
        version=payload.get("version"),
        provenance=payload.get("provenance"),
        source_path=payload.get("source_path"),
        metadata=dict(payload.get("metadata", {})),
    )


def _message_from_payload(payload: Mapping[str, Any]) -> MessageEnvelope:
    return MessageEnvelope(
        envelope_id=_require_non_empty(str(payload["envelope_id"]), "envelope_id"),
        channel_id=_require_non_empty(str(payload["channel_id"]), "channel_id"),
        thread_id=payload.get("thread_id"),
        sender=_require_non_empty(str(payload.get("sender", "gateway")), "sender"),
        recipient=payload.get("recipient"),
        kind=_require_non_empty(str(payload.get("kind", "message")), "kind"),
        payload=dict(payload.get("payload", {})),
        created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
        session_id=payload.get("session_id"),
        workstream_id=payload.get("workstream_id"),
        workspace=payload.get("workspace"),
        metadata=dict(payload.get("metadata", {})),
    )


def _thread_from_payload(payload: Mapping[str, Any]) -> TaskThread:
    skills = tuple(_skill_from_payload(item) for item in payload.get("skills", ()))
    messages = tuple(
        _message_from_payload(item) for item in payload.get("message_history", ())
    )
    return TaskThread(
        thread_id=_require_non_empty(str(payload["thread_id"]), "thread_id"),
        title=_require_non_empty(str(payload["title"]), "title"),
        mode=_require_non_empty(str(payload["mode"]), "mode"),
        status=_require_non_empty(str(payload["status"]), "status"),
        host=_require_non_empty(str(payload["host"]), "host"),
        session_id=payload.get("session_id"),
        workstream_id=payload.get("workstream_id"),
        workstream_name=payload.get("workstream_name"),
        workspace=payload.get("workspace"),
        workspace_roots=tuple(payload.get("workspace_roots", ())),
        channel=payload.get("channel"),
        repo_instructions=payload.get("repo_instructions"),
        workspace_instructions=payload.get("workspace_instructions"),
        instruction_stack=tuple(payload.get("instruction_stack", ())),
        skills=skills,
        approvals=tuple(payload.get("approvals", ())),
        checkpoints=tuple(payload.get("checkpoints", ())),
        steps=tuple(payload.get("steps", ())),
        created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
        updated_at=_require_non_empty(str(payload.get("updated_at", _utc_now())), "updated_at"),
        parent_thread_id=payload.get("parent_thread_id"),
        parent_checkpoint_id=payload.get("parent_checkpoint_id"),
        lineage=tuple(payload.get("lineage", ())),
        task_prompt=_require_non_empty(str(payload.get("task_prompt", "")), "task_prompt"),
        memory_summary=str(payload.get("memory_summary", "")),
        last_model_response=payload.get("last_model_response"),
        last_tool_result=payload.get("last_tool_result"),
        active_job_ids=tuple(payload.get("active_job_ids", ())),
        message_history=messages,
        metadata=dict(payload.get("metadata", {})),
    )


def _workstream_from_payload(payload: Mapping[str, Any]) -> Workstream:
    messages = tuple(
        _message_from_payload(item) for item in payload.get("messages", ())
    )
    queued_items = tuple(dict(item) for item in payload.get("queued_items", ()))
    return Workstream(
        workstream_id=_require_non_empty(str(payload["workstream_id"]), "workstream_id"),
        session_id=_require_non_empty(str(payload["session_id"]), "session_id"),
        name=_require_non_empty(str(payload["name"]), "name"),
        status=_require_non_empty(str(payload.get("status", "draft")), "status"),
        current_thread_id=payload.get("current_thread_id"),
        thread_ids=tuple(payload.get("thread_ids", ())),
        queued_items=queued_items,
        messages=messages,
        checkpoint_ids=tuple(payload.get("checkpoint_ids", ())),
        workspace=payload.get("workspace"),
        created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
        updated_at=_require_non_empty(str(payload.get("updated_at", _utc_now())), "updated_at"),
        metadata=dict(payload.get("metadata", {})),
    )


def _checkpoint_from_payload(payload: Mapping[str, Any]) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=_require_non_empty(str(payload["checkpoint_id"]), "checkpoint_id"),
        thread_id=_require_non_empty(str(payload["thread_id"]), "thread_id"),
        created_at=_require_non_empty(str(payload["created_at"]), "created_at"),
        lineage=tuple(payload.get("lineage", ())),
        thread_snapshot=dict(payload.get("thread_snapshot", {})),
        contract_metadata=dict(payload.get("contract_metadata", {})),
        model_context=dict(payload.get("model_context", {})),
        memory_summary=_require_non_empty(str(payload.get("memory_summary", "")), "memory_summary"),
        last_model_response=payload.get("last_model_response"),
        last_tool_result=payload.get("last_tool_result"),
        session_id=payload.get("session_id"),
        workstream_id=payload.get("workstream_id"),
        workstream_name=payload.get("workstream_name"),
        pending_approval_ids=tuple(payload.get("pending_approval_ids", ())),
        active_job_ids=tuple(payload.get("active_job_ids", ())),
        tool_observations=tuple(dict(item) for item in payload.get("tool_observations", ())),
    )


@dataclass(frozen=True)
class WorkItem:
    """Queued session work item."""

    item_id: str
    session_id: str
    workstream_id: str
    kind: str
    title: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: str = ""
    updated_at: str = ""
    thread_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("item_id", "session_id", "workstream_id", "kind", "title", "created_at", "updated_at"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.status not in _WORK_ITEM_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_WORK_ITEM_STATUSES)}, got {self.status!r}"
            )
        if self.thread_id is not None and not isinstance(self.thread_id, str):
            raise InterfaceFrameworkError("thread_id must be str or None")
        if not isinstance(self.payload, dict):
            raise InterfaceFrameworkError("payload must be a dict")
        if self.result is not None and not isinstance(self.result, dict):
            raise InterfaceFrameworkError("result must be dict or None")
        if self.error is not None and not isinstance(self.error, str):
            raise InterfaceFrameworkError("error must be str or None")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass
class SessionRecord:
    """JSON-safe persistent session snapshot."""

    session_id: str
    workspace: str | None
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    active_thread_id: str | None = None
    active_workstream_id: str | None = None
    memory_summary: str = ""
    workstreams: dict[str, Workstream] = field(default_factory=dict)
    threads: dict[str, TaskThread] = field(default_factory=dict)
    approvals: dict[str, ApprovalRequest] = field(default_factory=dict)
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    jobs: dict[str, BackgroundJob] = field(default_factory=dict)
    messages: dict[str, MessageEnvelope] = field(default_factory=dict)
    tool_calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    queue: list[WorkItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.session_id, "session_id")
        _require_non_empty(self.created_at, "created_at")
        _require_non_empty(self.updated_at, "updated_at")
        if self.status not in _SESSION_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_SESSION_STATUSES)}, got {self.status!r}"
            )
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise InterfaceFrameworkError("workspace must be str or None")
        if self.active_thread_id is not None and not isinstance(self.active_thread_id, str):
            raise InterfaceFrameworkError("active_thread_id must be str or None")
        if self.active_workstream_id is not None and not isinstance(self.active_workstream_id, str):
            raise InterfaceFrameworkError("active_workstream_id must be str or None")
        if not isinstance(self.workstreams, dict):
            raise InterfaceFrameworkError("workstreams must be a dict")
        if not isinstance(self.threads, dict):
            raise InterfaceFrameworkError("threads must be a dict")
        if not isinstance(self.approvals, dict):
            raise InterfaceFrameworkError("approvals must be a dict")
        if not isinstance(self.checkpoints, dict):
            raise InterfaceFrameworkError("checkpoints must be a dict")
        if not isinstance(self.jobs, dict):
            raise InterfaceFrameworkError("jobs must be a dict")
        if not isinstance(self.messages, dict):
            raise InterfaceFrameworkError("messages must be a dict")
        if not isinstance(self.tool_calls, dict):
            raise InterfaceFrameworkError("tool_calls must be a dict")
        if not isinstance(self.queue, list):
            raise InterfaceFrameworkError("queue must be a list")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionRecord":
        if not isinstance(payload, Mapping):
            raise InterfaceFrameworkError("session payload must be a mapping")
        workstreams = {
            key: _workstream_from_payload(value)
            for key, value in dict(payload.get("workstreams", {})).items()
        }
        threads = {
            key: _thread_from_payload(value)
            for key, value in dict(payload.get("threads", {})).items()
        }
        approvals = {
            key: ApprovalRequest(
                approval_id=_require_non_empty(str(value["approval_id"]), "approval_id"),
                thread_id=_require_non_empty(str(value["thread_id"]), "thread_id"),
                category=_require_non_empty(str(value["category"]), "category"),
                action_summary=_require_non_empty(str(value["action_summary"]), "action_summary"),
                affected_resources=tuple(value.get("affected_resources", ())),
                reason=_require_non_empty(str(value["reason"]), "reason"),
                reversible=value.get("reversible", False),
                minimum_scope=_require_non_empty(str(value.get("minimum_scope", "allow_once")), "minimum_scope"),
                decision=str(value.get("decision", "pending")),
                created_at=_require_non_empty(str(value.get("created_at", _utc_now())), "created_at"),
                decided_at=value.get("decided_at"),
                metadata=dict(value.get("metadata", {})),
            )
            for key, value in dict(payload.get("approvals", {})).items()
        }
        checkpoints = {
            key: _checkpoint_from_payload(value)
            for key, value in dict(payload.get("checkpoints", {})).items()
        }
        jobs = {
            key: BackgroundJob(**value)
            for key, value in dict(payload.get("jobs", {})).items()
        }
        messages = {
            key: _message_from_payload(value)
            for key, value in dict(payload.get("messages", {})).items()
        }
        queue = [WorkItem(**value) for value in payload.get("queue", [])]
        return cls(
            session_id=_require_non_empty(str(payload["session_id"]), "session_id"),
            workspace=payload.get("workspace"),
            status=str(payload.get("status", "active")),
            created_at=_require_non_empty(str(payload.get("created_at", _utc_now())), "created_at"),
            updated_at=_require_non_empty(str(payload.get("updated_at", _utc_now())), "updated_at"),
            active_thread_id=payload.get("active_thread_id"),
            active_workstream_id=payload.get("active_workstream_id"),
            memory_summary=str(payload.get("memory_summary", "")),
            workstreams=workstreams,
            threads=threads,
            approvals=approvals,
            checkpoints=checkpoints,
            jobs=jobs,
            messages=messages,
            tool_calls={k: [dict(entry) for entry in v] for k, v in dict(payload.get("tool_calls", {})).items()},
            queue=queue,
            metadata=dict(payload.get("metadata", {})),
        )


class SessionManager:
    """Manage local-first persistent Aurelius sessions and workstreams."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        root_dir: str | Path | None = None,
        persist: bool = True,
    ) -> None:
        self.persist = bool(persist)
        self.root_dir = Path(root_dir).expanduser().resolve() if root_dir is not None else None
        if state_dir is None:
            if self.root_dir is not None:
                resolved_state = self.root_dir / ".aurelius" / "sessions"
            else:
                resolved_state = Path.home() / ".aurelius" / "sessions"
        else:
            resolved_state = Path(state_dir).expanduser().resolve()
        self.state_dir = resolved_state
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._journal_dir = self.state_dir / "journals"
        self._journal_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionRecord] = {}
        self._journals: dict[str, SessionJournal] = {}

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------
    def create_session(
        self,
        session_id: str | None = None,
        *,
        workspace: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionRecord:
        if session_id is None:
            session_id = f"session-{uuid.uuid4()}"
        if session_id in self._sessions:
            raise InterfaceFrameworkError(f"session already exists: {session_id!r}")
        created_at = _utc_now()
        record = SessionRecord(
            session_id=session_id,
            workspace=_coerce_optional_text(workspace, "workspace"),
            created_at=created_at,
            updated_at=created_at,
            metadata=dict(metadata or {}),
        )
        self._sessions[session_id] = record
        self._persist(record)
        self._record_journal_entry(
            session_id,
            kind="session.created",
            summary=f"Created session {session_id}",
            payload={
                "session": self.snapshot(record),
            },
        )
        return record

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        journal = self.get_journal(session_id, create=False)
        if journal is None:
            journal = SessionJournal.create(
                session.session_id,
                created_at=session.created_at,
                metadata={"workspace": session.workspace},
            )
        return {
            "format": _SESSION_EXPORT_FORMAT,
            "schema_version": _SESSION_EXPORT_SCHEMA_VERSION,
            "exported_at": _utc_now(),
            "session_id": session.session_id,
            "state_dir": str(self.state_dir),
            "session": self.snapshot(session),
            "journal": journal.snapshot(),
        }

    def write_session_export(self, session_id: str, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.export_session(session_id)
        try:
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise InterfaceFrameworkError(f"cannot write session export: {target}") from exc
        return target

    def import_session_export(
        self,
        payload_or_path: Mapping[str, Any] | str | Path,
        *,
        replace: bool = False,
    ) -> SessionRecord:
        if isinstance(payload_or_path, (str, Path)):
            path = Path(payload_or_path).expanduser().resolve()
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except OSError as exc:  # pragma: no cover - filesystem failure
                raise InterfaceFrameworkError(f"cannot read session export: {path}") from exc
            except json.JSONDecodeError as exc:
                raise InterfaceFrameworkError(f"session export is not valid JSON: {path}") from exc
        elif isinstance(payload_or_path, Mapping):
            payload = dict(payload_or_path)
        else:
            raise InterfaceFrameworkError(
                "payload_or_path must be a mapping or a path to a JSON export"
            )
        if not isinstance(payload, dict):
            raise InterfaceFrameworkError("session export payload must be an object")
        if payload.get("format") != _SESSION_EXPORT_FORMAT:
            raise InterfaceFrameworkError("session export format is not recognized")
        if payload.get("schema_version") != _SESSION_EXPORT_SCHEMA_VERSION:
            raise InterfaceFrameworkError("session export schema_version is not supported")
        session_payload = payload.get("session")
        journal_payload = payload.get("journal")
        if not isinstance(session_payload, dict):
            raise InterfaceFrameworkError("session export missing session payload")
        if not isinstance(journal_payload, dict):
            raise InterfaceFrameworkError("session export missing journal payload")
        session = SessionRecord.from_dict(session_payload)
        journal = SessionJournal.from_dict(journal_payload)
        if session.session_id != journal.session_id:
            raise InterfaceFrameworkError("session export session and journal ids do not match")
        existing_session = self.get_session(session.session_id)
        if existing_session is not None and not replace:
            raise InterfaceFrameworkError(f"session already exists: {session.session_id!r}")
        self._sessions[session.session_id] = session
        self._journals[session.session_id] = journal
        self._persist(session)
        self._persist_journal(journal)
        return session

    def ensure_session(
        self,
        session_id: str | None = None,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionRecord:
        if session_id is None:
            return self.create_session(
                workspace=_coerce_optional_text(workspace, "workspace"),
                metadata=metadata,
            )
        session = self.get_session(session_id)
        if session is None:
            return self.create_session(
                session_id=session_id,
                workspace=_coerce_optional_text(workspace, "workspace"),
                metadata=metadata,
            )
        updated = False
        if workspace is not None:
            normalized_workspace = _coerce_optional_text(workspace, "workspace")
            if session.workspace != normalized_workspace:
                session.workspace = normalized_workspace
                updated = True
        if metadata:
            session.metadata.update(dict(metadata))
            updated = True
        if updated:
            session.status = "active"
            session.updated_at = _utc_now()
            self._persist(session)
            self._record_journal_entry(
                session_id,
                kind="session.updated",
                summary=f"Updated session {session_id}",
                payload={
                    "workspace": session.workspace,
                    "metadata": _json_safe(session.metadata),
                },
            )
        return session

    def get_session(self, session_id: str) -> SessionRecord | None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise InterfaceFrameworkError("session_id must be a non-empty string")
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        path = self._session_path(session_id)
        if not path.exists():
            return None
        session = self._load_session(path)
        self._sessions[session_id] = session
        return session

    def reload_session(self, session_id: str) -> SessionRecord:
        path = self._session_path(session_id)
        if not path.exists():
            raise InterfaceFrameworkError(f"unknown session: {session_id!r}")
        session = self._load_session(path)
        self._sessions[session_id] = session
        return session

    def resume_session(self, session_id: str) -> SessionRecord:
        return self.set_session_status(session_id, "active")

    def pause_session(self, session_id: str) -> SessionRecord:
        return self.set_session_status(session_id, "paused")

    def set_session_status(self, session_id: str, status: str) -> SessionRecord:
        if status not in _SESSION_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_SESSION_STATUSES)}, got {status!r}"
            )
        session = self._require_session(session_id)
        session.status = status
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="session.status.changed",
            summary=f"Set session {session_id} status to {status}",
            payload={"status": status},
        )
        return replace(session)

    def list_sessions(self) -> list[SessionRecord]:
        sessions = list(self._sessions.values())
        if not sessions:
            for path in sorted(self.state_dir.glob("*.json")):
                session = self._load_session(path)
                self._sessions[session.session_id] = session
            sessions = list(self._sessions.values())
        return sorted(sessions, key=lambda item: (item.updated_at, item.session_id))

    def session_count(self) -> int:
        return len(self.list_sessions())

    # ------------------------------------------------------------------
    # workstreams
    # ------------------------------------------------------------------
    def create_workstream(
        self,
        session_id: str,
        name: str,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workstream:
        session = self._require_session(session_id)
        workstream_id = f"workstream-{uuid.uuid4()}"
        created_at = _utc_now()
        workstream = Workstream(
            workstream_id=workstream_id,
            session_id=session.session_id,
            name=name,
            status="active",
            workspace=_coerce_optional_text(workspace, "workspace") or session.workspace,
            created_at=created_at,
            updated_at=created_at,
            metadata=dict(metadata or {}),
        )
        session.workstreams[workstream_id] = workstream
        session.active_workstream_id = workstream_id
        session.updated_at = created_at
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="workstream.created",
            summary=f"Created workstream {name}",
            workstream_id=workstream_id,
            payload={"workstream": _json_safe(asdict(workstream))},
        )
        return workstream

    def ensure_workstream(
        self,
        session_id: str,
        name_or_id: str,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workstream:
        session = self._require_session(session_id)
        workstream = self._resolve_workstream(session, name_or_id)
        if workstream is None:
            return self.create_workstream(
                session_id,
                name_or_id,
                workspace=workspace,
                metadata=metadata,
            )
        updated = False
        normalized_workspace = _coerce_optional_text(workspace, "workspace")
        if normalized_workspace is not None and workstream.workspace != normalized_workspace:
            workstream = replace(workstream, workspace=normalized_workspace)
            updated = True
        if metadata:
            workstream = replace(workstream, metadata={**workstream.metadata, **dict(metadata)})
            updated = True
        if updated:
            workstream = replace(workstream, updated_at=_utc_now())
            session.workstreams[workstream.workstream_id] = workstream
            session.updated_at = _utc_now()
            self._persist(session)
            self._record_journal_entry(
                session_id,
                kind="workstream.updated",
                summary=f"Updated workstream {workstream.name}",
                workstream_id=workstream.workstream_id,
                payload={"workstream": _json_safe(asdict(workstream))},
            )
        return workstream

    def get_workstream(
        self,
        session_id: str,
        workstream_id: str,
        *,
        missing_ok: bool = False,
    ) -> Workstream | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        workstream = session.workstreams.get(workstream_id)
        if workstream is not None:
            return workstream
        for candidate in session.workstreams.values():
            if candidate.name == workstream_id:
                return candidate
        if missing_ok:
            return None
        return None

    def set_workstream_status(self, session_id: str, workstream_id_or_name: str, status: str) -> Workstream:
        if status not in _WORKSTREAM_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_WORKSTREAM_STATUSES)}, got {status!r}"
            )
        session = self._require_session(session_id)
        workstream = self._resolve_workstream(session, workstream_id_or_name)
        if workstream is None:
            raise InterfaceFrameworkError(f"unknown workstream: {workstream_id_or_name!r}")
        updated = replace(workstream, status=status, updated_at=_utc_now())
        session.workstreams[updated.workstream_id] = updated
        if status == "active":
            session.active_workstream_id = updated.workstream_id
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="workstream.status.changed",
            summary=f"Set workstream {updated.name} status to {status}",
            workstream_id=updated.workstream_id,
            payload={"status": status, "workstream": _json_safe(asdict(updated))},
        )
        return updated

    def list_workstreams(self, session_id: str) -> list[Workstream]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return sorted(session.workstreams.values(), key=lambda item: (item.updated_at, item.name))

    # ------------------------------------------------------------------
    # journal
    # ------------------------------------------------------------------
    def get_journal(self, session_id: str, *, create: bool = True) -> SessionJournal | None:
        session = self._require_session(session_id)
        journal = self._journals.get(session_id)
        if journal is not None:
            return journal
        path = self._journal_path(session_id)
        if path.exists():
            journal = self._load_journal(path)
            self._journals[session_id] = journal
            return journal
        if not create:
            return None
        journal = SessionJournal.create(
            session_id,
            created_at=session.created_at,
            metadata={"workspace": session.workspace},
        )
        self._journals[session_id] = journal
        self._persist_journal(journal)
        return journal

    def append_journal_entry(
        self,
        session_id: str,
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
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        entry = journal.append(
            kind=kind,
            summary=summary,
            branch_id=branch_id,
            thread_id=thread_id,
            workstream_id=workstream_id,
            parent_entry_id=parent_entry_id,
            severity=severity,
            payload=payload,
            metadata=metadata,
            tags=tags,
        )
        self._persist_journal(journal)
        self._touch_session(session_id)
        return entry

    def branch_journal(
        self,
        session_id: str,
        name: str,
        *,
        from_entry_id: str | None = None,
        source_branch_id: str = "main",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        branch = journal.branch(
            name,
            from_entry_id=from_entry_id,
            source_branch_id=source_branch_id,
            metadata=metadata,
        )
        self._persist_journal(journal)
        self._touch_session(session_id)
        return {
            "branch": _json_safe(asdict(branch)),
            "anchor_entry": (
                _json_safe(asdict(journal.get_entry(branch.head_entry_id)))
                if branch.head_entry_id is not None
                else None
            ),
        }

    def compact_journal(
        self,
        session_id: str,
        *,
        branch_id: str = "main",
        keep_last_n: int = 4,
        policy: str = "oldest_first",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        compaction = journal.compact(
            branch_id=branch_id,
            keep_last_n=keep_last_n,
            policy=policy,
            metadata=metadata,
        )
        self._persist_journal(journal)
        self._touch_session(session_id)
        return {
            "compaction": _json_safe(asdict(compaction)),
            "entry": (
                _json_safe(asdict(journal.get_entry(compaction.summary_entry_id)))
                if compaction.summary_entry_id is not None
                else None
            ),
        }

    def list_journal_entries(
        self,
        session_id: str,
        *,
        branch_id: str | None = None,
    ) -> list[SessionJournalEntry]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        return list(journal.entries_for_branch(branch_id))

    def list_journal_branches(self, session_id: str) -> list[SessionJournalBranch]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        return list(journal.list_branches())

    def get_journal_entry(self, session_id: str, entry_id: str) -> SessionJournalEntry | None:
        journal = self.get_journal(session_id, create=False)
        if journal is None:
            return None
        return journal.get_entry(entry_id)

    def list_journal_compactions(
        self,
        session_id: str,
        *,
        branch_id: str | None = None,
    ) -> list[SessionJournalCompaction]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        compactions = list(journal.compactions.values())
        if branch_id is not None:
            compactions = [item for item in compactions if item.branch_id == branch_id]
        return sorted(compactions, key=lambda item: (item.created_at, item.compaction_id))

    def journal_summary(self, session_id: str) -> dict[str, Any]:
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        branches = self.list_journal_branches(session_id)
        compactions = self.list_journal_compactions(session_id)
        latest_compaction = compactions[-1] if compactions else None
        return {
            **journal.describe(),
            "branches": [
                {
                    "branch_id": branch.branch_id,
                    "name": branch.name,
                    "head_entry_id": branch.head_entry_id,
                    "base_entry_id": branch.base_entry_id,
                    "entry_count": len(branch.entry_ids),
                    "metadata": _json_safe(branch.metadata),
                }
                for branch in branches
            ],
            "compactions": [
                {
                    "compaction_id": compaction.compaction_id,
                    "branch_id": compaction.branch_id,
                    "policy": compaction.policy,
                    "keep_last_n": compaction.keep_last_n,
                    "dropped_count": len(compaction.dropped_entry_ids),
                    "retained_count": len(compaction.retained_entry_ids),
                    "summary_entry_id": compaction.summary_entry_id,
                    "created_at": compaction.created_at,
                }
                for compaction in compactions
            ],
            "latest_compaction": _json_safe(asdict(latest_compaction)) if latest_compaction is not None else None,
        }

    # ------------------------------------------------------------------
    # thread / approval / checkpoint / message / tool-call records
    # ------------------------------------------------------------------
    def register_thread(
        self,
        session_id: str,
        thread: TaskThread,
        *,
        workstream_id: str | None = None,
    ) -> TaskThread:
        session = self._require_session(session_id)
        if not isinstance(thread, TaskThread):
            raise InterfaceFrameworkError(f"thread must be TaskThread, got {type(thread).__name__}")
        session.threads[thread.thread_id] = thread
        session.active_thread_id = thread.thread_id
        workstream = self._resolve_workstream(session, thread.workstream_id or workstream_id)
        if workstream is not None:
            thread_ids = tuple(dict.fromkeys(workstream.thread_ids + (thread.thread_id,)))
            session.workstreams[workstream.workstream_id] = replace(
                workstream,
                current_thread_id=thread.thread_id,
                thread_ids=thread_ids,
                updated_at=_utc_now(),
            )
            session.active_workstream_id = workstream.workstream_id
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="thread.registered",
            summary=f"Registered thread {thread.title}",
            thread_id=thread.thread_id,
            workstream_id=thread.workstream_id or workstream_id,
            payload={"thread": _json_safe(asdict(thread))},
        )
        return thread

    def update_thread(self, session_id: str, thread: TaskThread) -> TaskThread:
        return self.register_thread(session_id, thread)

    def get_thread(self, session_id: str, thread_id: str) -> TaskThread | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.threads.get(thread_id)

    def list_threads(self, session_id: str) -> list[TaskThread]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return sorted(session.threads.values(), key=lambda item: (item.updated_at, item.thread_id))

    def register_approval(self, session_id: str, approval: ApprovalRequest) -> ApprovalRequest:
        session = self._require_session(session_id)
        session.approvals[approval.approval_id] = approval
        thread = session.threads.get(approval.thread_id)
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                approvals=tuple(dict.fromkeys(thread.approvals + (approval.approval_id,))),
                updated_at=_utc_now(),
            )
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="approval.registered",
            summary=f"Registered approval {approval.approval_id}",
            thread_id=approval.thread_id,
            payload={"approval": _json_safe(asdict(approval))},
        )
        return approval

    def get_approval(self, session_id: str, approval_id: str) -> ApprovalRequest | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.approvals.get(approval_id)

    def register_checkpoint(self, session_id: str, checkpoint: Checkpoint) -> Checkpoint:
        session = self._require_session(session_id)
        session.checkpoints[checkpoint.checkpoint_id] = checkpoint
        thread = session.threads.get(checkpoint.thread_id)
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                checkpoints=tuple(dict.fromkeys(thread.checkpoints + (checkpoint.checkpoint_id,))),
                memory_summary=checkpoint.memory_summary,
                updated_at=_utc_now(),
            )
        workstream = self._resolve_workstream(session, checkpoint.workstream_id)
        if workstream is not None:
            session.workstreams[workstream.workstream_id] = replace(
                workstream,
                checkpoint_ids=tuple(dict.fromkeys(workstream.checkpoint_ids + (checkpoint.checkpoint_id,))),
                updated_at=_utc_now(),
            )
        session.memory_summary = checkpoint.memory_summary
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="checkpoint.registered",
            summary=f"Registered checkpoint {checkpoint.checkpoint_id}",
            thread_id=checkpoint.thread_id,
            workstream_id=checkpoint.workstream_id,
            payload={"checkpoint": _json_safe(asdict(checkpoint))},
        )
        return checkpoint

    def get_checkpoint(self, session_id: str, checkpoint_id: str) -> Checkpoint | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.checkpoints.get(checkpoint_id)

    def register_message(self, session_id: str, message: MessageEnvelope) -> MessageEnvelope:
        session = self._require_session(session_id)
        session.messages[message.envelope_id] = message
        thread = session.threads.get(message.thread_id) if message.thread_id is not None else None
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                message_history=thread.message_history + (message,),
                updated_at=_utc_now(),
            )
        if message.workstream_id is not None:
            workstream = self._resolve_workstream(session, message.workstream_id)
            if workstream is not None:
                session.workstreams[workstream.workstream_id] = replace(
                    workstream,
                    messages=workstream.messages + (message,),
                    updated_at=_utc_now(),
                )
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="message.registered",
            summary=f"Registered message {message.envelope_id}",
            thread_id=message.thread_id,
            workstream_id=message.workstream_id,
            payload={"message": _json_safe(asdict(message))},
        )
        return message

    def get_message(self, session_id: str, envelope_id: str) -> MessageEnvelope | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.messages.get(envelope_id)

    def list_messages(
        self,
        session_id: str,
        *,
        channel_id: str | None = None,
        thread_id: str | None = None,
        workstream_id: str | None = None,
    ) -> list[MessageEnvelope]:
        session = self.get_session(session_id)
        if session is None:
            return []
        messages = list(session.messages.values())
        if channel_id is not None:
            messages = [message for message in messages if message.channel_id == channel_id]
        if thread_id is not None:
            messages = [message for message in messages if message.thread_id == thread_id]
        if workstream_id is not None:
            messages = [message for message in messages if message.workstream_id == workstream_id]
        return sorted(messages, key=lambda item: (item.created_at, item.envelope_id))

    def register_tool_call(self, session_id: str, thread_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        session = self._require_session(session_id)
        session.tool_calls.setdefault(thread_id, []).append(dict(entry))
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="tool_call.recorded",
            summary=f"Recorded tool call {entry.get('tool_name', 'tool')}",
            thread_id=thread_id,
            payload={"tool_call": _json_safe(entry)},
        )
        return entry

    def list_tool_calls(self, session_id: str, thread_id: str | None = None) -> list[dict[str, Any]]:
        session = self.get_session(session_id)
        if session is None:
            return []
        if thread_id is None:
            items: list[dict[str, Any]] = []
            for entries in session.tool_calls.values():
                items.extend(_json_safe(entries))
            return items
        return _json_safe(session.tool_calls.get(thread_id, []))

    # ------------------------------------------------------------------
    # work items / background jobs
    # ------------------------------------------------------------------
    def queue_work_item(
        self,
        session_id: str,
        workstream_name_or_id: str,
        *,
        kind: str,
        title: str | None = None,
        payload: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkItem:
        session = self._require_session(session_id)
        workstream = self._resolve_workstream(session, workstream_name_or_id)
        if workstream is None:
            workstream = self.create_workstream(
                session_id,
                workstream_name_or_id,
                workspace=session.workspace,
            )
            session = self._require_session(session_id)
        item = WorkItem(
            item_id=f"item-{uuid.uuid4()}",
            session_id=session.session_id,
            workstream_id=workstream.workstream_id,
            kind=_require_non_empty(kind, "kind"),
            title=_require_non_empty(title or kind, "title"),
            payload=dict(payload or {}),
            status="queued",
            created_at=_utc_now(),
            updated_at=_utc_now(),
            thread_id=thread_id,
            metadata=dict(metadata or {}),
        )
        session.queue.append(item)
        session.workstreams[workstream.workstream_id] = replace(
            workstream,
            queued_items=workstream.queued_items + (dict(asdict(item)),),
            updated_at=_utc_now(),
        )
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="work_item.queued",
            summary=f"Queued work item {item.title}",
            workstream_id=workstream.workstream_id,
            thread_id=thread_id,
            payload={"work_item": _json_safe(asdict(item))},
        )
        return item

    def update_work_item(
        self,
        session_id: str,
        item_id: str,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> WorkItem:
        session = self._require_session(session_id)
        if status not in _WORK_ITEM_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_WORK_ITEM_STATUSES)}, got {status!r}"
            )
        updated_item: WorkItem | None = None
        new_queue: list[WorkItem] = []
        for item in session.queue:
            if item.item_id != item_id:
                new_queue.append(item)
                continue
            updated_item = replace(
                item,
                status=_require_non_empty(status, "status"),
                updated_at=_utc_now(),
                result=dict(result or {}) if result is not None else item.result,
                error=error,
            )
            new_queue.append(updated_item)
        if updated_item is None:
            raise InterfaceFrameworkError(f"unknown work item: {item_id!r}")
        session.queue = new_queue
        workstream = session.workstreams.get(updated_item.workstream_id)
        if workstream is not None:
            updated_items = tuple(
                dict(asdict(item))
                for item in session.queue
                if item.workstream_id == workstream.workstream_id
            )
            session.workstreams[workstream.workstream_id] = replace(
                workstream,
                queued_items=updated_items,
                updated_at=_utc_now(),
                )
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="work_item.updated",
            summary=f"Updated work item {updated_item.title}",
            workstream_id=updated_item.workstream_id,
            thread_id=updated_item.thread_id,
            payload={"work_item": _json_safe(asdict(updated_item))},
        )
        return updated_item

    def cancel_work_item(self, session_id: str, item_id: str) -> WorkItem:
        return self.update_work_item(session_id, item_id, status="canceled")

    def register_background_job(self, session_id: str, job: BackgroundJob) -> BackgroundJob:
        session = self._require_session(session_id)
        session.jobs[job.job_id] = job
        thread = session.threads.get(job.thread_id)
        if thread is not None:
            session.threads[thread.thread_id] = replace(
                thread,
                active_job_ids=tuple(dict.fromkeys(thread.active_job_ids + (job.job_id,))),
                updated_at=_utc_now(),
            )
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="background_job.registered",
            summary=f"Registered background job {job.job_id}",
            thread_id=job.thread_id,
            payload={"job": _json_safe(asdict(job))},
        )
        return job

    def get_background_job(self, session_id: str, job_id: str) -> BackgroundJob | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.jobs.get(job_id)

    def list_background_jobs(self, session_id: str, workstream_id: str | None = None) -> list[BackgroundJob]:
        session = self.get_session(session_id)
        if session is None:
            return []
        jobs = list(session.jobs.values())
        if workstream_id is not None:
            jobs = [job for job in jobs if job.metadata.get("workstream_id") == workstream_id]
        return sorted(jobs, key=lambda item: (item.updated_at, item.job_id))

    def update_background_job(
        self,
        session_id: str,
        job_id: str,
        *,
        status: str | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> BackgroundJob:
        session = self._require_session(session_id)
        job = session.jobs.get(job_id)
        if job is None:
            raise InterfaceFrameworkError(f"unknown background job: {job_id!r}")
        if status is not None and status not in _BACKGROUND_JOB_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_BACKGROUND_JOB_STATUSES)}, got {status!r}"
            )
        updated = replace(
            job,
            status=_require_non_empty(status, "status") if status is not None else job.status,
            updated_at=_utc_now(),
            result=result if result is not None else job.result,
            metadata={**job.metadata, **({"error": error} if error is not None else {})},
        )
        session.jobs[job_id] = updated
        if updated.status in {"completed", "failed", "canceled"}:
            thread = session.threads.get(updated.thread_id)
            if thread is not None:
                session.threads[thread.thread_id] = replace(
                    thread,
                    active_job_ids=tuple(j for j in thread.active_job_ids if j != job_id),
                    updated_at=_utc_now(),
                )
        session.updated_at = _utc_now()
        self._persist(session)
        self._record_journal_entry(
            session_id,
            kind="background_job.updated",
            summary=f"Updated background job {updated.job_id} to {updated.status}",
            thread_id=updated.thread_id,
            payload={"job": _json_safe(asdict(updated))},
        )
        return updated

    def cancel_background_job(self, session_id: str, job_id: str) -> BackgroundJob:
        return self.update_background_job(session_id, job_id, status="canceled")

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def status(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise InterfaceFrameworkError(f"unknown session: {session_id!r}")
        journal = self.get_journal(session_id)
        if journal is None:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(f"unknown session journal: {session_id!r}")
        return {
            "session": self.snapshot(session),
            "counts": {
                "threads": len(session.threads),
                "workstreams": len(session.workstreams),
                "approvals": len(session.approvals),
                "checkpoints": len(session.checkpoints),
                "jobs": len(session.jobs),
                "messages": len(session.messages),
                "tool_calls": sum(len(entries) for entries in session.tool_calls.values()),
                "queue": len(session.queue),
                "journal_entries": len(journal.entries),
                "journal_branches": len(journal.branches),
                "journal_compactions": len(journal.compactions),
            },
            "thread_ids": list(session.threads),
            "workstream_ids": list(session.workstreams),
            "job_ids": list(session.jobs),
            "checkpoint_ids": list(session.checkpoints),
            "approval_ids": list(session.approvals),
            "journal": journal.describe(),
        }

    def snapshot(self, session: SessionRecord) -> dict[str, Any]:
        return _json_safe(asdict(session))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _journal_path(self, session_id: str) -> Path:
        return self._journal_dir / f"{session_id}.json"

    def _persist_journal(self, journal: SessionJournal) -> None:
        if not self.persist:
            return
        self._journal_path(journal.session_id).write_text(
            json.dumps(journal.snapshot(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_journal(self, path: Path) -> SessionJournal:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InterfaceFrameworkError(f"journal payload is not an object: {path}")
        return SessionJournal.from_dict(payload)

    def _touch_session(self, session_id: str) -> None:
        session = self._require_session(session_id)
        session.updated_at = _utc_now()
        self._persist(session)

    def _record_journal_entry(
        self,
        session_id: str,
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
        journal = self.get_journal(session_id)
        assert journal is not None
        entry = journal.append(
            kind=kind,
            summary=summary,
            branch_id=branch_id,
            thread_id=thread_id,
            workstream_id=workstream_id,
            parent_entry_id=parent_entry_id,
            severity=severity,
            payload=payload,
            metadata=metadata,
            tags=tags,
        )
        self._persist_journal(journal)
        self._touch_session(session_id)
        return entry

    def _session_path(self, session_id: str) -> Path:
        return self.state_dir / f"{session_id}.json"

    def _persist(self, session: SessionRecord) -> None:
        if not self.persist:
            return
        payload = self.snapshot(session)
        self._session_path(session.session_id).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_session(self, path: Path) -> SessionRecord:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InterfaceFrameworkError(f"session payload is not an object: {path}")
        return SessionRecord.from_dict(payload)

    def _require_session(self, session_id: str) -> SessionRecord:
        session = self.get_session(session_id)
        if session is None:
            raise InterfaceFrameworkError(f"unknown session: {session_id!r}")
        return session

    def _resolve_workstream(
        self,
        session: SessionRecord,
        workstream_id_or_name: str | None,
    ) -> Workstream | None:
        if workstream_id_or_name is None:
            if session.active_workstream_id is None:
                return None
            return session.workstreams.get(session.active_workstream_id)
        if workstream_id_or_name in session.workstreams:
            return session.workstreams[workstream_id_or_name]
        for workstream in session.workstreams.values():
            if workstream.name == workstream_id_or_name:
                return workstream
        return None
