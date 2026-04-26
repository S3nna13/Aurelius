"""Aurelius terminal shell surface.

This module keeps the shell thin and explicit: it delegates thread,
approval, checkpoint, subagent, background-job, channel-routing, and
tool-call work to :class:`src.model.interface_framework.AureliusInterfaceFramework`
and only maintains the minimal shell session state needed to render and
persist a terminal-first interaction surface.
"""

from __future__ import annotations

import copy
import json
import shlex
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agent.skill_catalog import SkillCatalog
from src.agent.surface_catalog import describe_ui_surface
from src.model.interface_framework import (
    ApprovalRequest,
    AureliusInterfaceFramework,
    BackgroundJob,
    Checkpoint,
    ModePolicy,
    SkillBundle,
    TaskThread,
    TaskThreadSpec,
)
from src.model.interface_framework import (
    MessageEnvelope as FrameworkMessageEnvelope,
)

__all__ = [
    "AureliusShell",
    "AureliusShellError",
    "MessageEnvelope",
    "SkillRecord",
    "Workstream",
    "WorkflowRun",
    "WorkflowStep",
]

_KNOWN_WORKFLOW_STEP_KINDS = (
    "message",
    "approval",
    "tool_call",
    "checkpoint",
    "background_job",
    "subagent",
)

_DEFAULT_SKILL_ROOT_NAMES = (
    "skills",
    ".codex/skills",
    ".agents/skills",
)


class AureliusShellError(ValueError):
    """Raised when the shell surface encounters malformed state or input."""


@dataclass(frozen=True)
class MessageEnvelope:
    """Channel-aware message envelope recorded by the shell."""

    channel: str
    thread_id: str | None
    sender: str
    kind: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("channel", "sender", "kind", "content", "created_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if self.thread_id is not None and not isinstance(self.thread_id, str):
            raise AureliusShellError("thread_id must be a str or None")
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MessageEnvelope:
        return cls(
            channel=payload["channel"],
            thread_id=payload.get("thread_id"),
            sender=payload["sender"],
            kind=payload["kind"],
            content=payload["content"],
            created_at=payload["created_at"],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SkillRecord:
    """Local-first skill catalog record."""

    skill_id: str
    name: str
    source_path: str
    provenance: str
    summary: str = ""
    scope: str = "workspace"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("skill_id", "name", "source_path", "provenance", "scope"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if not isinstance(self.summary, str):
            raise AureliusShellError("summary must be a string")
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SkillRecord:
        return cls(
            skill_id=payload["skill_id"],
            name=payload["name"],
            source_path=payload["source_path"],
            provenance=payload["provenance"],
            summary=payload.get("summary", ""),
            scope=payload.get("scope", "workspace"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class Workstream:
    """Named shell workstream grouping one or more Aurelius threads."""

    workstream_id: str
    name: str
    workspace: str | None
    thread_ids: tuple[str, ...] = field(default_factory=tuple)
    status: str = "open"
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("workstream_id", "name", "status", "created_at", "updated_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise AureliusShellError("workspace must be a str or None")
        if not isinstance(self.thread_ids, tuple):
            raise AureliusShellError("thread_ids must be a tuple")
        if not all(isinstance(item, str) and item for item in self.thread_ids):
            raise AureliusShellError("thread_ids entries must be non-empty strings")
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Workstream:
        return cls(
            workstream_id=payload["workstream_id"],
            name=payload["name"],
            workspace=payload.get("workspace"),
            thread_ids=tuple(payload.get("thread_ids", ())),
            status=payload.get("status", "open"),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorkflowStep:
    """Normalized workflow step for shell execution."""

    index: int
    kind: str
    payload: dict[str, Any]
    approval_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise AureliusShellError("index must be a non-negative integer")
        if self.kind not in _KNOWN_WORKFLOW_STEP_KINDS:
            raise AureliusShellError(
                f"kind must be one of {_KNOWN_WORKFLOW_STEP_KINDS}, got {self.kind!r}"
            )
        if not isinstance(self.payload, dict):
            raise AureliusShellError("payload must be a dict")
        if not isinstance(self.approval_required, bool):
            raise AureliusShellError("approval_required must be bool")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowStep:
        return cls(
            index=payload["index"],
            kind=payload["kind"],
            payload=dict(payload.get("payload", {})),
            approval_required=bool(payload.get("approval_required", False)),
        )


@dataclass(frozen=True)
class WorkflowRun:
    """Result of executing a workflow inside the shell."""

    run_id: str
    thread_id: str
    workflow_name: str
    status: str
    steps: tuple[WorkflowStep, ...]
    transcript: tuple[dict[str, Any], ...]
    created_at: str
    updated_at: str
    halted_step_index: int | None = None
    halted_reason: str | None = None
    final_artifact: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "thread_id",
            "workflow_name",
            "status",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AureliusShellError(f"{field_name} must be a non-empty string")
        if not isinstance(self.steps, tuple):
            raise AureliusShellError("steps must be a tuple")
        if not all(isinstance(item, WorkflowStep) for item in self.steps):
            raise AureliusShellError("steps entries must be WorkflowStep instances")
        if not isinstance(self.transcript, tuple):
            raise AureliusShellError("transcript must be a tuple")
        if not all(isinstance(item, dict) for item in self.transcript):
            raise AureliusShellError("transcript entries must be dict objects")
        if self.halted_step_index is not None and (
            not isinstance(self.halted_step_index, int) or self.halted_step_index < 0
        ):
            raise AureliusShellError("halted_step_index must be a non-negative int or None")
        if self.halted_reason is not None and not isinstance(self.halted_reason, str):
            raise AureliusShellError("halted_reason must be a str or None")
        if self.final_artifact is not None and not isinstance(self.final_artifact, dict):
            raise AureliusShellError("final_artifact must be a dict or None")
        if self.final_artifact is not None:
            json.dumps(self.final_artifact, sort_keys=True)
        if not isinstance(self.metadata, dict):
            raise AureliusShellError("metadata must be a dict")
        json.dumps(self.metadata, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowRun:
        return cls(
            run_id=payload["run_id"],
            thread_id=payload["thread_id"],
            workflow_name=payload["workflow_name"],
            status=payload["status"],
            steps=tuple(WorkflowStep.from_dict(item) for item in payload.get("steps", ())),
            transcript=tuple(dict(item) for item in payload.get("transcript", ())),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            halted_step_index=payload.get("halted_step_index"),
            halted_reason=payload.get("halted_reason"),
            final_artifact=payload.get("final_artifact"),
            metadata=dict(payload.get("metadata", {})),
        )


class AureliusShell:
    """Terminal-first Aurelius shell built on the interface framework."""

    def __init__(
        self,
        framework: AureliusInterfaceFramework | None = None,
        *,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
        session_id: str | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        self.framework = framework or AureliusInterfaceFramework.from_repo_root(
            root_dir=root_dir,
            variant_id=variant_id,
        )
        self.session_id = session_id or f"session-{uuid.uuid4()}"
        self.workspace = str(workspace or self.framework.paths.repo_root)
        self.current_mode = self._default_mode_name()
        self.active_thread_id: str | None = None
        self.active_workstream_id: str | None = None
        self._threads: dict[str, TaskThread] = {}
        self._workstreams: dict[str, Workstream] = {}
        self._jobs: dict[str, BackgroundJob] = {}
        self._approvals: dict[str, ApprovalRequest] = {}
        self._checkpoints: dict[str, Checkpoint] = {}
        self._messages: list[MessageEnvelope] = []
        self._workflow_runs: dict[str, WorkflowRun] = {}
        self._tool_calls: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def from_repo_root(
        cls,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
        *,
        session_id: str | None = None,
        workspace: str | Path | None = None,
    ) -> AureliusShell:
        return cls(
            root_dir=root_dir,
            variant_id=variant_id,
            session_id=session_id,
            workspace=workspace,
        )

    @property
    def threads(self) -> tuple[TaskThread, ...]:
        return tuple(self._threads.values())

    @property
    def workstreams(self) -> tuple[Workstream, ...]:
        return tuple(self._workstreams.values())

    @property
    def jobs(self) -> tuple[BackgroundJob, ...]:
        return tuple(self._jobs.values())

    @property
    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(self._approvals.values())

    @property
    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(self._checkpoints.values())

    @property
    def messages(self) -> tuple[MessageEnvelope, ...]:
        return tuple(self._messages)

    @property
    def workflow_runs(self) -> tuple[WorkflowRun, ...]:
        return tuple(self._workflow_runs.values())

    def list_messages(
        self,
        *,
        channel: str | None = None,
        thread_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """List routed shell envelopes with optional channel/thread filters."""
        filtered: list[dict[str, Any]] = []
        for message in self._messages:
            if channel is not None and message.channel != channel:
                continue
            if thread_id is not None and message.thread_id != thread_id:
                continue
            filtered.append(asdict(message))
        return tuple(filtered)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the shell session."""
        framework_summary = self.framework.describe()
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "current_mode": self.current_mode,
            "active_thread_id": self.active_thread_id,
            "active_workstream_id": self.active_workstream_id,
            "counts": {
                "threads": len(self._threads),
                "workstreams": len(self._workstreams),
                "jobs": len(self._jobs),
                "approvals": len(self._approvals),
                "checkpoints": len(self._checkpoints),
                "messages": len(self._messages),
                "workflow_runs": len(self._workflow_runs),
                "tool_calls": sum(len(entries) for entries in self._tool_calls.values()),
            },
            "framework": framework_summary,
            "backend_surface": self._describe_backends(),
            "surface_catalog": self.surface_catalog(),
            "shell_capabilities": {
                "thread_status": True,
                "workstream_status": True,
                "job_status": True,
                "workflow_execution": True,
                "skill_discovery": True,
            },
        }

    def list_modes(self) -> tuple[str, ...]:
        return tuple(self.framework.mode_catalog.keys())

    def set_mode(self, mode_name: str) -> ModePolicy:
        policy = self.framework.select_mode(mode_name)
        self.current_mode = policy.name
        return policy

    def create_workstream(
        self,
        name: str,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workstream:
        if not isinstance(name, str) or not name.strip():
            raise AureliusShellError("name must be a non-empty string")
        workstream_id = self._build_workstream_id(name)
        now = _utc_now()
        workstream = Workstream(
            workstream_id=workstream_id,
            name=name,
            workspace=str(workspace) if workspace is not None else self.workspace,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._workstreams[workstream_id] = workstream
        self.active_workstream_id = workstream_id
        return workstream

    def create_thread(
        self,
        *,
        title: str,
        task_prompt: str,
        mode: str | None = None,
        host: str = "cli",
        workspace: str | Path | None = None,
        workstream_id: str | None = None,
        workstream_name: str | None = None,
        parent_checkpoint_id: str | None = None,
        channel: str | None = None,
        skills: Sequence[str | SkillBundle | SkillRecord] | None = None,
        repo_instructions: str | None = None,
        workspace_instructions: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        parent_thread_id: str | None = None,
    ) -> TaskThread:
        normalized_mode = mode or self.current_mode
        if not isinstance(normalized_mode, str) or not normalized_mode.strip():
            raise AureliusShellError("mode must be a non-empty string")
        skill_bundles = self._normalize_skill_inputs(skills or ())
        workspace_value = str(workspace) if workspace is not None else self.workspace
        workspace_roots = (workspace_value,) if workspace_value else ()
        resolved_workstream_name = workstream_name
        if workstream_id is None and resolved_workstream_name is not None:
            created_workstream = self.create_workstream(
                resolved_workstream_name,
                workspace=workspace_value,
            )
            workstream_id = created_workstream.workstream_id
            resolved_workstream_name = created_workstream.name
        if resolved_workstream_name is None and workstream_id is not None:
            existing_workstream = self._workstreams.get(workstream_id)
            if existing_workstream is not None:
                resolved_workstream_name = existing_workstream.name
        spec = TaskThreadSpec(
            title=title,
            mode=normalized_mode,
            task_prompt=task_prompt,
            host=host,
            session_id=self.session_id,
            workstream_id=workstream_id,
            workstream_name=resolved_workstream_name,
            workspace=workspace_value,
            workspace_roots=workspace_roots,
            channel=channel,
            attached_skills=tuple(skill.skill_id for skill in skill_bundles),
            repo_instructions=repo_instructions,
            workspace_instructions=workspace_instructions,
            metadata=dict(metadata or {}),
            thread_id=thread_id,
            parent_thread_id=parent_thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        thread = self.framework.create_thread(spec)
        if skill_bundles:
            thread = self.framework.attach_skills(thread, skill_bundles)
        if workstream_id is None and self.active_workstream_id is not None:
            workstream_id = self.active_workstream_id
        if workstream_id is not None:
            resolved_workstream = self._coerce_workstream(workstream_id)
            thread = replace(
                thread,
                workstream_id=resolved_workstream.workstream_id,
                workstream_name=resolved_workstream.name,
                updated_at=_utc_now(),
            )
        self._threads[thread.thread_id] = thread
        self.active_thread_id = thread.thread_id
        if workstream_id is None and self.active_workstream_id is not None:
            workstream_id = self.active_workstream_id
        if workstream_id is not None:
            self._attach_thread_to_workstream(workstream_id, thread.thread_id)
        return thread

    def attach_skills(
        self,
        thread: str | TaskThread,
        skill_inputs: Sequence[str | SkillBundle | SkillRecord],
    ) -> TaskThread:
        target = self._coerce_thread(thread)
        bundles = self._normalize_skill_inputs(skill_inputs)
        updated = self.framework.attach_skills(target, bundles)
        self._threads[updated.thread_id] = updated
        self.active_thread_id = updated.thread_id
        return updated

    def request_approval(
        self,
        thread: str | TaskThread,
        *,
        category: str,
        action_summary: str,
        affected_resources: Sequence[str] = (),
        reason: str,
        reversible: bool,
        minimum_scope: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ApprovalRequest:
        target = self._coerce_thread(thread)
        approval = self.framework.request_approval(
            target,
            category=category,
            action_summary=action_summary,
            affected_resources=tuple(affected_resources),
            reason=reason,
            reversible=reversible,
            minimum_scope=minimum_scope,
            metadata=metadata,
        )
        self._approvals[approval.approval_id] = approval
        current_thread = self._threads.get(target.thread_id)
        if current_thread is not None:
            updated_approvals = _dedupe_strings(current_thread.approvals + (approval.approval_id,))
            self._threads[target.thread_id] = replace(
                current_thread,
                approvals=updated_approvals,
                updated_at=_utc_now(),
            )
        self.active_thread_id = target.thread_id
        return approval

    def checkpoint_thread(
        self,
        thread: str | TaskThread,
        *,
        memory_summary: str,
        last_model_response: str | None = None,
        last_tool_result: Mapping[str, Any] | None = None,
    ) -> Checkpoint:
        target = self._coerce_thread(thread)
        checkpoint = self.framework.checkpoint_thread(
            target,
            memory_summary=memory_summary,
            last_model_response=last_model_response,
            last_tool_result=dict(last_tool_result or {}) or None,
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        resumed = self.framework.resume_thread(checkpoint)
        self._threads[resumed.thread_id] = resumed
        self.active_thread_id = resumed.thread_id
        return checkpoint

    def resume_thread(
        self,
        checkpoint: str | Checkpoint,
    ) -> TaskThread:
        target = self._coerce_checkpoint(checkpoint)
        resumed = self.framework.resume_thread(target)
        self._threads[resumed.thread_id] = resumed
        self.active_thread_id = resumed.thread_id
        return resumed

    def spawn_subagent(
        self,
        thread: str | TaskThread,
        *,
        title: str,
        task_prompt: str,
        mode: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskThread:
        parent = self._coerce_thread(thread)
        child = self.framework.spawn_subagent(
            parent,
            title=title,
            task_prompt=task_prompt,
            mode=mode,
            metadata=metadata,
        )
        self._threads[child.thread_id] = child
        self.active_thread_id = child.thread_id
        self._inherit_workstream(parent.thread_id, child.thread_id)
        return child

    def launch_background_job(
        self,
        thread: str | TaskThread,
        *,
        description: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> BackgroundJob:
        target = self._coerce_thread(thread)
        job = self.framework.launch_background_job(
            target,
            description=description,
            metadata=metadata,
        )
        self._jobs[job.job_id] = job
        current_thread = self._threads.get(target.thread_id)
        if current_thread is not None:
            updated_ids = _dedupe_strings(current_thread.active_job_ids + (job.job_id,))
            self._threads[target.thread_id] = replace(
                current_thread,
                active_job_ids=updated_ids,
                updated_at=_utc_now(),
            )
        return job

    def cancel_background_job(self, job: str | BackgroundJob) -> BackgroundJob:
        target = self._coerce_job(job)
        canceled = self.framework.cancel_background_job(target)
        self._jobs[canceled.job_id] = canceled
        current_thread = self._threads.get(canceled.thread_id)
        if current_thread is not None:
            updated_ids = tuple(
                job_id for job_id in current_thread.active_job_ids if job_id != canceled.job_id
            )
            self._threads[canceled.thread_id] = replace(
                current_thread,
                active_job_ids=updated_ids,
                updated_at=_utc_now(),
            )
        return canceled

    def route_channel(
        self,
        *,
        channel: str,
        content: str,
        thread: str | TaskThread | None = None,
        sender: str = "aurelius",
        host: str = "cli",
        recipient: str | None = None,
        kind: str = "message",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self._coerce_thread(thread) if thread is not None else None
        routing = self.framework.route_channel(
            host=host,
            channel=channel,
            thread=target,
            recipient=recipient,
            metadata=metadata,
        )
        envelope = MessageEnvelope(
            channel=channel,
            thread_id=target.thread_id if target is not None else None,
            sender=sender,
            kind=kind,
            content=content,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        self._messages.append(envelope)
        if target is not None:
            self.active_thread_id = target.thread_id
            current_thread = self._threads.get(target.thread_id)
            if current_thread is not None:
                message_history = current_thread.message_history + (
                    FrameworkMessageEnvelope(
                        envelope_id=f"envelope-{uuid.uuid4()}",
                        channel_id=channel,
                        thread_id=target.thread_id,
                        sender=sender,
                        recipient=recipient,
                        kind=kind,
                        payload={"content": content, "metadata": dict(metadata or {})},
                        created_at=envelope.created_at,
                        session_id=self.session_id,
                        workstream_id=current_thread.workstream_id,
                        workspace=current_thread.workspace,
                        metadata=dict(metadata or {}),
                    ),
                )
                self._threads[target.thread_id] = replace(
                    current_thread,
                    message_history=message_history,
                    updated_at=_utc_now(),
                )
            self._append_toolless_step(target.thread_id, envelope)
        return {
            "routing": routing,
            "envelope": asdict(envelope),
        }

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        thread: str | TaskThread | None = None,
        call_id: str | None = None,
        host_step_id: str | None = None,
        status: str = "validated",
    ) -> dict[str, Any]:
        target = self._coerce_thread(thread) if thread is not None else None
        record = self.framework.record_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id,
            host_step_id=host_step_id,
            status=status,
            thread=target,
        )
        if target is not None:
            self._tool_calls.setdefault(target.thread_id, []).append(record)
            self.active_thread_id = target.thread_id
            self._append_toolless_step(
                target.thread_id,
                MessageEnvelope(
                    channel=target.channel or "tool_call",
                    thread_id=target.thread_id,
                    sender="tool",
                    kind="tool_call",
                    content=tool_name,
                    created_at=record["recorded_at"],
                    metadata={"arguments": copy.deepcopy(arguments)},
                ),
            )
        return record

    def discover_skills(
        self,
        roots: Sequence[str | Path] | None = None,
    ) -> tuple[SkillRecord, ...]:
        candidate_roots = self._skill_roots(roots)
        discovered: dict[str, SkillRecord] = {}
        for root in candidate_roots:
            if not root.exists():
                continue
            skill_files = self._skill_files_under(root)
            for skill_file in skill_files:
                record = self._build_skill_record(root, skill_file)
                discovered.setdefault(record.skill_id, record)
        return tuple(sorted(discovered.values(), key=lambda record: record.skill_id))

    def attach_skill_ids(
        self,
        thread: str | TaskThread,
        skill_ids: Sequence[str],
        *,
        roots: Sequence[str | Path] | None = None,
    ) -> TaskThread:
        catalog = {record.skill_id: record for record in self.discover_skills(roots)}
        bundles: list[SkillBundle] = []
        for skill_id in skill_ids:
            if skill_id in catalog:
                bundles.append(self._bundle_from_skill_record(catalog[skill_id]))
            else:
                bundles.append(SkillBundle(skill_id=skill_id, name=skill_id))
        return self.attach_skills(thread, bundles)

    def execute_workflow(
        self,
        thread: str | TaskThread,
        workflow: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        approval_resolver: Callable[[ApprovalRequest], Any] | None = None,
    ) -> WorkflowRun:
        current_thread = self._coerce_thread(thread)
        if current_thread.workstream_id is None:
            workstream = (
                self._workstreams.get(self.active_workstream_id)
                if self.active_workstream_id
                else None
            )
            if workstream is None:
                workstream = self.create_workstream(
                    f"{current_thread.title} workstream",
                    workspace=current_thread.workspace,
                )
            current_thread = replace(
                current_thread,
                workstream_id=workstream.workstream_id,
                workstream_name=workstream.name,
                updated_at=_utc_now(),
            )
            self._threads[current_thread.thread_id] = current_thread
            self._attach_thread_to_workstream(workstream.workstream_id, current_thread.thread_id)
        workflow_name, normalized_steps, metadata = self._normalize_workflow(workflow)
        run_id = f"workflow-{uuid.uuid4()}"
        created_at = _utc_now()
        transcript: list[dict[str, Any]] = []
        status = "running"
        halted_step_index: int | None = None
        halted_reason: str | None = None
        final_artifact: dict[str, Any] | None = None

        for step in normalized_steps:
            try:
                step_result = self._execute_workflow_step(
                    current_thread,
                    step,
                    approval_resolver=approval_resolver,
                )
                transcript.append(
                    {
                        "step_index": step.index,
                        "kind": step.kind,
                        "approval_required": step.approval_required,
                        "result": step_result,
                    }
                )
                if step.kind == "subagent" and isinstance(step_result, dict):
                    child_thread_id = step_result.get("thread", {}).get("thread_id")
                    if isinstance(child_thread_id, str) and child_thread_id in self._threads:
                        current_thread = self._threads[child_thread_id]
                elif step.kind == "checkpoint" and isinstance(step_result, dict):
                    checkpoint_id = step_result.get("checkpoint", {}).get("checkpoint_id")
                    if isinstance(checkpoint_id, str) and checkpoint_id in self._checkpoints:
                        current_thread = self._threads[self._checkpoints[checkpoint_id].thread_id]
                current_thread = self._threads.get(current_thread.thread_id, current_thread)
                if step.kind == "approval":
                    decision = step_result.get("decision", "pending")
                    if decision == "pending":
                        status = "pending_approval"
                        halted_step_index = step.index
                        halted_reason = "approval pending"
                        final_artifact = step_result
                        break
                    if decision == "deny":
                        status = "blocked"
                        halted_step_index = step.index
                        halted_reason = "approval denied"
                        final_artifact = step_result
                        break
                final_artifact = step_result
            except Exception as exc:  # noqa: BLE001 - explicit failure surface
                status = "failed"
                halted_step_index = step.index
                halted_reason = str(exc)
                final_artifact = {"error": str(exc)}
                transcript.append(
                    {
                        "step_index": step.index,
                        "kind": step.kind,
                        "approval_required": step.approval_required,
                        "error": str(exc),
                    }
                )
                break
        else:
            if status == "running":
                status = "completed"

        run = WorkflowRun(
            run_id=run_id,
            thread_id=current_thread.thread_id,
            workflow_name=workflow_name,
            status=status,
            steps=normalized_steps,
            transcript=tuple(transcript),
            created_at=created_at,
            updated_at=_utc_now(),
            halted_step_index=halted_step_index,
            halted_reason=halted_reason,
            final_artifact=final_artifact,
            metadata=metadata,
        )
        self._threads[current_thread.thread_id] = replace(
            current_thread,
            status=status,
            updated_at=_utc_now(),
        )
        self.active_thread_id = current_thread.thread_id
        self._workflow_runs[run.run_id] = run
        return run

    def render_status(self, thread_id: str | None = None) -> str:
        if thread_id is not None:
            return self.render_thread_status(thread_id)

        counts = Counter(thread.status for thread in self._threads.values())
        job_counts = Counter(job.status for job in self._jobs.values())
        lines = [
            "Aurelius Shell",
            f"Session: {self.session_id}",
            f"Workspace: {self.workspace}",
            f"Mode: {self.current_mode}",
            f"Active thread: {self.active_thread_id or 'none'}",
            f"Active workstream: {self.active_workstream_id or 'none'}",
            "Threads:",
        ]
        if self._threads:
            for thread in self._threads.values():
                lines.append(f"  - {self._render_thread_row(thread)}")
        else:
            lines.append("  - none")
        lines.append(
            "Thread counts: "
            + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
            if counts
            else "Thread counts: none"
        )
        lines.append("Workstreams:")
        if self._workstreams:
            for workstream in self._workstreams.values():
                lines.append(f"  - {self._render_workstream_row(workstream)}")
        else:
            lines.append("  - none")
        lines.append("Jobs:")
        if self._jobs:
            for job in self._jobs.values():
                lines.append(f"  - {self._render_job_row(job)}")
        else:
            lines.append("  - none")
        if job_counts:
            lines.append(
                "Job counts: "
                + ", ".join(f"{status}={count}" for status, count in sorted(job_counts.items()))
            )
        else:
            lines.append("Job counts: none")
        lines.append(f"Workflow runs: {len(self._workflow_runs)}")
        lines.append(
            f"Tool-call audit entries: {sum(len(entries) for entries in self._tool_calls.values())}"
        )
        lines.append("Backends:")
        backend_names = self._backend_names()
        if backend_names:
            for backend_name in backend_names:
                lines.append(f"  - {backend_name}")
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def render_thread_status(self, thread_id: str) -> str:
        thread = self._coerce_thread(thread_id)
        skill_labels = (
            ", ".join(f"{skill.name} [{skill.skill_id}]" for skill in thread.skills) or "none"
        )
        approval_ids = (
            ", ".join(
                approval.approval_id
                for approval in self._approvals.values()
                if approval.thread_id == thread.thread_id
            )
            or "none"
        )
        job_ids = (
            ", ".join(
                job.job_id for job in self._jobs.values() if job.thread_id == thread.thread_id
            )
            or "none"
        )
        lines = [
            f"Thread: {thread.thread_id}",
            f"Title: {thread.title}",
            f"Mode: {thread.mode}",
            f"Status: {thread.status}",
            f"Host: {thread.host}",
            f"Session: {thread.session_id or 'none'}",
            f"Workstream id: {thread.workstream_id or 'none'}",
            f"Workstream name: {thread.workstream_name or 'none'}",
            f"Workspace: {thread.workspace or 'none'}",
            f"Channel: {thread.channel or 'none'}",
            f"Parent thread: {thread.parent_thread_id or 'none'}",
            f"Parent checkpoint: {thread.parent_checkpoint_id or 'none'}",
            f"Lineage: {', '.join(thread.lineage) if thread.lineage else 'none'}",
            f"Skills: {skill_labels}",
            f"Approvals: {approval_ids}",
            f"Checkpoints: {', '.join(thread.checkpoints) if thread.checkpoints else 'none'}",
            f"Background jobs: {job_ids}",
            f"Tool calls: {len(self._tool_calls.get(thread.thread_id, ()))}",
            f"Message history: {len(thread.message_history)}",
            f"Memory summary: {thread.memory_summary or 'none'}",
            f"Last model response: {thread.last_model_response or 'none'}",
            f"Active jobs: {', '.join(thread.active_job_ids) if thread.active_job_ids else 'none'}",
            "Instruction stack:",
        ]
        if thread.instruction_stack:
            for layer in thread.instruction_stack:
                lines.append(f"  - {layer}")
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def render_workstream_status(self, workstream_id: str) -> str:
        workstream = self._coerce_workstream(workstream_id)
        lines = [
            f"Workstream: {workstream.workstream_id}",
            f"Name: {workstream.name}",
            f"Status: {workstream.status}",
            f"Workspace: {workstream.workspace or 'none'}",
            f"Threads: {len(workstream.thread_ids)}",
        ]
        if workstream.thread_ids:
            for thread_id in workstream.thread_ids:
                thread = self._threads.get(thread_id)
                if thread is None:
                    lines.append(f"  - {thread_id} (missing)")
                else:
                    lines.append(f"  - {self._render_thread_row(thread)}")
        else:
            lines.append("  - none")
        return "\n".join(lines)

    def render_job_status(self, job_id: str) -> str:
        job = self._coerce_job(job_id)
        lines = [
            f"Job: {job.job_id}",
            f"Thread: {job.thread_id}",
            f"Status: {job.status}",
            f"Cancelable: {job.cancelable}",
            f"Description: {job.description}",
            f"Created at: {job.created_at}",
            f"Updated at: {job.updated_at}",
        ]
        if job.result is not None:
            lines.append(f"Result: {job.result!r}")
        if job.metadata:
            lines.append(f"Metadata: {json.dumps(job.metadata, sort_keys=True)}")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe shell state snapshot."""
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "current_mode": self.current_mode,
            "active_thread_id": self.active_thread_id,
            "active_workstream_id": self.active_workstream_id,
            "framework_signature": {
                "title": self.framework.contract["metadata"]["title"],
                "schema_version": self.framework.contract["metadata"]["schema_version"],
                "variant_id": self.framework.model_context.get("variant_id"),
            },
            "threads": {thread_id: asdict(thread) for thread_id, thread in self._threads.items()},
            "workstreams": {
                workstream_id: asdict(workstream)
                for workstream_id, workstream in self._workstreams.items()
            },
            "jobs": {job_id: asdict(job) for job_id, job in self._jobs.items()},
            "approvals": {
                approval_id: asdict(approval) for approval_id, approval in self._approvals.items()
            },
            "checkpoints": {
                checkpoint_id: asdict(checkpoint)
                for checkpoint_id, checkpoint in self._checkpoints.items()
            },
            "messages": [asdict(message) for message in self._messages],
            "workflow_runs": {run_id: asdict(run) for run_id, run in self._workflow_runs.items()},
            "tool_calls": copy.deepcopy(self._tool_calls),
        }

    @classmethod
    def restore_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
        framework: AureliusInterfaceFramework | None = None,
    ) -> AureliusShell:
        if not isinstance(snapshot, Mapping):
            raise AureliusShellError("snapshot must be a mapping")
        shell = cls(
            framework=framework,
            root_dir=root_dir,
            variant_id=variant_id,
            session_id=snapshot.get("session_id"),
            workspace=snapshot.get("workspace"),
        )
        snapshot_mode = snapshot.get("current_mode", shell.current_mode)
        if isinstance(snapshot_mode, str):
            shell.set_mode(snapshot_mode)
        else:
            shell.current_mode = shell._default_mode_name()
        shell.active_thread_id = snapshot.get("active_thread_id")
        shell.active_workstream_id = snapshot.get("active_workstream_id")

        signature = snapshot.get("framework_signature", {})
        expected_signature = {
            "title": shell.framework.contract["metadata"]["title"],
            "schema_version": shell.framework.contract["metadata"]["schema_version"],
            "variant_id": shell.framework.model_context.get("variant_id"),
        }
        if dict(signature) and dict(signature) != expected_signature:
            raise AureliusShellError(
                "snapshot framework signature does not match the requested framework"
            )

        for thread_id, payload in dict(snapshot.get("threads", {})).items():
            thread = _thread_from_dict(payload)
            shell._threads[thread_id] = thread
        for workstream_id, payload in dict(snapshot.get("workstreams", {})).items():
            shell._workstreams[workstream_id] = Workstream.from_dict(payload)
        for job_id, payload in dict(snapshot.get("jobs", {})).items():
            shell._jobs[job_id] = BackgroundJob(**payload)
        for approval_id, payload in dict(snapshot.get("approvals", {})).items():
            shell._approvals[approval_id] = ApprovalRequest(**payload)
        for checkpoint_id, payload in dict(snapshot.get("checkpoints", {})).items():
            shell._checkpoints[checkpoint_id] = Checkpoint(
                checkpoint_id=payload["checkpoint_id"],
                thread_id=payload["thread_id"],
                created_at=payload["created_at"],
                lineage=tuple(payload.get("lineage", ())),
                thread_snapshot=dict(payload["thread_snapshot"]),
                contract_metadata=dict(payload["contract_metadata"]),
                model_context=dict(payload["model_context"]),
                memory_summary=payload["memory_summary"],
                last_model_response=payload.get("last_model_response"),
                last_tool_result=payload.get("last_tool_result"),
                session_id=payload.get("session_id"),
                workstream_id=payload.get("workstream_id"),
                workstream_name=payload.get("workstream_name"),
                pending_approval_ids=tuple(payload.get("pending_approval_ids", ())),
                active_job_ids=tuple(payload.get("active_job_ids", ())),
                tool_observations=tuple(
                    dict(item) for item in payload.get("tool_observations", ())
                ),
            )
        shell._messages = [
            MessageEnvelope.from_dict(payload) for payload in snapshot.get("messages", [])
        ]
        shell._workflow_runs = {
            run_id: WorkflowRun.from_dict(payload)
            for run_id, payload in dict(snapshot.get("workflow_runs", {})).items()
        }
        shell._tool_calls = {
            thread_id: list(entries)
            for thread_id, entries in dict(snapshot.get("tool_calls", {})).items()
        }
        return shell

    def execute_command(self, command_line: str) -> str:
        """Execute a tiny shell command and return the textual result."""
        if not isinstance(command_line, str) or not command_line.strip():
            raise AureliusShellError("command_line must be a non-empty string")
        argv = shlex.split(command_line)
        if not argv:
            raise AureliusShellError("command_line produced no tokens")
        command = argv[0]
        if command == "status":
            return self.render_status()
        if command == "mode" and len(argv) >= 3 and argv[1] == "set":
            policy = self.set_mode(argv[2])
            return json.dumps(
                {
                    "current_mode": self.current_mode,
                    "policy": asdict(policy),
                },
                sort_keys=True,
            )
        if command == "backend" and len(argv) >= 2 and argv[1] == "list":
            return json.dumps(self._describe_backends(), sort_keys=True)
        if command == "backend" and len(argv) >= 3 and argv[1] == "show":
            backend_name = argv[2]
            backends = self._backend_surface()
            try:
                adapter = backends.get_backend(backend_name)
            except Exception as exc:
                raise AureliusShellError(str(exc)) from exc
            return json.dumps(
                {
                    "backend": {
                        "backend_name": adapter.contract.backend_name,
                        "adapter_class": type(adapter).__name__,
                        "contract": asdict(adapter.contract),
                        "runtime_info": adapter.runtime_info(),
                    }
                },
                sort_keys=True,
            )
        if command == "backend" and len(argv) >= 3 and argv[1] == "engine" and argv[2] == "list":
            return json.dumps(
                {"engine_surface": self.surface_catalog()["engine_adapters"]},
                sort_keys=True,
            )
        if command == "backend" and len(argv) >= 4 and argv[1] == "engine" and argv[2] == "show":
            engine_name = argv[3]
            engine_surface = self.surface_catalog()["engine_adapters"]
            for record in engine_surface["engine_adapters"]:
                if record["backend_name"] == engine_name:
                    return json.dumps({"engine": record}, sort_keys=True)
            raise AureliusShellError(
                f"unknown engine adapter: {engine_name!r}; known: {engine_surface['names']}"
            )
        if command == "skill" and len(argv) >= 2 and argv[1] == "summary":
            return json.dumps(
                {"summary": self.catalog_skill_summary()},
                sort_keys=True,
            )
        if command == "skill" and len(argv) >= 3 and argv[1] == "show":
            return json.dumps(
                {"skill": self.catalog_skill_show(argv[2])},
                sort_keys=True,
            )
        if command == "skill" and len(argv) >= 3 and argv[1] == "search":
            query = " ".join(argv[2:]).strip()
            if not query:
                raise AureliusShellError("skill search requires a non-empty query")
            matches = self.catalog_skill_search(query)
            return json.dumps(
                {
                    "count": len(matches),
                    "skills": matches,
                },
                sort_keys=True,
            )
        if command == "channel" and len(argv) >= 2 and argv[1] == "list":
            return json.dumps(
                {
                    "count": len(self.list_messages()),
                    "messages": list(self.list_messages()),
                },
                sort_keys=True,
            )
        if command == "journal" and len(argv) >= 3 and argv[1] == "branch":
            return json.dumps(
                {"journal": self.journal_branch_summary(argv[2])},
                sort_keys=True,
            )
        if command == "journal" and len(argv) >= 3 and argv[1] == "compaction":
            return json.dumps(
                {"journal": self.journal_compaction_summary(branch_id=argv[2])},
                sort_keys=True,
            )
        if command == "capability" and len(argv) >= 2 and argv[1] == "summary":
            return json.dumps(
                {"capability": self.capability_summary()},
                sort_keys=True,
            )
        if command == "capability" and len(argv) >= 2 and argv[1] == "schema":
            return json.dumps(
                {"schema": self.capability_summary_schema()},
                sort_keys=True,
            )
        if command == "surface" and len(argv) >= 2 and argv[1] == "summary":
            return json.dumps(
                {"surface": self.surface_catalog()},
                sort_keys=True,
            )
        if command == "surface" and len(argv) >= 2 and argv[1] == "schema":
            return json.dumps(
                {"schema": self.surface_catalog_schema()},
                sort_keys=True,
            )
        if command == "thread" and len(argv) >= 3 and argv[1] == "status":
            thread_id = argv[2] if len(argv) > 2 else self.active_thread_id
            if thread_id is None:
                raise AureliusShellError("no active thread selected")
            return self.render_thread_status(thread_id)
        raise AureliusShellError(f"unsupported shell command: {command_line!r}")

    def catalog_skill_summary(self) -> dict[str, Any]:
        """Return the local-first skill catalog provenance summary."""
        return self._build_skill_catalog().provenance_summary()

    def catalog_skill_show(self, skill_id: str) -> dict[str, Any]:
        """Return one catalog skill record as a JSON-safe mapping."""
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise AureliusShellError("skill_id must be a non-empty string")
        entry = self._build_skill_catalog().get(skill_id)
        if entry is None:
            raise AureliusShellError(f"unknown skill: {skill_id!r}")
        return asdict(entry)

    def catalog_skill_search(self, query: str) -> list[dict[str, Any]]:
        """Search the local-first skill catalog."""
        if not isinstance(query, str) or not query.strip():
            raise AureliusShellError("query must be a non-empty string")
        return [asdict(entry) for entry in self._build_skill_catalog().search(query)]

    def instruction_layers(
        self,
        *,
        workspace: str | Path | None = None,
        skill_ids: Sequence[str] = (),
        mode_name: str | None = None,
        memory_summary: str | None = None,
    ) -> tuple[str, ...]:
        """Expose the active instruction layering model for terminal consumers."""
        return self._build_skill_catalog().instruction_layers_for(
            workspace=workspace,
            repo_root=self.framework.paths.repo_root,
            skill_ids=skill_ids,
            mode_name=mode_name,
            memory_summary=memory_summary,
        )

    def journal_branch_summary(self, branch_id: str = "main") -> dict[str, Any]:
        """Return a persisted journal branch summary for the shell session."""
        runtime = self._build_runtime()
        if runtime.session_manager.get_session(self.session_id) is None:
            return {
                "branch_id": branch_id,
                "name": branch_id,
                "base_entry_id": None,
                "head_entry_id": None,
                "entry_count": 0,
                "latest_entry_id": None,
                "latest_entry_kind": None,
                "compaction_count": 0,
                "latest_compaction_id": None,
                "metadata": {},
            }
        return runtime.journal_branch_summary(self.session_id, branch_id)

    def journal_compaction_summary(
        self,
        *,
        branch_id: str | None = None,
        compaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a persisted journal compaction summary for the shell session."""
        runtime = self._build_runtime()
        if runtime.session_manager.get_session(self.session_id) is None:
            return {
                "compaction_id": None,
                "branch_id": branch_id,
                "policy": None,
                "keep_last_n": 0,
                "dropped_count": 0,
                "retained_count": 0,
                "summary_entry_id": None,
                "facts_count": 0,
                "summary_text": "",
                "created_at": None,
                "metadata": {},
            }
        return runtime.journal_compaction_summary(
            self.session_id,
            compaction_id=compaction_id,
            branch_id=branch_id,
        )

    def capability_summary(self) -> dict[str, Any]:
        """Return a runtime-backed capability summary for the shell session."""
        return self._build_runtime().capability_summary(self.session_id)

    def capability_summary_schema(self) -> dict[str, Any]:
        """Return the versioned schema for capability summaries."""
        return self._build_runtime().capability_summary_schema()

    def surface_catalog(self) -> dict[str, Any]:
        """Return a runtime-backed surface catalog with UI coverage added."""
        catalog = self._build_runtime().surface_catalog()
        catalog["ui"] = describe_ui_surface()
        return catalog

    def surface_catalog_schema(self) -> dict[str, Any]:
        """Return the versioned schema for surface catalogs."""
        return self._build_runtime().surface_catalog_schema()

    def _default_mode_name(self) -> str:
        if "chat" in self.framework.mode_catalog:
            return "chat"
        return next(iter(self.framework.mode_catalog))

    def _build_skill_catalog(self) -> SkillCatalog:
        return SkillCatalog(self.framework.paths.repo_root)

    def _build_runtime(self):
        from src.agent.interface_runtime import AureliusInterfaceRuntime

        return AureliusInterfaceRuntime.from_repo_root(
            root_dir=self.framework.paths.repo_root,
            variant_id=self.framework.model_context.get("variant_id"),
        )

    def _build_workstream_id(self, name: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip())
        slug = "-".join(part for part in slug.split("-") if part)
        slug = slug[:32] or "workstream"
        return f"workstream-{slug}-{uuid.uuid4().hex[:8]}"

    def _coerce_thread(self, thread: str | TaskThread) -> TaskThread:
        if isinstance(thread, TaskThread):
            return thread
        if isinstance(thread, str) and thread:
            try:
                return self._threads[thread]
            except KeyError as exc:
                raise AureliusShellError(f"unknown thread: {thread!r}") from exc
        raise AureliusShellError("thread must be a TaskThread or non-empty thread id")

    def _coerce_workstream(self, workstream: str | Workstream) -> Workstream:
        if isinstance(workstream, Workstream):
            return workstream
        if isinstance(workstream, str) and workstream:
            try:
                return self._workstreams[workstream]
            except KeyError as exc:
                raise AureliusShellError(f"unknown workstream: {workstream!r}") from exc
        raise AureliusShellError("workstream must be a Workstream or non-empty workstream id")

    def _coerce_job(self, job: str | BackgroundJob) -> BackgroundJob:
        if isinstance(job, BackgroundJob):
            job_id = job.job_id
        elif isinstance(job, str) and job:
            job_id = job
        else:
            raise AureliusShellError("job must be a BackgroundJob or non-empty job id")
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise AureliusShellError(f"unknown job: {job_id!r}") from exc

    def _coerce_checkpoint(self, checkpoint: str | Checkpoint) -> Checkpoint:
        if isinstance(checkpoint, Checkpoint):
            checkpoint_id = checkpoint.checkpoint_id
        elif isinstance(checkpoint, str) and checkpoint:
            checkpoint_id = checkpoint
        else:
            raise AureliusShellError("checkpoint must be a Checkpoint or non-empty checkpoint id")
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise AureliusShellError(f"unknown checkpoint: {checkpoint_id!r}") from exc

    def _attach_thread_to_workstream(self, workstream_id: str, thread_id: str) -> None:
        workstream = self._coerce_workstream(workstream_id)
        if thread_id not in self._threads:
            raise AureliusShellError(f"cannot attach unknown thread {thread_id!r}")
        thread_ids = _dedupe_strings(workstream.thread_ids + (thread_id,))
        updated = replace(
            workstream,
            thread_ids=thread_ids,
            updated_at=_utc_now(),
        )
        self._workstreams[updated.workstream_id] = updated
        self.active_workstream_id = updated.workstream_id

    def _inherit_workstream(self, parent_thread_id: str, child_thread_id: str) -> None:
        for workstream in self._workstreams.values():
            if parent_thread_id in workstream.thread_ids:
                self._attach_thread_to_workstream(workstream.workstream_id, child_thread_id)
                return

    def _append_toolless_step(self, thread_id: str, envelope: MessageEnvelope) -> None:
        thread = self._threads.get(thread_id)
        if thread is None:
            return
        steps = list(thread.steps)
        steps.append(
            {
                "kind": envelope.kind,
                "channel": envelope.channel,
                "content": envelope.content,
                "created_at": envelope.created_at,
                "metadata": copy.deepcopy(envelope.metadata),
            }
        )
        self._threads[thread_id] = replace(thread, steps=tuple(steps), updated_at=_utc_now())

    def _normalize_skill_inputs(
        self,
        skill_inputs: Sequence[str | SkillBundle | SkillRecord],
    ) -> tuple[SkillBundle, ...]:
        normalized: list[SkillBundle] = []
        seen: set[str] = set()
        for item in skill_inputs:
            if isinstance(item, SkillBundle):
                bundle = item
            elif isinstance(item, SkillRecord):
                bundle = self._bundle_from_skill_record(item)
            elif isinstance(item, str) and item.strip():
                bundle = SkillBundle(skill_id=item, name=item)
            else:
                raise AureliusShellError(
                    "skill inputs must be strings, SkillBundle instances, or SkillRecord instances"
                )
            if bundle.skill_id in seen:
                continue
            seen.add(bundle.skill_id)
            normalized.append(bundle)
        return tuple(normalized)

    def _bundle_from_skill_record(self, record: SkillRecord) -> SkillBundle:
        return SkillBundle(
            skill_id=record.skill_id,
            name=record.name,
            scope=_normalize_skill_scope(record.scope),
            provenance=record.provenance,
            source_path=record.source_path,
            metadata=copy.deepcopy(record.metadata),
        )

    def _skill_roots(self, roots: Sequence[str | Path] | None) -> tuple[Path, ...]:
        if roots is not None:
            return tuple(Path(root).expanduser().resolve() for root in roots)
        repo_root = self.framework.paths.repo_root
        candidate_roots = [repo_root / name for name in _DEFAULT_SKILL_ROOT_NAMES]
        candidate_roots.extend(
            [
                Path.home() / ".codex" / "skills",
                Path.home() / ".agents" / "skills",
            ]
        )
        return tuple(dict.fromkeys(path.resolve() for path in candidate_roots))

    def _skill_files_under(self, root: Path) -> tuple[Path, ...]:
        if root.is_file():
            return (root,) if root.name == "SKILL.md" else ()
        if not root.is_dir():
            return ()
        return tuple(sorted(path for path in root.rglob("SKILL.md") if path.is_file()))

    def _build_skill_record(self, root: Path, skill_file: Path) -> SkillRecord:
        try:
            text = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise AureliusShellError(f"failed to read skill bundle at {skill_file}: {exc}") from exc
        if not text.strip():
            raise AureliusShellError(f"skill bundle is empty: {skill_file}")
        skill_id = self._skill_id_for_file(root, skill_file)
        name, summary = _parse_skill_markdown(text, skill_id)
        provenance = (
            "repo-local"
            if skill_file.resolve().is_relative_to(self.framework.paths.repo_root)
            else "global"
        )
        scope = "repo" if provenance == "repo-local" else "global"
        return SkillRecord(
            skill_id=skill_id,
            name=name,
            source_path=str(skill_file),
            provenance=provenance,
            summary=summary,
            scope=scope,
            metadata={
                "root": str(root),
                "bundle_path": str(skill_file),
            },
        )

    def _skill_id_for_file(self, root: Path, skill_file: Path) -> str:
        parent = skill_file.parent
        try:
            rel = parent.resolve().relative_to(root.resolve())
        except ValueError:
            rel = parent.name
        if isinstance(rel, Path):
            parts = rel.parts
        elif isinstance(rel, str):
            parts = (rel,)
        else:
            parts = ()
        if parts:
            return "/".join(parts)
        return parent.name

    def _normalize_workflow(
        self,
        workflow: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> tuple[str, tuple[WorkflowStep, ...], dict[str, Any]]:
        if isinstance(workflow, Mapping):
            workflow_name = workflow.get("name") or workflow.get("title") or "workflow"
            metadata = dict(workflow.get("metadata", {}))
            raw_steps = workflow.get("steps")
        elif isinstance(workflow, Sequence) and not isinstance(workflow, (str, bytes)):
            workflow_name = "workflow"
            metadata = {}
            raw_steps = workflow
        else:
            raise AureliusShellError(
                "workflow must be a mapping with 'steps' or a sequence of step mappings"
            )
        if raw_steps is None:
            raise AureliusShellError("workflow is missing a steps collection")
        if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
            raise AureliusShellError("workflow steps must be a sequence of mappings")
        normalized_steps: list[WorkflowStep] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise AureliusShellError(
                    f"workflow step {index} must be a mapping, got {type(raw_step).__name__}"
                )
            kind_value = raw_step.get("kind", raw_step.get("type"))
            if not isinstance(kind_value, str) or not kind_value.strip():
                raise AureliusShellError(f"workflow step {index} is missing kind/type")
            kind = kind_value.strip().lower()
            if kind not in _KNOWN_WORKFLOW_STEP_KINDS:
                raise AureliusShellError(f"workflow step {index} has unsupported kind {kind!r}")
            payload = {
                key: copy.deepcopy(value)
                for key, value in raw_step.items()
                if key not in {"kind", "type"}
            }
            normalized_steps.append(
                WorkflowStep(
                    index=index,
                    kind=kind,
                    payload=payload,
                    approval_required=kind == "approval",
                )
            )
        return str(workflow_name), tuple(normalized_steps), metadata

    def _execute_workflow_step(
        self,
        thread: TaskThread,
        step: WorkflowStep,
        *,
        approval_resolver: Callable[[ApprovalRequest], Any] | None,
    ) -> dict[str, Any]:
        payload = step.payload
        if step.kind == "message":
            content = payload.get("content", payload.get("text", ""))
            if not isinstance(content, str) or not content.strip():
                raise AureliusShellError("message workflow steps require content")
            channel = payload.get("channel") or thread.channel or "workflow"
            routing = self.route_channel(
                channel=str(channel),
                content=content,
                thread=thread,
                sender=str(payload.get("sender", "workflow")),
                host=thread.host,
                recipient=payload.get("recipient"),
                kind="message",
                metadata=dict(payload.get("metadata", {})),
            )
            return routing
        if step.kind == "approval":
            approval = self.request_approval(
                thread,
                category=str(payload.get("category", "")),
                action_summary=str(payload.get("action_summary", "")),
                affected_resources=tuple(payload.get("affected_resources", ())),
                reason=str(payload.get("reason", "")),
                reversible=bool(payload.get("reversible", True)),
                minimum_scope=str(payload.get("minimum_scope", "")),
                metadata=payload.get("metadata"),
            )
            decision = payload.get("decision")
            if decision is None and approval_resolver is not None:
                decision = approval_resolver(approval)
            normalized = _normalize_decision(decision)
            resolved_approval = replace(
                approval,
                decision=normalized,
                decided_at=_utc_now() if normalized != "pending" else None,
            )
            self._approvals[resolved_approval.approval_id] = resolved_approval
            if normalized == "allow":
                return {
                    "approval": asdict(resolved_approval),
                    "decision": "allow",
                    "status": "approved",
                }
            if normalized == "deny":
                return {
                    "approval": asdict(resolved_approval),
                    "decision": "deny",
                    "status": "denied",
                }
            return {
                "approval": asdict(resolved_approval),
                "decision": "pending",
                "status": "pending",
            }
        if step.kind == "tool_call":
            tool_name = payload.get("tool_name")
            arguments = payload.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise AureliusShellError("tool_call workflow steps require tool_name")
            if not isinstance(arguments, dict):
                raise AureliusShellError("tool_call workflow steps require arguments dict")
            record = self.record_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                thread=thread,
                call_id=payload.get("call_id"),
                host_step_id=payload.get("host_step_id"),
                status=str(payload.get("status", "validated")),
            )
            return {"tool_call": record}
        if step.kind == "checkpoint":
            memory_summary = payload.get("memory_summary")
            if not isinstance(memory_summary, str) or not memory_summary.strip():
                raise AureliusShellError("checkpoint workflow steps require memory_summary")
            checkpoint = self.checkpoint_thread(
                thread,
                memory_summary=memory_summary,
                last_model_response=payload.get("last_model_response"),
                last_tool_result=payload.get("last_tool_result"),
            )
            return {"checkpoint": asdict(checkpoint)}
        if step.kind == "background_job":
            description = payload.get("description")
            if not isinstance(description, str) or not description.strip():
                raise AureliusShellError("background_job workflow steps require description")
            job = self.launch_background_job(
                thread,
                description=description,
                metadata=payload.get("metadata"),
            )
            return {"job": asdict(job)}
        if step.kind == "subagent":
            title = payload.get("title")
            task_prompt = payload.get("task_prompt")
            if not isinstance(title, str) or not title.strip():
                raise AureliusShellError("subagent workflow steps require title")
            if not isinstance(task_prompt, str) or not task_prompt.strip():
                raise AureliusShellError("subagent workflow steps require task_prompt")
            child = self.spawn_subagent(
                thread,
                title=title,
                task_prompt=task_prompt,
                mode=payload.get("mode"),
                metadata=payload.get("metadata"),
            )
            return {"thread": asdict(child)}
        raise AureliusShellError(f"unsupported workflow step kind: {step.kind!r}")

    def _render_thread_row(self, thread: TaskThread) -> str:
        skill_count = len(thread.skills)
        checkpoint_count = len(thread.checkpoints)
        return (
            f"{thread.thread_id} | {thread.mode} | {thread.status} | "
            f"{thread.title} | skills={skill_count} checkpoints={checkpoint_count}"
        )

    def _render_workstream_row(self, workstream: Workstream) -> str:
        return (
            f"{workstream.workstream_id} | {workstream.status} | {workstream.name} "
            f"| threads={len(workstream.thread_ids)}"
        )

    def _render_job_row(self, job: BackgroundJob) -> str:
        return f"{job.job_id} | {job.status} | {job.thread_id} | {job.description}"

    def _backend_surface(self):
        import src.backends as backends

        return backends

    def _backend_names(self) -> tuple[str, ...]:
        return self._describe_backends()["names"]

    def _describe_backends(self) -> dict[str, Any]:
        backends = self._backend_surface()
        names = backends.list_backends()
        records = []
        for backend_name in names:
            adapter = backends.get_backend(backend_name)
            records.append(
                {
                    "backend_name": adapter.contract.backend_name,
                    "adapter_class": type(adapter).__name__,
                    "contract": asdict(adapter.contract),
                    "runtime_info": adapter.runtime_info(),
                }
            )
        return {
            "count": len(records),
            "names": list(names),
            "backends": records,
        }


def _thread_from_dict(payload: Mapping[str, Any]) -> TaskThread:
    try:
        skills = tuple(
            SkillBundle(
                skill_id=item["skill_id"],
                name=item["name"],
                description=item.get("description", ""),
                scope=item.get("scope", "thread"),
                instructions=item.get("instructions", ""),
                scripts=tuple(item.get("scripts", ())),
                resources=tuple(item.get("resources", ())),
                entrypoints=tuple(item.get("entrypoints", ())),
                version=item.get("version"),
                provenance=item.get("provenance"),
                source_path=item.get("source_path"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("skills", ())
        )
        message_history = tuple(
            FrameworkMessageEnvelope(
                envelope_id=item["envelope_id"],
                channel_id=item["channel_id"],
                thread_id=item.get("thread_id"),
                sender=item["sender"],
                recipient=item.get("recipient"),
                kind=item["kind"],
                payload=dict(item.get("payload", {})),
                created_at=item["created_at"],
                session_id=item.get("session_id"),
                workstream_id=item.get("workstream_id"),
                workspace=item.get("workspace"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("message_history", ())
        )
        return TaskThread(
            thread_id=payload["thread_id"],
            title=payload["title"],
            mode=payload["mode"],
            status=payload["status"],
            host=payload["host"],
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
            steps=tuple(dict(item) for item in payload.get("steps", ())),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            parent_thread_id=payload.get("parent_thread_id"),
            parent_checkpoint_id=payload.get("parent_checkpoint_id"),
            lineage=tuple(payload.get("lineage", ())),
            task_prompt=payload.get("task_prompt", ""),
            memory_summary=payload.get("memory_summary", ""),
            last_model_response=payload.get("last_model_response"),
            last_tool_result=payload.get("last_tool_result"),
            active_job_ids=tuple(payload.get("active_job_ids", ())),
            message_history=message_history,
            metadata=dict(payload.get("metadata", {})),
        )
    except KeyError as exc:
        raise AureliusShellError(f"thread snapshot missing required field: {exc.args[0]}") from exc


def _normalize_decision(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, bool):
        return "allow" if value else "deny"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "allow",
            "allow_once",
            "allow_for_thread",
            "allow_for_scope",
        }:
            return "allow"
        if normalized in {"deny", "block", "blocked"}:
            return "deny"
        if normalized in {"pending", "wait"}:
            return "pending"
    return "pending"


def _normalize_skill_scope(scope: str) -> str:
    normalized = str(scope or "thread").strip().lower()
    if normalized in {"workspace", "local"}:
        return "repo"
    if normalized in {"global", "org", "repo", "thread"}:
        return normalized
    return "thread"


def _parse_skill_markdown(text: str, skill_id: str) -> tuple[str, str]:
    title = ""
    summary_lines: list[str] = []
    in_summary = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_summary:
                break
            continue
        if not title and line.startswith("#"):
            title = line.lstrip("#").strip()
            continue
        if title and not in_summary:
            in_summary = True
            summary_lines.append(line)
        elif in_summary:
            summary_lines.append(line)
    if not title:
        title = skill_id.replace("/", " ").replace("-", " ").title()
    summary = " ".join(summary_lines).strip()
    return title, summary


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
