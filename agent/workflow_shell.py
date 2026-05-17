"""JSON-first workflow shell with approval-gated execution."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from src.model.interface_framework import (
    InterfaceFrameworkError,
    MessageEnvelope,
    TaskThread,
)

from .interface_runtime import AureliusInterfaceRuntime

__all__ = [
    "WorkflowStep",
    "WorkflowRun",
    "WorkflowShell",
]


_STEP_KINDS = {
    "message",
    "approval",
    "tool_call",
    "checkpoint",
    "background_job",
    "subagent",
    "macro",
}
_ALLOW_DECISIONS = {"allow", "allow_once", "allow_for_thread", "allow_for_scope"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
    return value


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _coerce_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raise InterfaceFrameworkError(f"{field_name} must be a sequence of strings, not a string")
    try:
        items = tuple(value)
    except TypeError as exc:  # pragma: no cover - defensive
        raise InterfaceFrameworkError(f"{field_name} must be iterable") from exc
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise InterfaceFrameworkError(f"{field_name} entries must be non-empty strings")
        normalized.append(item)
    return tuple(normalized)


def _normalize_decision(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, bool):
        return "allow" if value else "deny"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ALLOW_DECISIONS:
            return normalized
        if normalized in {"deny", "block", "blocked"}:
            return "deny"
        if normalized in {"pending", "wait"}:
            return "pending"
    return "pending"


@dataclass(frozen=True)
class WorkflowStep:
    """Normalized workflow step."""

    step_id: str
    kind: str
    summary: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("step_id", "kind", "summary"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.kind not in _STEP_KINDS:
            raise InterfaceFrameworkError(
                f"kind must be one of {sorted(_STEP_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.payload, dict):
            raise InterfaceFrameworkError("payload must be a dict")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")
        _json_safe(self.payload)
        _json_safe(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class WorkflowRun:
    """Execution transcript for one workflow invocation."""

    run_id: str
    workflow_id: str
    title: str
    session_id: str
    workstream_id: str
    thread_id: str
    status: str
    steps: tuple[WorkflowStep, ...]
    transcript: tuple[dict[str, Any], ...]
    checkpoint_ids: tuple[str, ...] = field(default_factory=tuple)
    job_ids: tuple[str, ...] = field(default_factory=tuple)
    subthread_ids: tuple[str, ...] = field(default_factory=tuple)
    approval_ids: tuple[str, ...] = field(default_factory=tuple)
    halted_step_index: int | None = None
    halted_reason: str | None = None
    final_artifact: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "workflow_id",
            "title",
            "session_id",
            "workstream_id",
            "thread_id",
            "status",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if not isinstance(self.steps, tuple):
            raise InterfaceFrameworkError("steps must be a tuple")
        if not all(isinstance(step, WorkflowStep) for step in self.steps):
            raise InterfaceFrameworkError("steps entries must be WorkflowStep instances")
        if not isinstance(self.transcript, tuple):
            raise InterfaceFrameworkError("transcript must be a tuple")
        if not all(isinstance(item, dict) for item in self.transcript):
            raise InterfaceFrameworkError("transcript entries must be dict objects")
        for item in self.transcript:
            _json_safe(item)
        for tuple_name in ("checkpoint_ids", "job_ids", "subthread_ids", "approval_ids"):
            value = getattr(self, tuple_name)
            if not isinstance(value, tuple):
                raise InterfaceFrameworkError(f"{tuple_name} must be a tuple")
            if not all(isinstance(item, str) and item for item in value):
                raise InterfaceFrameworkError(f"{tuple_name} entries must be non-empty strings")
        if self.halted_step_index is not None and (
            not isinstance(self.halted_step_index, int) or self.halted_step_index < 0
        ):
            raise InterfaceFrameworkError("halted_step_index must be a non-negative int or None")
        if self.halted_reason is not None and not isinstance(self.halted_reason, str):
            raise InterfaceFrameworkError("halted_reason must be str or None")
        if self.final_artifact is not None and not isinstance(self.final_artifact, dict):
            raise InterfaceFrameworkError("final_artifact must be dict or None")
        if self.final_artifact is not None:
            _json_safe(self.final_artifact)
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")
        _json_safe(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


class WorkflowShell:
    """Execute JSON-safe workflow definitions against the runtime facade."""

    def __init__(self, runtime: AureliusInterfaceRuntime) -> None:
        if not isinstance(runtime, AureliusInterfaceRuntime):
            raise InterfaceFrameworkError(
                f"runtime must be AureliusInterfaceRuntime, got {type(runtime).__name__}"
            )
        self.runtime = runtime

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime.describe(),
        }

    def normalize_workflow(self, workflow: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(workflow, Mapping):
            raise InterfaceFrameworkError(
                f"workflow must be a mapping, got {type(workflow).__name__}"
            )
        workflow_id = str(
            workflow.get("workflow_id") or workflow.get("id") or f"workflow-{uuid.uuid4()}"
        )
        title = str(workflow.get("title") or workflow_id)
        if "macros" in workflow and not isinstance(workflow.get("macros"), Mapping):
            raise InterfaceFrameworkError("workflow.macros must be a mapping")
        macros = workflow.get("macros", {})
        steps = self._normalize_steps(workflow.get("steps", ()), macros or {})
        thread_spec = workflow.get("thread_spec")
        if thread_spec is None:
            thread_spec = workflow.get("thread")
        if thread_spec is not None and not isinstance(thread_spec, Mapping):
            raise InterfaceFrameworkError("workflow.thread_spec must be a mapping")
        if thread_spec is None:
            thread_spec = {}
        if "metadata" in workflow and not isinstance(workflow.get("metadata"), Mapping):
            raise InterfaceFrameworkError("workflow.metadata must be a mapping")
        metadata = workflow.get("metadata", {})
        return {
            "workflow_id": workflow_id,
            "title": title,
            "mode": str(workflow.get("mode") or thread_spec.get("mode") or "code"),
            "thread_spec": dict(thread_spec),
            "steps": steps,
            "macros": _json_safe(dict(macros)),
            "metadata": dict(metadata),
        }

    def run_workflow(
        self,
        workflow: Mapping[str, Any],
        *,
        session_id: str | None = None,
        workstream_name: str | None = None,
        thread: TaskThread | None = None,
        approval_decisions: Mapping[str, str] | None = None,
    ) -> WorkflowRun:
        normalized = self.normalize_workflow(workflow)
        current_thread = thread or self._ensure_thread(
            normalized,
            session_id=session_id,
            workstream_name=workstream_name,
        )
        transcript: list[dict[str, Any]] = []
        checkpoint_ids: list[str] = []
        job_ids: list[str] = []
        subthread_ids: list[str] = []
        approval_ids: list[str] = []
        status = "completed"
        halted_step_index: int | None = None
        halted_reason: str | None = None
        final_artifact: dict[str, Any] | None = None

        for index, step in enumerate(normalized["steps"], start=1):
            result = self._execute_step(
                current_thread,
                step,
                transcript,
                approval_decisions=approval_decisions,
            )
            transcript.append(
                {
                    "step": step.to_dict(),
                    "result": result,
                }
            )
            if "approval_id" in result:
                approval_ids.append(result["approval_id"])
            if "checkpoint_id" in result:
                checkpoint_ids.append(result["checkpoint_id"])
            if "job_id" in result:
                job_ids.append(result["job_id"])
            if "subthread_id" in result:
                subthread_ids.append(result["subthread_id"])
            current_thread = self._refresh_thread(current_thread)
            if step.kind == "approval" and result["status"] in {"pending", "denied"}:
                status = "pending_approval" if result["status"] == "pending" else "blocked"
                halted_reason = (
                    "approval pending" if result["status"] == "pending" else "approval denied"
                )
                halted_step_index = index
                final_artifact = result
                break
            if result["status"] in {"blocked", "failed"}:
                status = result["status"]
                halted_reason = result.get("reason") or f"workflow {status}"
                halted_step_index = index
                final_artifact = result
                break
            final_artifact = result

        final_thread = replace(current_thread, status=status, updated_at=_utc_now())
        if final_thread.session_id is not None:
            self.runtime.session_manager.register_thread(
                final_thread.session_id,
                final_thread,
                workstream_id=final_thread.workstream_id,
            )

        return WorkflowRun(
            run_id=f"run-{uuid.uuid4()}",
            workflow_id=normalized["workflow_id"],
            title=normalized["title"],
            session_id=current_thread.session_id or session_id or "",
            workstream_id=current_thread.workstream_id or "",
            thread_id=current_thread.thread_id,
            status=status,
            steps=normalized["steps"],
            transcript=tuple(transcript),
            checkpoint_ids=tuple(checkpoint_ids),
            job_ids=tuple(job_ids),
            subthread_ids=tuple(subthread_ids),
            approval_ids=tuple(approval_ids),
            halted_step_index=halted_step_index,
            halted_reason=halted_reason,
            final_artifact=final_artifact,
            metadata=dict(normalized["metadata"]),
        )

    def _ensure_thread(
        self,
        workflow: Mapping[str, Any],
        *,
        session_id: str | None,
        workstream_name: str | None,
    ) -> TaskThread:
        thread_spec = dict(workflow.get("thread_spec", {}))
        if session_id is not None:
            thread_spec["session_id"] = session_id
        if workstream_name is not None:
            thread_spec["workstream_name"] = workstream_name
        if "title" not in thread_spec:
            thread_spec["title"] = workflow["title"]
        if "task_prompt" not in thread_spec:
            thread_spec["task_prompt"] = workflow["title"]
        if "mode" not in thread_spec:
            thread_spec["mode"] = workflow["mode"]
        if "host" not in thread_spec:
            thread_spec["host"] = "cli"
        return self.runtime.create_thread(thread_spec)

    def _refresh_thread(self, thread: TaskThread) -> TaskThread:
        if thread.session_id is None:
            return thread
        refreshed = self.runtime.get_thread(thread.session_id, thread.thread_id)
        return refreshed or thread

    def _normalize_steps(
        self,
        raw_steps: Any,
        macros: Mapping[str, Any],
    ) -> tuple[WorkflowStep, ...]:
        if isinstance(raw_steps, str) or not isinstance(raw_steps, Iterable):
            raise InterfaceFrameworkError("workflow steps must be a sequence")
        steps: list[WorkflowStep] = []
        next_step_index = 1

        _macro_stack: set[str] = set()

        def _consume(collection: Any) -> None:
            nonlocal next_step_index
            if isinstance(collection, str) or not isinstance(collection, Iterable):
                raise InterfaceFrameworkError("workflow steps must be a sequence")
            for index, raw_step in enumerate(collection, start=1):
                if not isinstance(raw_step, Mapping):
                    raise InterfaceFrameworkError(
                        f"workflow step {index} must be a mapping, got {type(raw_step).__name__}"
                    )
                kind = str(raw_step.get("kind") or raw_step.get("type") or "")
                if kind == "macro":
                    macro_name = str(raw_step.get("name") or raw_step.get("macro") or "")
                    expanded = macros.get(macro_name)
                    if expanded is None:
                        raise InterfaceFrameworkError(f"unknown workflow macro: {macro_name!r}")
                    if macro_name in _macro_stack:
                        raise InterfaceFrameworkError(
                            f"circular macro reference detected: {macro_name!r}"
                        )
                    _macro_stack.add(macro_name)
                    _consume(expanded)
                    _macro_stack.discard(macro_name)
                    continue
                summary = str(
                    raw_step.get("summary")
                    or raw_step.get("name")
                    or kind
                    or f"step-{next_step_index}"
                )
                payload: dict[str, Any] = {}
                raw_payload = raw_step.get("payload")
                if raw_payload is not None:
                    if not isinstance(raw_payload, Mapping):
                        raise InterfaceFrameworkError(
                            f"workflow step {index} payload must be a mapping"
                        )
                    payload.update(dict(raw_payload))
                for key, value in raw_step.items():
                    if key in {
                        "kind",
                        "type",
                        "summary",
                        "name",
                        "step_id",
                        "id",
                        "metadata",
                        "payload",
                        "macro",
                    }:
                        continue
                    payload.setdefault(key, value)
                metadata = raw_step.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    raise InterfaceFrameworkError(
                        f"workflow step {index} metadata must be a mapping"
                    )
                step_id = str(
                    raw_step.get("step_id") or raw_step.get("id") or f"step-{next_step_index}"
                )
                next_step_index += 1
                steps.append(
                    WorkflowStep(
                        step_id=step_id,
                        kind=kind,
                        summary=summary,
                        payload=payload,
                        metadata=dict(metadata),
                    )
                )

        _consume(raw_steps)
        return tuple(steps)

    def _execute_step(
        self,
        thread: TaskThread,
        step: WorkflowStep,
        transcript: list[dict[str, Any]],
        *,
        approval_decisions: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = dict(step.payload)
        base = {
            "step_id": step.step_id,
            "kind": step.kind,
            "summary": step.summary,
            "timestamp": _utc_now(),
            "status": "ok",
            "payload": _json_safe(payload),
        }

        if step.kind == "message":
            message_payload = payload.get("message")
            if isinstance(message_payload, Mapping):
                normalized_message_payload: dict[str, Any] = dict(message_payload)
            elif message_payload is None:
                normalized_message_payload = dict(payload)
            else:
                normalized_message_payload = {"content": message_payload}
            envelope = MessageEnvelope(
                envelope_id=f"envelope-{uuid.uuid4()}",
                channel_id=str(payload.get("channel") or thread.channel or "workflow"),
                thread_id=thread.thread_id,
                sender=str(payload.get("sender") or thread.host),
                recipient=payload.get("recipient"),
                kind="message",
                payload=normalized_message_payload,
                created_at=_utc_now(),
                session_id=thread.session_id,
                workstream_id=thread.workstream_id,
                workspace=thread.workspace,
                metadata=dict(step.metadata),
            )
            if thread.session_id is not None:
                self.runtime.session_manager.register_message(thread.session_id, envelope)
            base["message_envelope"] = asdict(envelope)
            return base

        if step.kind == "approval":
            affected_resources = _coerce_string_tuple(
                payload.get("affected_resources"), "affected_resources"
            )
            approval = self.runtime.request_approval(
                thread,
                category=str(payload.get("category") or "shell command"),
                action_summary=str(payload.get("action_summary") or step.summary),
                affected_resources=affected_resources,
                reason=str(payload.get("reason") or "workflow approval step"),
                reversible=bool(payload.get("reversible", False)),
                minimum_scope=str(payload.get("minimum_scope") or "allow_once"),
                metadata=dict(step.metadata),
            )
            decision = payload.get("decision")
            if decision is None and approval_decisions is not None:
                decision = approval_decisions.get(step.step_id) or approval_decisions.get(
                    step.summary
                )
            normalized = _normalize_decision(decision or approval.decision)
            approval = replace(
                approval,
                decision=normalized,
                decided_at=_utc_now() if normalized != "pending" else None,
            )
            if thread.session_id is not None:
                self.runtime.session_manager.register_approval(thread.session_id, approval)
            base["approval_id"] = approval.approval_id
            base["decision"] = approval.decision
            if approval.decision == "pending":
                base["status"] = "pending"
            elif approval.decision not in _ALLOW_DECISIONS:
                base["status"] = "denied"
            return base

        if step.kind == "tool_call":
            tool_name = str(payload.get("tool_name") or payload.get("name") or "")
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, Mapping):
                raise InterfaceFrameworkError("tool_call arguments must be a mapping")
            normalized = self.runtime.record_tool_call(
                tool_name=tool_name,
                arguments=dict(arguments),
                call_id=payload.get("call_id"),
                host_step_id=payload.get("host_step_id"),
                status=str(payload.get("status") or "validated"),
                thread=thread,
            )
            base["tool_call"] = normalized
            return base

        if step.kind == "checkpoint":
            checkpoint = self.runtime.checkpoint_thread(
                thread,
                memory_summary=str(
                    payload.get("memory_summary") or thread.memory_summary or step.summary
                ),
                last_model_response=payload.get("last_model_response"),
                last_tool_result=payload.get("last_tool_result"),
            )
            base["checkpoint_id"] = checkpoint.checkpoint_id
            return base

        if step.kind == "background_job":
            job = self.runtime.launch_background_job(
                thread,
                description=str(payload.get("description") or step.summary),
                metadata=dict(step.metadata),
            )
            if thread.session_id is not None:
                if payload.get("cancel"):
                    job = self.runtime.cancel_background_job(job, session_id=thread.session_id)
                elif "result" in payload or str(payload.get("status") or "").strip():
                    job = self.runtime.session_manager.update_background_job(
                        thread.session_id,
                        job.job_id,
                        status=str(payload.get("status") or "completed"),
                        result=payload.get("result"),
                        error=payload.get("error"),
                    )
            base["job_id"] = job.job_id
            base["job_status"] = job.status
            base["job_result"] = job.result
            return base

        if step.kind == "subagent":
            child = self.runtime.spawn_subagent(
                thread,
                title=str(payload.get("title") or f"{thread.title} subtask"),
                task_prompt=str(payload.get("task_prompt") or step.summary),
                mode=payload.get("mode"),
                metadata=dict(step.metadata),
            )
            if payload.get("attached_skills"):
                child = self.runtime.attach_skills(
                    child,
                    _coerce_string_tuple(payload["attached_skills"], "attached_skills"),
                )
            base["subthread_id"] = child.thread_id
            base["subthread"] = _json_safe(
                {
                    "thread_id": child.thread_id,
                    "title": child.title,
                    "mode": child.mode,
                    "status": child.status,
                }
            )
            return base

        raise InterfaceFrameworkError(f"unsupported workflow step kind: {step.kind!r}")
