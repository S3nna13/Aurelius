"""Runtime facade that binds the interface framework to local agent state."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from src.model.interface_framework import (
    ApprovalRequest,
    AureliusInterfaceFramework,
    BackgroundJob,
    Checkpoint,
    InterfaceFrameworkError,
    MessageEnvelope,
    TaskThread,
    TaskThreadSpec,
    Workstream,
)

from .session_manager import SessionManager
from .skill_catalog import SkillCatalog

__all__ = [
    "AureliusInterfaceRuntime",
]


def _json_safe(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value, sort_keys=True))


def _coerce_workspace(value: str | Path | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise InterfaceFrameworkError("workspace must be a non-empty string or None")
    return value


def _coerce_skill_ids(skill_ids: tuple[str, ...] | list[str] | tuple[Any, ...] | list[Any]) -> tuple[str, ...]:
    if isinstance(skill_ids, str):
        raise InterfaceFrameworkError("skill_ids must be a sequence of non-empty strings")
    try:
        items = tuple(skill_ids)
    except TypeError as exc:  # pragma: no cover - defensive
        raise InterfaceFrameworkError(f"skill_ids must be iterable, got {type(skill_ids).__name__}") from exc
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise InterfaceFrameworkError("skill_ids entries must be non-empty strings")
        normalized.append(item)
    return tuple(normalized)


def _normalize_thread_spec_mapping(spec: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(spec)
    if "workspace" in data:
        data["workspace"] = _coerce_workspace(data["workspace"])
    if "workspace_roots" in data:
        roots = data["workspace_roots"]
        if isinstance(roots, str):
            raise InterfaceFrameworkError("workspace_roots must be a sequence of strings")
        try:
            items = tuple(roots)
        except TypeError as exc:  # pragma: no cover - defensive
            raise InterfaceFrameworkError(
                f"workspace_roots must be iterable, got {type(roots).__name__}"
            ) from exc
        normalized_roots: list[str] = []
        for item in items:
            root = _coerce_workspace(item)
            if root is None:
                raise InterfaceFrameworkError("workspace_roots entries must be non-empty strings")
            normalized_roots.append(root)
        data["workspace_roots"] = tuple(normalized_roots)
    if "attached_skills" in data:
        data["attached_skills"] = _coerce_skill_ids(data["attached_skills"])
    return data


class AureliusInterfaceRuntime:
    """Aurelius-native runtime tying contract, state, and skill bundles together."""

    def __init__(
        self,
        framework: AureliusInterfaceFramework,
        *,
        session_manager: SessionManager | None = None,
        skill_catalog: SkillCatalog | None = None,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
    ) -> None:
        if not isinstance(framework, AureliusInterfaceFramework):
            raise InterfaceFrameworkError(
                f"framework must be AureliusInterfaceFramework, got {type(framework).__name__}"
            )
        self.framework = framework
        self.root_dir = Path(root_dir).expanduser().resolve() if root_dir is not None else framework.paths.repo_root
        self.variant_id = variant_id or framework.model_context.get("variant_id")
        self.session_manager = session_manager or SessionManager(root_dir=self.root_dir)
        self.skill_catalog = skill_catalog or SkillCatalog(
            self.root_dir,
            state_dir=self.root_dir / ".aurelius" / "skills",
        )

    @classmethod
    def from_repo_root(
        cls,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
    ) -> "AureliusInterfaceRuntime":
        framework = AureliusInterfaceFramework.from_repo_root(
            root_dir=root_dir,
            variant_id=variant_id,
        )
        root = Path(root_dir).expanduser().resolve() if root_dir is not None else framework.paths.repo_root
        return cls(framework, root_dir=root, variant_id=variant_id)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def bundle(self):  # pragma: no cover - trivial proxy
        return self.framework.bundle

    @property
    def paths(self):  # pragma: no cover - trivial proxy
        return self.framework.paths

    @property
    def contract(self):  # pragma: no cover - trivial proxy
        return self.framework.contract

    @property
    def schema(self):  # pragma: no cover - trivial proxy
        return self.framework.schema

    @property
    def mode_catalog(self):  # pragma: no cover - trivial proxy
        return self.framework.mode_catalog

    @property
    def host_catalog(self):  # pragma: no cover - trivial proxy
        return self.framework.host_catalog

    @property
    def model_context(self):  # pragma: no cover - trivial proxy
        return self.framework.model_context

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        sessions = self.session_manager.list_sessions()
        return {
            **self.framework.describe(),
            "session_count": len(sessions),
            "session_ids": [session.session_id for session in sessions],
            "skill_catalog": self.skill_catalog.describe(),
        }

    def validate(self) -> None:
        self.framework.validate()

    # ------------------------------------------------------------------
    # session / workstream helpers
    # ------------------------------------------------------------------
    def create_session(
        self,
        *,
        session_id: str | None = None,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        return self.session_manager.create_session(
            session_id=session_id,
            workspace=_coerce_workspace(workspace),
            metadata=metadata,
        )

    def create_workstream(
        self,
        session_id: str,
        name: str,
        *,
        workspace: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Workstream:
        return self.session_manager.create_workstream(
            session_id,
            name,
            workspace=_coerce_workspace(workspace),
            metadata=metadata,
        )

    def get_session(self, session_id: str):
        return self.session_manager.get_session(session_id)

    def get_thread(self, session_id: str, thread_id: str) -> TaskThread | None:
        return self.session_manager.get_thread(session_id, thread_id)

    def get_background_job(self, session_id: str, job_id: str) -> BackgroundJob | None:
        return self.session_manager.get_background_job(session_id, job_id)

    # ------------------------------------------------------------------
    # thread lifecycle
    # ------------------------------------------------------------------
    def create_thread(
        self,
        spec: TaskThreadSpec | Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
        workstream_name: str | None = None,
        workspace: str | Path | None = None,
        attached_skills: tuple[str, ...] | None = None,
        repo_instructions: str | None = None,
        workspace_instructions: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskThread:
        thread_spec = self._coerce_thread_spec(
            spec,
            session_id=session_id,
            workstream_name=workstream_name,
            workspace=_coerce_workspace(workspace),
            attached_skills=attached_skills,
            repo_instructions=repo_instructions,
            workspace_instructions=workspace_instructions,
            metadata=metadata,
        )
        session = self._ensure_session(thread_spec.session_id, workspace=thread_spec.workspace)
        workstream = self._ensure_workstream(session.session_id, thread_spec)
        thread_spec = replace(
            thread_spec,
            session_id=session.session_id,
            workstream_id=workstream.workstream_id,
            workstream_name=workstream.name,
            workspace=thread_spec.workspace or session.workspace,
            repo_instructions=thread_spec.repo_instructions
            if thread_spec.repo_instructions is not None
            else self._load_instruction_layers(thread_spec.workspace or session.workspace, label="repo"),
            workspace_instructions=thread_spec.workspace_instructions
            if thread_spec.workspace_instructions is not None
            else self._load_instruction_layers(thread_spec.workspace or session.workspace, label="workspace"),
        )
        thread = self.framework.create_thread(thread_spec)
        skill_bundles = self._resolve_skill_bundles(thread_spec.attached_skills)
        thread = replace(
            thread,
            skills=skill_bundles,
            instruction_stack=self._compose_instruction_stack(thread, skill_bundles),
        )
        self.session_manager.register_thread(session.session_id, thread, workstream_id=workstream.workstream_id)
        return thread

    def attach_skills(
        self,
        thread: TaskThread,
        skill_ids: tuple[str, ...] | list[str] | tuple[Any, ...] | list[Any],
    ) -> TaskThread:
        skill_bundles = self._resolve_skill_bundles(_coerce_skill_ids(skill_ids))
        updated = self.framework.attach_skills(thread, skill_bundles)
        updated = replace(
            updated,
            instruction_stack=self._compose_instruction_stack(updated, skill_bundles),
        )
        self._update_thread(updated)
        return updated

    def request_approval(self, thread: TaskThread, **kwargs: Any) -> ApprovalRequest:
        approval = self.framework.request_approval(thread, **kwargs)
        if thread.session_id is not None:
            self.session_manager.register_approval(thread.session_id, approval)
        return approval

    def checkpoint_thread(self, thread: TaskThread, **kwargs: Any) -> Checkpoint:
        checkpoint = self.framework.checkpoint_thread(thread, **kwargs)
        if thread.session_id is not None:
            self.session_manager.register_checkpoint(thread.session_id, checkpoint)
        return checkpoint

    def resume_thread(self, checkpoint: Checkpoint) -> TaskThread:
        thread = self.framework.resume_thread(checkpoint)
        if checkpoint.session_id is not None:
            self.session_manager.register_thread(
                checkpoint.session_id,
                thread,
                workstream_id=checkpoint.workstream_id,
            )
        return thread

    def spawn_subagent(self, thread: TaskThread, **kwargs: Any) -> TaskThread:
        child = self.framework.spawn_subagent(thread, **kwargs)
        if thread.session_id is not None:
            self.session_manager.register_thread(thread.session_id, child, workstream_id=child.workstream_id)
        return child

    def launch_background_job(self, thread: TaskThread, **kwargs: Any) -> BackgroundJob:
        job = self.framework.launch_background_job(thread, **kwargs)
        if thread.session_id is not None:
            self.session_manager.register_background_job(thread.session_id, job)
        return job

    def cancel_background_job(self, job: BackgroundJob | str, *, session_id: str | None = None) -> BackgroundJob:
        canceled = self.framework.cancel_background_job(job)
        sid = session_id or canceled.metadata.get("session_id")
        if sid is not None:
            self.session_manager.update_background_job(sid, canceled.job_id, status="canceled", result=canceled.result)
        return canceled

    def route_channel(self, *, host: str, channel: str, thread: TaskThread | None = None, recipient: str | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        routed = self.framework.route_channel(
            host=host,
            channel=channel,
            thread=thread,
            recipient=recipient,
            metadata=metadata,
        )
        if thread is not None and thread.session_id is not None:
            envelope = MessageEnvelope(**routed["envelope"])
            self.session_manager.register_message(thread.session_id, envelope)
        return routed

    def record_tool_call(self, *, tool_name: str, arguments: dict[str, Any], call_id: str | None = None, host_step_id: str | None = None, status: str = "validated", thread: TaskThread | None = None) -> dict[str, Any]:
        normalized = self.framework.record_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id,
            host_step_id=host_step_id,
            status=status,
            thread=thread,
        )
        if thread is not None and thread.session_id is not None:
            self.session_manager.register_tool_call(thread.session_id, thread.thread_id, normalized)
        return normalized

    # ------------------------------------------------------------------
    # inspection helpers
    # ------------------------------------------------------------------
    def thread_status(self, session_id: str, thread_id: str) -> dict[str, Any]:
        session = self.session_manager.get_session(session_id)
        thread = self.session_manager.get_thread(session_id, thread_id)
        if session is None or thread is None:
            raise InterfaceFrameworkError(f"unknown thread: {thread_id!r}")
        return {
            "session": self.session_manager.snapshot(session),
            "thread": _json_safe(asdict(thread)),
            "tool_calls": self.session_manager.list_tool_calls(session_id, thread_id),
        }

    def session_status(self, session_id: str) -> dict[str, Any]:
        return self.session_manager.status(session_id)

    def list_skills(self, *, scope: str | None = None, active: bool | None = None, include_archived: bool = True):
        return self.skill_catalog.list(scope=scope, active=active, include_archived=include_archived)

    def search_skills(self, query: str):
        return self.skill_catalog.search(query)

    def activate_skill(self, skill_id: str):
        return self.skill_catalog.activate(skill_id)

    def deactivate_skill(self, skill_id: str):
        return self.skill_catalog.deactivate(skill_id)

    def workflow_shell(self):
        from .workflow_shell import WorkflowShell

        return WorkflowShell(self)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _default_session_id(self) -> str | None:
        if self.session_manager.list_sessions():
            return self.session_manager.list_sessions()[0].session_id
        return None

    def _ensure_session(self, session_id: str | None, *, workspace: str | None = None):
        if session_id is None:
            return self.create_session(workspace=workspace)
        session = self.session_manager.get_session(session_id)
        if session is not None:
            return session
        return self.create_session(session_id=session_id, workspace=workspace)

    def _ensure_workstream(self, session_id: str, spec: TaskThreadSpec):
        workstream = None
        if spec.workstream_id is not None:
            workstream = self.session_manager.get_workstream(session_id, spec.workstream_id)
        if workstream is None and spec.workstream_name is not None:
            for existing in self.session_manager.list_workstreams(session_id):
                if existing.name == spec.workstream_name:
                    workstream = existing
                    break
        if workstream is None:
            name = spec.workstream_name or "default"
            workstream = self.create_workstream(session_id, name, workspace=spec.workspace)
        return workstream

    def _coerce_thread_spec(
        self,
        spec: TaskThreadSpec | Mapping[str, Any] | None,
        *,
        session_id: str | None,
        workstream_name: str | None,
        workspace: str | Path | None,
        attached_skills: tuple[str, ...] | None,
        repo_instructions: str | None,
        workspace_instructions: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> TaskThreadSpec:
        if spec is None:
            raise InterfaceFrameworkError("spec is required")
        if isinstance(spec, Mapping):
            spec = TaskThreadSpec(**_normalize_thread_spec_mapping(spec))
        if not isinstance(spec, TaskThreadSpec):
            raise InterfaceFrameworkError(f"spec must be TaskThreadSpec, got {type(spec).__name__}")
        extras = dict(metadata or {})
        merged_skills = attached_skills if attached_skills is not None else spec.attached_skills
        return replace(
            spec,
            session_id=session_id or spec.session_id,
            workstream_name=workstream_name or spec.workstream_name,
            workspace=_coerce_workspace(workspace) or spec.workspace or str(self.root_dir),
            attached_skills=_coerce_skill_ids(merged_skills),
            repo_instructions=repo_instructions if repo_instructions is not None else spec.repo_instructions,
            workspace_instructions=workspace_instructions if workspace_instructions is not None else spec.workspace_instructions,
            metadata={**spec.metadata, **extras},
        )

    def _resolve_skill_bundles(self, skill_ids: tuple[str, ...]) -> tuple[Any, ...]:
        if not skill_ids:
            return tuple()
        return self.skill_catalog.resolve_skill_bundles(skill_ids)

    def _compose_instruction_stack(self, thread: TaskThread, skills: tuple[Any, ...]) -> tuple[str, ...]:
        layers = [
            "system policy: canonical interface contract",
            f"user task prompt: {thread.task_prompt}",
        ]
        if thread.repo_instructions:
            layers.append(f"repo instructions: {thread.repo_instructions}")
        if thread.workspace_instructions:
            layers.append(f"workspace instructions: {thread.workspace_instructions}")
        mode = self.framework.select_mode(thread.mode)
        layers.append(f"mode instructions: {mode.name} ({mode.intent}; {mode.tool_policy})")
        for skill in skills:
            layers.append(
                f"skill instructions: {skill.skill_id}"
                + (f": {skill.instructions}" if getattr(skill, "instructions", "") else "")
            )
        if thread.parent_checkpoint_id:
            layers.append(f"thread memory / checkpoints: {thread.parent_checkpoint_id}")
        if thread.memory_summary:
            layers.append(f"thread memory summary: {thread.memory_summary}")
        return tuple(layers)

    def _update_thread(self, thread: TaskThread) -> None:
        if thread.session_id is not None:
            self.session_manager.register_thread(thread.session_id, thread, workstream_id=thread.workstream_id)

    def _load_instruction_layers(self, workspace: str | None, *, label: str) -> str | None:
        if workspace is None:
            return None
        layers = self.skill_catalog.instruction_layers_for(workspace=workspace, repo_root=self.root_dir)
        matching = [layer for layer in layers if layer.startswith(f"{label} instructions:")]
        if not matching:
            return None
        return "\n\n".join(matching)
