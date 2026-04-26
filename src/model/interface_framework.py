"""Small Aurelius-native facade over the canonical interface contract.

This module does not introduce a second platform. It wraps the canonical
contract bundle in explicit dataclasses and a thin framework class that can
normalize threads, modes, skills, approvals, checkpoints, background jobs,
tool-call audit records, and channel routing in Aurelius-native terms.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agent.background_executor import BackgroundTask, TaskStatus
from src.agent.mcp_client import MCP_PROTOCOL_VERSION

from .family import get_variant_by_id
from .interface_contract import (
    InterfaceContractBundle,
    load_interface_contract_bundle,
    validate_interface_contract,
)
from .manifest import dump_manifest

__all__ = [
    "TaskThreadSpec",
    "TaskThread",
    "ModePolicy",
    "SkillBundle",
    "ApprovalRequest",
    "Checkpoint",
    "BackgroundJob",
    "Workstream",
    "MessageEnvelope",
    "InterfaceFrameworkError",
    "AureliusInterfaceFramework",
]


_KNOWN_THREAD_STATUSES = frozenset(
    {
        "draft",
        "pending_approval",
        "running",
        "blocked",
        "completed",
        "failed",
        "canceled",
    }
)
_KNOWN_APPROVAL_DECISIONS = frozenset(
    {
        "pending",
        "allow",
        "deny",
        "allow_once",
        "allow_for_thread",
        "allow_for_scope",
    }
)
_KNOWN_JOB_STATUSES = frozenset(
    {
        "pending",
        "running",
        "completed",
        "failed",
        "canceled",
    }
)
_KNOWN_WORKSTREAM_STATUSES = frozenset(
    {
        "draft",
        "active",
        "blocked",
        "completed",
        "failed",
        "canceled",
    }
)
_KNOWN_MESSAGE_KINDS = frozenset(
    {
        "message",
        "tool_call",
        "approval",
        "checkpoint",
        "background_job",
        "subagent",
        "status",
        "routing",
    }
)
_KNOWN_SKILL_SCOPES = frozenset({"global", "org", "repo", "thread"})
_CANONICAL_HOSTS = ("cli", "ide", "web", "plugin", "gateway")
_TOOL_CALL_REQUIRED_FIELDS = ("tool_name", "arguments", "call_id", "host_step_id", "status")


class InterfaceFrameworkError(Exception):
    """Raised when the Aurelius interface framework encounters bad input."""


@dataclass(frozen=True)
class ModePolicy:
    """Resolved policy preset for a canonical contract mode."""

    name: str
    intent: str
    tool_policy: str
    default_constraints: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SkillBundle:
    """Normalized skill metadata used by the framework."""

    skill_id: str
    name: str
    description: str = ""
    scope: str = "thread"
    instructions: str = ""
    scripts: tuple[str, ...] = field(default_factory=tuple)
    resources: tuple[str, ...] = field(default_factory=tuple)
    entrypoints: tuple[str, ...] = field(default_factory=tuple)
    version: str | None = None
    provenance: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise InterfaceFrameworkError("skill_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise InterfaceFrameworkError("name must be a non-empty string")
        if not isinstance(self.description, str):
            raise InterfaceFrameworkError("description must be a string")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise InterfaceFrameworkError("scope must be a non-empty string")
        if self.scope not in _KNOWN_SKILL_SCOPES:
            raise InterfaceFrameworkError(
                f"scope must be one of {sorted(_KNOWN_SKILL_SCOPES)}, got {self.scope!r}"
            )
        if not isinstance(self.instructions, str):
            raise InterfaceFrameworkError("instructions must be a string")
        for field_name in ("scripts", "resources", "entrypoints"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise InterfaceFrameworkError(f"{field_name} must be a tuple")
            if not all(isinstance(item, str) and item for item in value):
                raise InterfaceFrameworkError(f"{field_name} entries must be non-empty strings")
        if self.version is not None and not isinstance(self.version, str):
            raise InterfaceFrameworkError("version must be str or None")
        if self.provenance is not None and not isinstance(self.provenance, str):
            raise InterfaceFrameworkError("provenance must be str or None")
        if self.source_path is not None and not isinstance(self.source_path, str):
            raise InterfaceFrameworkError("source_path must be str or None")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass(frozen=True)
class ApprovalRequest:
    """Structured approval record created by the framework."""

    approval_id: str
    thread_id: str
    category: str
    action_summary: str
    affected_resources: tuple[str, ...]
    reason: str
    reversible: bool
    minimum_scope: str
    decision: str = "pending"
    created_at: str = ""
    decided_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.approval_id, str) or not self.approval_id:
            raise InterfaceFrameworkError("approval_id must be a non-empty string")
        if not isinstance(self.thread_id, str) or not self.thread_id:
            raise InterfaceFrameworkError("thread_id must be a non-empty string")
        for field_name in ("category", "action_summary", "reason", "minimum_scope"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
        if self.decision not in _KNOWN_APPROVAL_DECISIONS:
            raise InterfaceFrameworkError(
                f"decision must be one of {sorted(_KNOWN_APPROVAL_DECISIONS)}, got {self.decision!r}"  # noqa: E501
            )
        if not isinstance(self.affected_resources, tuple):
            raise InterfaceFrameworkError("affected_resources must be a tuple")
        if not all(isinstance(item, str) and item for item in self.affected_resources):
            raise InterfaceFrameworkError("affected_resources entries must be non-empty strings")
        if not isinstance(self.reversible, bool):
            raise InterfaceFrameworkError("reversible must be bool")
        if self.created_at and not isinstance(self.created_at, str):
            raise InterfaceFrameworkError("created_at must be an ISO string")
        if self.decided_at is not None and not isinstance(self.decided_at, str):
            raise InterfaceFrameworkError("decided_at must be an ISO string or None")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass(frozen=True)
class BackgroundJob:
    """Tracked background work wrapper with explicit lifecycle state."""

    job_id: str
    thread_id: str
    description: str
    status: str = "pending"
    created_at: str = ""
    updated_at: str = ""
    result: Any = None
    cancelable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise InterfaceFrameworkError("job_id must be a non-empty string")
        if not isinstance(self.thread_id, str) or not self.thread_id:
            raise InterfaceFrameworkError("thread_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise InterfaceFrameworkError("description must be a non-empty string")
        if self.status not in _KNOWN_JOB_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_KNOWN_JOB_STATUSES)}, got {self.status!r}"
            )
        if not isinstance(self.cancelable, bool):
            raise InterfaceFrameworkError("cancelable must be bool")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass(frozen=True)
class MessageEnvelope:
    """Channel-scoped message or event envelope used by the gateway seam."""

    envelope_id: str
    channel_id: str
    thread_id: str | None
    sender: str
    recipient: str | None
    kind: str
    payload: dict[str, Any]
    created_at: str
    session_id: str | None = None
    workstream_id: str | None = None
    workspace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("envelope_id", "channel_id", "sender", "kind", "created_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
        if self.kind not in _KNOWN_MESSAGE_KINDS:
            raise InterfaceFrameworkError(
                f"kind must be one of {sorted(_KNOWN_MESSAGE_KINDS)}, got {self.kind!r}"
            )
        if self.thread_id is not None and not isinstance(self.thread_id, str):
            raise InterfaceFrameworkError("thread_id must be str or None")
        if self.recipient is not None and not isinstance(self.recipient, str):
            raise InterfaceFrameworkError("recipient must be str or None")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise InterfaceFrameworkError("session_id must be str or None")
        if self.workstream_id is not None and not isinstance(self.workstream_id, str):
            raise InterfaceFrameworkError("workstream_id must be str or None")
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise InterfaceFrameworkError("workspace must be str or None")
        if not isinstance(self.payload, dict):
            raise InterfaceFrameworkError("payload must be a dict")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass(frozen=True)
class Workstream:
    """Named parallel workstream within a persistent session."""

    workstream_id: str
    session_id: str
    name: str
    status: str = "draft"
    current_thread_id: str | None = None
    thread_ids: tuple[str, ...] = field(default_factory=tuple)
    queued_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    messages: tuple[MessageEnvelope, ...] = field(default_factory=tuple)
    checkpoint_ids: tuple[str, ...] = field(default_factory=tuple)
    workspace: str | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("workstream_id", "session_id", "name", "created_at", "updated_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
        if self.status not in _KNOWN_WORKSTREAM_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_KNOWN_WORKSTREAM_STATUSES)}, got {self.status!r}"
            )
        if self.current_thread_id is not None and not isinstance(self.current_thread_id, str):
            raise InterfaceFrameworkError("current_thread_id must be str or None")
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise InterfaceFrameworkError("workspace must be str or None")
        for tuple_name in ("thread_ids", "checkpoint_ids"):
            value = getattr(self, tuple_name)
            if not isinstance(value, tuple):
                raise InterfaceFrameworkError(f"{tuple_name} must be a tuple")
            if not all(isinstance(item, str) and item for item in value):
                raise InterfaceFrameworkError(f"{tuple_name} entries must be non-empty strings")
        if not isinstance(self.queued_items, tuple):
            raise InterfaceFrameworkError("queued_items must be a tuple")
        if not all(isinstance(item, dict) for item in self.queued_items):
            raise InterfaceFrameworkError("queued_items entries must be dict objects")
        if not isinstance(self.messages, tuple):
            raise InterfaceFrameworkError("messages must be a tuple")
        if not all(isinstance(item, MessageEnvelope) for item in self.messages):
            raise InterfaceFrameworkError("messages entries must be MessageEnvelope instances")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass(frozen=True)
class TaskThreadSpec:
    """Normalized input shape for creating a task thread."""

    title: str
    mode: str
    task_prompt: str
    host: str = "cli"
    session_id: str | None = None
    workstream_id: str | None = None
    workstream_name: str | None = None
    workspace: str | None = None
    workspace_roots: tuple[str, ...] = field(default_factory=tuple)
    channel: str | None = None
    attached_skills: tuple[str, ...] = field(default_factory=tuple)
    repo_instructions: str | None = None
    workspace_instructions: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    thread_id: str | None = None
    parent_thread_id: str | None = None
    parent_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("title", "mode", "task_prompt", "host"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise InterfaceFrameworkError("workspace must be str or None")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise InterfaceFrameworkError("session_id must be str or None")
        if self.workstream_id is not None and not isinstance(self.workstream_id, str):
            raise InterfaceFrameworkError("workstream_id must be str or None")
        if self.workstream_name is not None and not isinstance(self.workstream_name, str):
            raise InterfaceFrameworkError("workstream_name must be str or None")
        if self.channel is not None and not isinstance(self.channel, str):
            raise InterfaceFrameworkError("channel must be str or None")
        if self.thread_id is not None and not isinstance(self.thread_id, str):
            raise InterfaceFrameworkError("thread_id must be str or None")
        if self.parent_thread_id is not None and not isinstance(self.parent_thread_id, str):
            raise InterfaceFrameworkError("parent_thread_id must be str or None")
        if self.parent_checkpoint_id is not None and not isinstance(self.parent_checkpoint_id, str):
            raise InterfaceFrameworkError("parent_checkpoint_id must be str or None")
        if not isinstance(self.workspace_roots, tuple):
            raise InterfaceFrameworkError("workspace_roots must be a tuple")
        if not all(isinstance(item, str) and item for item in self.workspace_roots):
            raise InterfaceFrameworkError("workspace_roots entries must be non-empty strings")
        if not isinstance(self.attached_skills, tuple):
            raise InterfaceFrameworkError("attached_skills must be a tuple")
        if not all(isinstance(item, str) and item for item in self.attached_skills):
            raise InterfaceFrameworkError("attached_skills entries must be non-empty strings")
        if self.repo_instructions is not None and not isinstance(self.repo_instructions, str):
            raise InterfaceFrameworkError("repo_instructions must be str or None")
        if self.workspace_instructions is not None and not isinstance(
            self.workspace_instructions, str
        ):
            raise InterfaceFrameworkError("workspace_instructions must be str or None")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass(frozen=True)
class TaskThread:
    """Immutable task-thread snapshot used by the framework."""

    thread_id: str
    title: str
    mode: str
    status: str
    host: str
    session_id: str | None
    workstream_id: str | None
    workstream_name: str | None
    workspace: str | None
    workspace_roots: tuple[str, ...]
    channel: str | None
    instruction_stack: tuple[str, ...]
    repo_instructions: str | None = None
    workspace_instructions: str | None = None
    skills: tuple[SkillBundle, ...] = field(default_factory=tuple)
    approvals: tuple[str, ...] = field(default_factory=tuple)
    checkpoints: tuple[str, ...] = field(default_factory=tuple)
    steps: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    created_at: str = ""
    updated_at: str = ""
    parent_thread_id: str | None = None
    parent_checkpoint_id: str | None = None
    lineage: tuple[str, ...] = field(default_factory=tuple)
    task_prompt: str = ""
    memory_summary: str = ""
    last_model_response: str | None = None
    last_tool_result: dict[str, Any] | None = None
    active_job_ids: tuple[str, ...] = field(default_factory=tuple)
    message_history: tuple[MessageEnvelope, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "thread_id",
            "title",
            "mode",
            "status",
            "host",
            "created_at",
            "updated_at",
            "task_prompt",
            "memory_summary",
        ):
            value = getattr(self, field_name)
            if field_name == "memory_summary":
                if not isinstance(value, str):
                    raise InterfaceFrameworkError("memory_summary must be a string")
                continue
            if not isinstance(value, str) or not value.strip():
                raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
        if self.status not in _KNOWN_THREAD_STATUSES:
            raise InterfaceFrameworkError(
                f"status must be one of {sorted(_KNOWN_THREAD_STATUSES)}, got {self.status!r}"
            )
        if self.host not in _CANONICAL_HOSTS:
            raise InterfaceFrameworkError(
                f"host must be one of {_CANONICAL_HOSTS}, got {self.host!r}"
            )
        if self.workspace is not None and not isinstance(self.workspace, str):
            raise InterfaceFrameworkError("workspace must be str or None")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise InterfaceFrameworkError("session_id must be str or None")
        if self.workstream_id is not None and not isinstance(self.workstream_id, str):
            raise InterfaceFrameworkError("workstream_id must be str or None")
        if self.workstream_name is not None and not isinstance(self.workstream_name, str):
            raise InterfaceFrameworkError("workstream_name must be str or None")
        if self.channel is not None and not isinstance(self.channel, str):
            raise InterfaceFrameworkError("channel must be str or None")
        if self.repo_instructions is not None and not isinstance(self.repo_instructions, str):
            raise InterfaceFrameworkError("repo_instructions must be str or None")
        if self.workspace_instructions is not None and not isinstance(
            self.workspace_instructions, str
        ):
            raise InterfaceFrameworkError("workspace_instructions must be str or None")
        if self.parent_thread_id is not None and not isinstance(self.parent_thread_id, str):
            raise InterfaceFrameworkError("parent_thread_id must be str or None")
        if self.parent_checkpoint_id is not None and not isinstance(self.parent_checkpoint_id, str):
            raise InterfaceFrameworkError("parent_checkpoint_id must be str or None")
        if not isinstance(self.workspace_roots, tuple):
            raise InterfaceFrameworkError("workspace_roots must be a tuple")
        if not all(isinstance(item, str) and item for item in self.workspace_roots):
            raise InterfaceFrameworkError("workspace_roots entries must be non-empty strings")
        if not isinstance(self.instruction_stack, tuple):
            raise InterfaceFrameworkError("instruction_stack must be a tuple")
        if not all(isinstance(item, str) and item for item in self.instruction_stack):
            raise InterfaceFrameworkError("instruction_stack entries must be non-empty strings")
        if not isinstance(self.skills, tuple):
            raise InterfaceFrameworkError("skills must be a tuple")
        if not all(isinstance(item, SkillBundle) for item in self.skills):
            raise InterfaceFrameworkError("skills entries must be SkillBundle instances")
        for tuple_name in ("approvals", "checkpoints", "lineage"):
            value = getattr(self, tuple_name)
            if not isinstance(value, tuple):
                raise InterfaceFrameworkError(f"{tuple_name} must be a tuple")
            if not all(isinstance(item, str) and item for item in value):
                raise InterfaceFrameworkError(f"{tuple_name} entries must be non-empty strings")
        if not isinstance(self.steps, tuple):
            raise InterfaceFrameworkError("steps must be a tuple")
        if not all(isinstance(item, dict) for item in self.steps):
            raise InterfaceFrameworkError("steps entries must be dict objects")
        if not isinstance(self.active_job_ids, tuple):
            raise InterfaceFrameworkError("active_job_ids must be a tuple")
        if not all(isinstance(item, str) and item for item in self.active_job_ids):
            raise InterfaceFrameworkError("active_job_ids entries must be non-empty strings")
        if not isinstance(self.message_history, tuple):
            raise InterfaceFrameworkError("message_history must be a tuple")
        if not all(isinstance(item, MessageEnvelope) for item in self.message_history):
            raise InterfaceFrameworkError(
                "message_history entries must be MessageEnvelope instances"
            )
        if self.last_model_response is not None and not isinstance(self.last_model_response, str):
            raise InterfaceFrameworkError("last_model_response must be str or None")
        if self.last_tool_result is not None and not isinstance(self.last_tool_result, dict):
            raise InterfaceFrameworkError("last_tool_result must be dict or None")
        if not isinstance(self.metadata, dict):
            raise InterfaceFrameworkError("metadata must be a dict")


@dataclass(frozen=True)
class Checkpoint:
    """JSON-safe checkpoint snapshot for a thread."""

    checkpoint_id: str
    thread_id: str
    created_at: str
    lineage: tuple[str, ...]
    thread_snapshot: dict[str, Any]
    contract_metadata: dict[str, Any]
    model_context: dict[str, Any]
    memory_summary: str
    last_model_response: str | None = None
    last_tool_result: dict[str, Any] | None = None
    session_id: str | None = None
    workstream_id: str | None = None
    workstream_name: str | None = None
    pending_approval_ids: tuple[str, ...] = field(default_factory=tuple)
    active_job_ids: tuple[str, ...] = field(default_factory=tuple)
    tool_observations: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("checkpoint_id", "thread_id", "created_at", "memory_summary"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise InterfaceFrameworkError(f"{field_name} must be a non-empty string")
        if not isinstance(self.lineage, tuple):
            raise InterfaceFrameworkError("lineage must be a tuple")
        if not all(isinstance(item, str) and item for item in self.lineage):
            raise InterfaceFrameworkError("lineage entries must be non-empty strings")
        if not isinstance(self.thread_snapshot, dict):
            raise InterfaceFrameworkError("thread_snapshot must be a dict")
        if not isinstance(self.contract_metadata, dict):
            raise InterfaceFrameworkError("contract_metadata must be a dict")
        if not isinstance(self.model_context, dict):
            raise InterfaceFrameworkError("model_context must be a dict")
        if self.last_model_response is not None and not isinstance(self.last_model_response, str):
            raise InterfaceFrameworkError("last_model_response must be str or None")
        if self.last_tool_result is not None and not isinstance(self.last_tool_result, dict):
            raise InterfaceFrameworkError("last_tool_result must be dict or None")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise InterfaceFrameworkError("session_id must be str or None")
        if self.workstream_id is not None and not isinstance(self.workstream_id, str):
            raise InterfaceFrameworkError("workstream_id must be str or None")
        if self.workstream_name is not None and not isinstance(self.workstream_name, str):
            raise InterfaceFrameworkError("workstream_name must be str or None")
        if not isinstance(self.pending_approval_ids, tuple):
            raise InterfaceFrameworkError("pending_approval_ids must be a tuple")
        if not all(isinstance(item, str) and item for item in self.pending_approval_ids):
            raise InterfaceFrameworkError("pending_approval_ids entries must be non-empty strings")
        if not isinstance(self.active_job_ids, tuple):
            raise InterfaceFrameworkError("active_job_ids must be a tuple")
        if not all(isinstance(item, str) and item for item in self.active_job_ids):
            raise InterfaceFrameworkError("active_job_ids entries must be non-empty strings")
        if not isinstance(self.tool_observations, tuple):
            raise InterfaceFrameworkError("tool_observations must be a tuple")
        if not all(isinstance(item, dict) for item in self.tool_observations):
            raise InterfaceFrameworkError("tool_observations entries must be dict objects")


class AureliusInterfaceFramework:
    """Aurelius-native facade over the canonical interface contract bundle."""

    def __init__(
        self,
        bundle: InterfaceContractBundle,
        *,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
    ) -> None:
        if not isinstance(bundle, InterfaceContractBundle):
            raise InterfaceFrameworkError(
                f"bundle must be InterfaceContractBundle, got {type(bundle).__name__}"
            )
        self.bundle = bundle
        self.paths = bundle.paths
        self.contract = bundle.contract_json
        self.schema = bundle.schema_json
        self._root_dir = _resolve_root_dir(root_dir, fallback=self.paths.repo_root)
        self._variant_id = variant_id
        self.mode_catalog = self._build_mode_catalog(self.contract)
        self.host_catalog = self._build_host_catalog(self.contract)
        self.model_context = self._build_model_context(variant_id)
        self._approvals: dict[str, list[ApprovalRequest]] = {}
        self._background_jobs: dict[str, BackgroundJob] = {}
        self._tool_observations: dict[str, list[dict[str, Any]]] = {}
        self.validate()

    @classmethod
    def from_repo_root(
        cls,
        root_dir: str | Path | None = None,
        variant_id: str | None = None,
    ) -> AureliusInterfaceFramework:
        bundle = load_interface_contract_bundle(repo_root=root_dir)
        return cls(bundle, root_dir=root_dir, variant_id=variant_id)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the loaded framework envelope."""
        return {
            "title": self.contract["metadata"]["title"],
            "schema_version": self.contract["metadata"]["schema_version"],
            "root_dir": str(self._root_dir),
            "variant_id": self.model_context.get("variant_id"),
            "mode_names": list(self.mode_catalog.keys()),
            "host_names": list(self.host_catalog.keys()),
            "canonical_hosts": list(_CANONICAL_HOSTS),
            "source_product_patterns": sorted(self.contract["source_product_patterns"].keys()),
            "capability_counts": {
                "modes": len(self.mode_catalog),
                "hosts": len(self.host_catalog),
                "source_patterns": len(self.contract["source_product_patterns"]),
                "skill_contract_sections": len(self.contract["skill_contract"]),
                "tool_observations": sum(
                    len(entries) for entries in self._tool_observations.values()
                ),
            },
            "model_context": copy.deepcopy(self.model_context),
            "tooling_seams": {
                "mcp_protocol_version": MCP_PROTOCOL_VERSION,
                "session_manager": "SessionManager",
                "tool_dispatcher": "ToolRegistryDispatcher",
                "tool_validator": "FunctionCallValidator",
                "repo_context": "RepoContextPacker",
                "background_task_shape": BackgroundTask.__name__,
                "background_status_enum": TaskStatus.__name__,
                "workflow_shell": "WorkflowShell",
                "skill_catalog": "SkillCatalog",
                "runtime": "AureliusInterfaceRuntime",
            },
        }

    def select_mode(self, mode_name: str) -> ModePolicy:
        """Resolve one known mode from the loaded contract."""
        if not isinstance(mode_name, str) or not mode_name.strip():
            raise InterfaceFrameworkError("mode_name must be a non-empty string")
        try:
            return self.mode_catalog[mode_name]
        except KeyError as exc:
            raise InterfaceFrameworkError(f"unknown mode: {mode_name!r}") from exc

    def create_thread(self, spec: TaskThreadSpec) -> TaskThread:
        """Normalize a task spec into a deterministic task thread."""
        if not isinstance(spec, TaskThreadSpec):
            raise InterfaceFrameworkError(f"spec must be TaskThreadSpec, got {type(spec).__name__}")
        mode = self.select_mode(spec.mode)
        if spec.host not in self.host_catalog:
            raise InterfaceFrameworkError(
                f"unknown host: {spec.host!r}; expected one of {sorted(self.host_catalog)}"
            )
        created_at = _utc_now()
        thread_id = spec.thread_id or self._deterministic_thread_id(spec)
        instruction_stack = self._build_instruction_stack(spec, mode)
        skill_bundles = self._normalize_skill_inputs(spec.attached_skills)
        lineage = tuple(filter(None, (spec.parent_thread_id,)))
        return TaskThread(
            thread_id=thread_id,
            title=spec.title,
            mode=mode.name,
            status="draft",
            host=spec.host,
            session_id=spec.session_id,
            workstream_id=spec.workstream_id,
            workstream_name=spec.workstream_name,
            workspace=spec.workspace,
            workspace_roots=_dedupe_strings(spec.workspace_roots),
            channel=spec.channel,
            repo_instructions=spec.repo_instructions,
            workspace_instructions=spec.workspace_instructions,
            instruction_stack=instruction_stack,
            skills=skill_bundles,
            created_at=created_at,
            updated_at=created_at,
            parent_thread_id=spec.parent_thread_id,
            parent_checkpoint_id=spec.parent_checkpoint_id,
            lineage=lineage,
            task_prompt=spec.task_prompt,
            memory_summary=spec.metadata.get("memory_summary", "") if spec.metadata else "",
            last_model_response=spec.metadata.get("last_model_response")
            if isinstance(spec.metadata, Mapping)
            else None,
            last_tool_result=copy.deepcopy(spec.metadata.get("last_tool_result"))
            if isinstance(spec.metadata, Mapping)
            and isinstance(spec.metadata.get("last_tool_result"), dict)
            else None,
            active_job_ids=_string_tuple(
                spec.metadata.get("active_job_ids", ()), "metadata.active_job_ids"
            )
            if isinstance(spec.metadata, Mapping) and "active_job_ids" in spec.metadata
            else tuple(),
            message_history=tuple(),
            metadata=copy.deepcopy(spec.metadata),
        )

    def attach_skills(
        self,
        thread: TaskThread,
        skill_inputs: tuple[str | SkillBundle, ...]
        | list[str | SkillBundle]
        | tuple[str, ...]
        | list[str],
    ) -> TaskThread:
        """Normalize and deduplicate skill ids without a second registry."""
        _require_thread(thread)
        normalized = self._normalize_skill_inputs(skill_inputs)
        existing = {skill.skill_id: skill for skill in thread.skills}
        ordered = list(thread.skills)
        for skill in normalized:
            if skill.skill_id in existing:
                continue
            existing[skill.skill_id] = skill
            ordered.append(skill)
        instruction_stack = list(thread.instruction_stack)
        for skill in ordered[len(thread.skills) :]:
            instruction_stack.append(
                f"skill instructions: {skill.skill_id}"
                + (f" ({skill.version})" if skill.version else "")
            )
        return replace(
            thread,
            skills=tuple(ordered),
            instruction_stack=tuple(instruction_stack),
            updated_at=_utc_now(),
        )

    def request_approval(
        self,
        thread: TaskThread,
        *,
        category: str,
        action_summary: str,
        affected_resources: tuple[str, ...] | list[str] = (),
        reason: str,
        reversible: bool,
        minimum_scope: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ApprovalRequest:
        """Create a structured approval record without prompting the user."""
        _require_thread(thread)
        approval = ApprovalRequest(
            approval_id=f"approval-{uuid.uuid4()}",
            thread_id=thread.thread_id,
            category=category,
            action_summary=action_summary,
            affected_resources=_dedupe_strings(tuple(affected_resources)),
            reason=reason,
            reversible=reversible,
            minimum_scope=minimum_scope,
            decision="pending",
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        self._approvals.setdefault(thread.thread_id, []).append(approval)
        return approval

    def checkpoint_thread(
        self,
        thread: TaskThread,
        *,
        memory_summary: str,
        last_model_response: str | None = None,
        last_tool_result: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Serialize a thread into a JSON-safe checkpoint snapshot."""
        _require_thread(thread)
        if not isinstance(memory_summary, str) or not memory_summary.strip():
            raise InterfaceFrameworkError("memory_summary must be a non-empty string")
        checkpoint_id = f"checkpoint-{uuid.uuid4()}"
        created_at = _utc_now()
        thread_with_checkpoint = replace(
            thread,
            checkpoints=thread.checkpoints + (checkpoint_id,),
            updated_at=created_at,
        )
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            thread_id=thread.thread_id,
            created_at=created_at,
            lineage=thread.lineage + (thread.thread_id,),
            thread_snapshot=_json_safe_dataclass(thread_with_checkpoint),
            contract_metadata={
                "title": self.contract["metadata"]["title"],
                "schema_version": self.contract["metadata"]["schema_version"],
                "doc_path": self.contract["metadata"]["doc_path"],
            },
            model_context=copy.deepcopy(self.model_context),
            memory_summary=memory_summary,
            last_model_response=last_model_response,
            last_tool_result=copy.deepcopy(last_tool_result),
            session_id=thread.session_id,
            workstream_id=thread.workstream_id,
            workstream_name=thread.workstream_name,
            pending_approval_ids=thread.approvals,
            active_job_ids=thread.active_job_ids,
            tool_observations=tuple(
                copy.deepcopy(item) for item in self._tool_observations.get(thread.thread_id, ())
            ),
        )

    def resume_thread(self, checkpoint: Checkpoint) -> TaskThread:
        """Rebuild a thread from a checkpoint and validate compatibility."""
        if not isinstance(checkpoint, Checkpoint):
            raise InterfaceFrameworkError(
                f"checkpoint must be Checkpoint, got {type(checkpoint).__name__}"
            )
        metadata = checkpoint.contract_metadata
        if metadata.get("title") != self.contract["metadata"]["title"]:
            raise InterfaceFrameworkError("checkpoint contract title does not match framework")
        if metadata.get("schema_version") != self.contract["metadata"]["schema_version"]:
            raise InterfaceFrameworkError("checkpoint schema_version does not match framework")
        checkpoint_variant = checkpoint.model_context.get("variant_id")
        if checkpoint_variant != self.model_context.get("variant_id"):
            raise InterfaceFrameworkError("checkpoint variant_id does not match framework")

        snapshot = checkpoint.thread_snapshot
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
                for item in snapshot.get("skills", ())
            )
            messages = tuple(
                MessageEnvelope(**item) for item in snapshot.get("message_history", ())
            )
            thread = TaskThread(
                thread_id=snapshot["thread_id"],
                title=snapshot["title"],
                mode=snapshot["mode"],
                status=snapshot["status"],
                host=snapshot["host"],
                session_id=snapshot.get("session_id"),
                workstream_id=snapshot.get("workstream_id"),
                workstream_name=snapshot.get("workstream_name"),
                workspace=snapshot.get("workspace"),
                workspace_roots=tuple(snapshot.get("workspace_roots", ())),
                channel=snapshot.get("channel"),
                repo_instructions=snapshot.get("repo_instructions"),
                workspace_instructions=snapshot.get("workspace_instructions"),
                instruction_stack=tuple(snapshot.get("instruction_stack", ())),
                skills=skills,
                approvals=tuple(snapshot.get("approvals", ())),
                checkpoints=tuple(snapshot.get("checkpoints", ())),
                steps=tuple(snapshot.get("steps", ())),
                created_at=snapshot["created_at"],
                updated_at=snapshot["updated_at"],
                parent_thread_id=snapshot.get("parent_thread_id"),
                parent_checkpoint_id=snapshot.get("parent_checkpoint_id"),
                lineage=tuple(snapshot.get("lineage", ())),
                task_prompt=snapshot["task_prompt"],
                memory_summary=snapshot.get("memory_summary", ""),
                last_model_response=snapshot.get("last_model_response"),
                last_tool_result=copy.deepcopy(snapshot.get("last_tool_result")),
                active_job_ids=tuple(snapshot.get("active_job_ids", ())),
                message_history=messages,
                metadata=dict(snapshot.get("metadata", {})),
            )
        except KeyError as exc:
            raise InterfaceFrameworkError(
                f"checkpoint snapshot missing required thread field: {exc.args[0]}"
            ) from exc
        self.select_mode(thread.mode)
        return thread

    def spawn_subagent(
        self,
        thread: TaskThread,
        *,
        title: str,
        task_prompt: str,
        mode: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskThread:
        """Create a bounded child thread with inherited scope and lineage."""
        _require_thread(thread)
        child_spec = TaskThreadSpec(
            title=title,
            mode=mode or thread.mode,
            task_prompt=task_prompt,
            host=thread.host,
            session_id=thread.session_id,
            workstream_id=thread.workstream_id,
            workstream_name=thread.workstream_name,
            workspace=thread.workspace,
            workspace_roots=thread.workspace_roots,
            channel=thread.channel,
            repo_instructions=thread.repo_instructions,
            workspace_instructions=thread.workspace_instructions,
            attached_skills=tuple(skill.skill_id for skill in thread.skills),
            metadata=dict(metadata or {}),
            parent_thread_id=thread.thread_id,
        )
        child = self.create_thread(child_spec)
        parent_checkpoint_id = (
            thread.checkpoints[-1] if thread.checkpoints else thread.parent_checkpoint_id
        )
        return replace(
            child,
            lineage=thread.lineage + (thread.thread_id,),
            parent_checkpoint_id=parent_checkpoint_id,
        )

    def launch_background_job(
        self,
        thread: TaskThread,
        *,
        description: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> BackgroundJob:
        """Create a tracked background job wrapper."""
        _require_thread(thread)
        created_at = _utc_now()
        job = BackgroundJob(
            job_id=f"job-{uuid.uuid4()}",
            thread_id=thread.thread_id,
            description=description,
            status="pending",
            created_at=created_at,
            updated_at=created_at,
            metadata={
                **dict(metadata or {}),
                "session_id": thread.session_id,
                "workstream_id": thread.workstream_id,
                "workstream_name": thread.workstream_name,
            },
        )
        self._background_jobs[job.job_id] = job
        return job

    def cancel_background_job(self, job: BackgroundJob | str) -> BackgroundJob:
        """Mark a background job canceled and preserve final state."""
        target = self._resolve_job(job)
        if target.status in {"completed", "failed", "canceled"}:
            return target
        updated = replace(target, status="canceled", updated_at=_utc_now(), cancelable=False)
        self._background_jobs[updated.job_id] = updated
        return updated

    def get_background_job(self, job: BackgroundJob | str) -> BackgroundJob:
        """Return the tracked background job or raise loudly if unknown."""
        return self._resolve_job(job)

    def list_background_jobs(
        self, thread: TaskThread | str | None = None
    ) -> tuple[BackgroundJob, ...]:
        """Return tracked background jobs, optionally filtered by thread."""
        if thread is None:
            return tuple(self._background_jobs.values())
        thread_id = thread.thread_id if isinstance(thread, TaskThread) else thread
        if not isinstance(thread_id, str) or not thread_id:
            raise InterfaceFrameworkError("thread must be TaskThread, thread id, or None")
        return tuple(job for job in self._background_jobs.values() if job.thread_id == thread_id)

    def list_tool_observations(
        self, thread: TaskThread | str | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Return the normalized tool-call audit trail."""
        if thread is None:
            observations: list[dict[str, Any]] = []
            for entries in self._tool_observations.values():
                observations.extend(copy.deepcopy(entries))
            return tuple(observations)
        thread_id = thread.thread_id if isinstance(thread, TaskThread) else thread
        if not isinstance(thread_id, str) or not thread_id:
            raise InterfaceFrameworkError("thread must be TaskThread, thread id, or None")
        return tuple(copy.deepcopy(self._tool_observations.get(thread_id, ())))

    def route_channel(
        self,
        *,
        host: str,
        channel: str,
        thread: TaskThread | None = None,
        recipient: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Represent gateway/channel routing without creating a messaging platform."""
        if host not in self.host_catalog:
            raise InterfaceFrameworkError(f"unknown host: {host!r}")
        if not isinstance(channel, str) or not channel.strip():
            raise InterfaceFrameworkError("channel must be a non-empty string")
        if recipient is not None and not isinstance(recipient, str):
            raise InterfaceFrameworkError("recipient must be str or None")
        if thread is not None:
            _require_thread(thread)
        envelope = MessageEnvelope(
            envelope_id=f"envelope-{uuid.uuid4()}",
            channel_id=channel,
            thread_id=thread.thread_id if thread is not None else None,
            sender=host,
            recipient=recipient,
            kind="routing",
            payload={
                "host": host,
                "channel": channel,
                "recipient": recipient,
                "thread_id": thread.thread_id if thread is not None else None,
            },
            created_at=_utc_now(),
            session_id=thread.session_id if thread is not None else None,
            workstream_id=thread.workstream_id if thread is not None else None,
            workspace=thread.workspace if thread is not None else None,
            metadata=dict(metadata or {}),
        )
        return {
            "host": host,
            "channel": channel,
            "recipient": recipient,
            "thread_id": thread.thread_id if thread is not None else None,
            "workspace": thread.workspace if thread is not None else None,
            "workspace_roots": list(thread.workspace_roots) if thread is not None else [],
            "variant_id": self.model_context.get("variant_id"),
            "metadata": dict(metadata or {}),
            "envelope": _json_safe_dataclass(envelope),
        }

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
        host_step_id: str | None = None,
        status: str = "validated",
        thread: TaskThread | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize one tool-call observation for audit/replay."""
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise InterfaceFrameworkError("tool_name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise InterfaceFrameworkError("arguments must be a dict")
        if call_id is not None and not isinstance(call_id, str):
            raise InterfaceFrameworkError("call_id must be str or None")
        if host_step_id is not None and not isinstance(host_step_id, str):
            raise InterfaceFrameworkError("host_step_id must be str or None")
        if not isinstance(status, str) or not status.strip():
            raise InterfaceFrameworkError("status must be a non-empty string")
        if thread is not None:
            _require_thread(thread)

        normalized = {
            "tool_name": tool_name,
            "arguments": copy.deepcopy(arguments),
            "call_id": call_id or f"tool-{uuid.uuid4()}",
            "host_step_id": host_step_id or f"step-{uuid.uuid4()}",
            "status": status,
            "recorded_at": _utc_now(),
            "thread_id": thread.thread_id if thread is not None else None,
            "session_id": thread.session_id if thread is not None else None,
            "workstream_id": thread.workstream_id if thread is not None else None,
            "workspace": thread.workspace if thread is not None else None,
            "channel": thread.channel if thread is not None else None,
        }
        required_fields = tuple(self.contract["canonical_nouns"]["tool_call"]["normalized_fields"])
        missing = [field for field in _TOOL_CALL_REQUIRED_FIELDS if field not in required_fields]
        if missing:
            raise InterfaceFrameworkError(
                f"contract tool_call.normalized_fields missing runtime-required fields: {missing}"
            )
        if thread is not None:
            self._tool_observations.setdefault(thread.thread_id, []).append(normalized)
        return normalized

    def validate(self) -> None:
        """Run structural validation of the loaded contract and framework invariants."""
        validate_interface_contract(
            self.contract,
            paths=self.paths,
            schema_json=self.schema,
            markdown_text=self.bundle.markdown_text,
            prompt_yaml_text=self.bundle.prompt_yaml_text,
        )
        if not self.mode_catalog:
            raise InterfaceFrameworkError("mode_catalog must not be empty")
        missing_hosts = [host for host in _CANONICAL_HOSTS if host not in self.host_catalog]
        if missing_hosts:
            raise InterfaceFrameworkError(f"host_catalog missing canonical hosts: {missing_hosts}")
        task_thread_fields = tuple(self.contract["canonical_nouns"]["task_thread"]["fields"])
        for required_field in ("thread_id", "title", "mode", "status"):
            if required_field not in task_thread_fields:
                raise InterfaceFrameworkError(
                    f"contract task_thread.fields missing required field: {required_field}"
                )
        skill_scope_values = tuple(self.contract["canonical_nouns"]["skill"]["scope_values"])
        missing_scopes = [scope for scope in _KNOWN_SKILL_SCOPES if scope not in skill_scope_values]
        if missing_scopes:
            raise InterfaceFrameworkError(
                f"contract skill.scope_values missing runtime-required scopes: {missing_scopes}"
            )
        normalized_fields = tuple(
            self.contract["canonical_nouns"]["tool_call"]["normalized_fields"]
        )
        missing_tool_fields = [
            field for field in _TOOL_CALL_REQUIRED_FIELDS if field not in normalized_fields
        ]
        if missing_tool_fields:
            raise InterfaceFrameworkError(
                f"contract tool_call.normalized_fields missing runtime-required fields: {missing_tool_fields}"  # noqa: E501
            )

    def _build_mode_catalog(self, contract: Mapping[str, Any]) -> dict[str, ModePolicy]:
        raw_modes = contract.get("mode_semantics")
        if not isinstance(raw_modes, Mapping):
            raise InterfaceFrameworkError("contract.mode_semantics must be a mapping")
        catalog: dict[str, ModePolicy] = {}
        for name, payload in raw_modes.items():
            if not isinstance(name, str):
                raise InterfaceFrameworkError("mode names must be strings")
            if not isinstance(payload, Mapping):
                raise InterfaceFrameworkError(f"mode_semantics.{name} must be a mapping")
            intent = payload.get("intent")
            tool_policy = payload.get("tool_policy")
            default_constraints = payload.get("default_constraints", ())
            if not isinstance(intent, str) or not intent.strip():
                raise InterfaceFrameworkError(
                    f"mode_semantics.{name}.intent must be a non-empty string"
                )
            if not isinstance(tool_policy, str) or not tool_policy.strip():
                raise InterfaceFrameworkError(
                    f"mode_semantics.{name}.tool_policy must be a non-empty string"
                )
            constraints = _string_tuple(
                default_constraints, f"mode_semantics.{name}.default_constraints"
            )
            catalog[name] = ModePolicy(
                name=name,
                intent=intent,
                tool_policy=tool_policy,
                default_constraints=constraints,
            )
        return catalog

    def _build_host_catalog(self, contract: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        raw_hosts = contract.get("host_adapters")
        if not isinstance(raw_hosts, Mapping):
            raise InterfaceFrameworkError("contract.host_adapters must be a mapping")
        catalog: dict[str, tuple[str, ...]] = {}
        for name, payload in raw_hosts.items():
            if not isinstance(name, str):
                raise InterfaceFrameworkError("host names must be strings")
            if not isinstance(payload, Mapping):
                raise InterfaceFrameworkError(f"host_adapters.{name} must be a mapping")
            capabilities = _string_tuple(
                payload.get("must_support", ()), f"host_adapters.{name}.must_support"
            )
            if not capabilities:
                raise InterfaceFrameworkError(
                    f"host_adapters.{name}.must_support must not be empty"
                )
            catalog[name] = capabilities
        return catalog

    def _build_model_context(self, variant_id: str | None) -> dict[str, Any]:
        if variant_id is None:
            return {
                "variant_id": None,
                "manifest": None,
                "family_name": None,
            }
        try:
            variant = get_variant_by_id(variant_id)
        except KeyError as exc:
            raise InterfaceFrameworkError(f"unknown variant_id: {variant_id!r}") from exc
        manifest = dump_manifest(variant.manifest)
        family_name, _, variant_name = variant_id.partition("/")
        return {
            "variant_id": variant_id,
            "family_name": family_name,
            "variant_name": variant_name,
            "manifest": manifest,
        }

    def _build_instruction_stack(
        self,
        spec: TaskThreadSpec,
        mode: ModePolicy,
    ) -> tuple[str, ...]:
        layers = [
            "system policy: canonical interface contract",
            f"user task prompt: {spec.task_prompt}",
        ]
        if spec.repo_instructions:
            layers.append(f"repo instructions: {spec.repo_instructions}")
        if spec.workspace_instructions:
            layers.append(f"workspace instructions: {spec.workspace_instructions}")
        layers.append(f"mode instructions: {mode.name} ({mode.intent}; {mode.tool_policy})")
        for skill_id in spec.attached_skills:
            layers.append(f"skill instructions: {skill_id}")
        if spec.parent_checkpoint_id:
            layers.append(f"thread memory / checkpoints: {spec.parent_checkpoint_id}")
        memory_summary = spec.metadata.get("memory_summary")
        if isinstance(memory_summary, str) and memory_summary.strip():
            layers.append(f"thread memory summary: {memory_summary}")
        return tuple(layers)

    def _deterministic_thread_id(self, spec: TaskThreadSpec) -> str:
        payload = {
            "title": spec.title,
            "mode": spec.mode,
            "task_prompt": spec.task_prompt,
            "host": spec.host,
            "session_id": spec.session_id,
            "workstream_id": spec.workstream_id,
            "workstream_name": spec.workstream_name,
            "workspace": spec.workspace,
            "workspace_roots": list(spec.workspace_roots),
            "channel": spec.channel,
            "attached_skills": list(spec.attached_skills),
            "parent_thread_id": spec.parent_thread_id,
            "parent_checkpoint_id": spec.parent_checkpoint_id,
            "variant_id": self.model_context.get("variant_id"),
        }
        material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return str(uuid.uuid5(uuid.NAMESPACE_URL, material))

    def _normalize_skill_inputs(
        self,
        skill_inputs: tuple[str | SkillBundle, ...]
        | list[str | SkillBundle]
        | tuple[str, ...]
        | list[str],
    ) -> tuple[SkillBundle, ...]:
        if isinstance(skill_inputs, (str, SkillBundle)):
            raise InterfaceFrameworkError("skill_inputs must be a sequence, not a single item")
        try:
            raw_items = tuple(skill_inputs)
        except TypeError as exc:
            raise InterfaceFrameworkError("skill_inputs must be iterable") from exc
        seen: set[str] = set()
        normalized: list[SkillBundle] = []
        for item in raw_items:
            if isinstance(item, SkillBundle):
                bundle = item
            elif isinstance(item, str) and item.strip():
                bundle = SkillBundle(skill_id=item, name=item)
            else:
                raise InterfaceFrameworkError(
                    "skill_inputs entries must be non-empty strings or SkillBundle instances"
                )
            if bundle.skill_id in seen:
                continue
            seen.add(bundle.skill_id)
            normalized.append(bundle)
        return tuple(normalized)

    def _resolve_job(self, job: BackgroundJob | str) -> BackgroundJob:
        if isinstance(job, BackgroundJob):
            job_id = job.job_id
        elif isinstance(job, str) and job:
            job_id = job
        else:
            raise InterfaceFrameworkError("job must be BackgroundJob or non-empty job id")
        try:
            return self._background_jobs[job_id]
        except KeyError as exc:
            raise InterfaceFrameworkError(f"unknown background job: {job_id!r}") from exc


def _resolve_root_dir(root_dir: str | Path | None, *, fallback: Path) -> Path:
    if root_dir is None:
        return fallback
    path = Path(root_dir).expanduser().resolve()
    if not path.exists():
        raise InterfaceFrameworkError(f"root_dir does not exist: {path}")
    if not path.is_dir():
        raise InterfaceFrameworkError(f"root_dir is not a directory: {path}")
    return path


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise InterfaceFrameworkError(f"{context} must be a sequence of strings, got bare str")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise InterfaceFrameworkError(f"{context} must be iterable") from exc
    for item in items:
        if not isinstance(item, str) or not item:
            raise InterfaceFrameworkError(f"{context} entries must be non-empty strings")
    return items


def _dedupe_strings(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    return json.loads(json.dumps(payload, sort_keys=True))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_thread(thread: TaskThread) -> None:
    if not isinstance(thread, TaskThread):
        raise InterfaceFrameworkError(f"expected TaskThread, got {type(thread).__name__}")
