"""Unit tests for ``src.model.interface_framework``."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest

from src.model.interface_framework import (
    ApprovalRequest,
    AureliusInterfaceFramework,
    BackgroundJob,
    Checkpoint,
    InterfaceFrameworkError,
    MessageEnvelope,
    ModePolicy,
    SkillBundle,
    TaskThread,
    TaskThreadSpec,
    Workstream,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _framework() -> AureliusInterfaceFramework:
    return AureliusInterfaceFramework.from_repo_root(root_dir=_repo_root())


def test_public_dataclasses_expose_minimum_framework_fields():
    expected_fields = {
        TaskThreadSpec: {"title", "mode", "task_prompt", "host"},
        TaskThread: {
            "thread_id",
            "title",
            "mode",
            "status",
            "workspace",
            "skills",
            "approvals",
            "checkpoints",
        },
        MessageEnvelope: {"envelope_id", "channel_id", "kind", "payload"},
        ModePolicy: {"name", "intent", "tool_policy"},
        SkillBundle: {"skill_id", "name", "scope"},
        ApprovalRequest: {"approval_id", "thread_id", "category"},
        Checkpoint: {"checkpoint_id", "thread_id", "thread_snapshot"},
        BackgroundJob: {"job_id", "thread_id", "status"},
        Workstream: {"workstream_id", "session_id", "name", "status"},
    }

    for cls, required_fields in expected_fields.items():
        assert is_dataclass(cls)
        assert required_fields <= set(cls.__dataclass_fields__)


def test_select_mode_and_create_thread_use_canonical_contract_modes():
    framework = _framework()
    mode = framework.select_mode("code")
    spec = TaskThreadSpec(
        title="Implement parity facade",
        mode="code",
        task_prompt="Build the Aurelius-native interface framework.",
        host="cli",
        workspace=str(_repo_root()),
        workspace_roots=(str(_repo_root()), str(_repo_root())),
        attached_skills=("repo/skill-a",),
        repo_instructions="Respect AGENTS.md semantics.",
    )

    thread_a = framework.create_thread(spec)
    thread_b = framework.create_thread(spec)

    assert isinstance(mode, ModePolicy)
    assert mode.name == "code"
    assert thread_a.thread_id == thread_b.thread_id
    assert thread_a.mode == "code"
    assert thread_a.status == "draft"
    assert thread_a.workspace == str(_repo_root())
    assert thread_a.workspace_roots == (str(_repo_root()),)
    assert thread_a.skills[0].skill_id == "repo/skill-a"
    assert any(layer.startswith("repo instructions:") for layer in thread_a.instruction_stack)


def test_create_thread_rejects_unknown_mode():
    framework = _framework()

    with pytest.raises(InterfaceFrameworkError, match="unknown mode"):
        framework.create_thread(
            TaskThreadSpec(
                title="Bad mode",
                mode="not-real",
                task_prompt="This should fail loudly.",
                host="cli",
            )
        )


def test_attach_skills_deduplicates_strings_and_skill_bundles():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Attach skills",
            mode="review",
            task_prompt="Normalize skills.",
            host="cli",
            attached_skills=("repo/skill-a",),
        )
    )

    updated = framework.attach_skills(
        thread,
        [
            "repo/skill-a",
            SkillBundle(
                skill_id="workspace/skill-b",
                name="workspace/skill-b",
                provenance="workspace",
            ),
            "workspace/skill-b",
        ],
    )

    assert [skill.skill_id for skill in updated.skills] == [
        "repo/skill-a",
        "workspace/skill-b",
    ]
    assert len(updated.instruction_stack) == len(thread.instruction_stack) + 1


def test_checkpoint_round_trip_survives_json_safe_serialization():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Checkpoint me",
            mode="background",
            task_prompt="Create and resume a checkpoint.",
            host="cli",
            workspace=str(_repo_root()),
        )
    )
    checkpoint = framework.checkpoint_thread(
        thread,
        memory_summary="Focused on interface framework coverage.",
        last_model_response="Created initial checkpoint.",
        last_tool_result={"status": "ok"},
    )

    payload = json.loads(json.dumps(asdict(checkpoint)))
    restored = Checkpoint(
        checkpoint_id=payload["checkpoint_id"],
        thread_id=payload["thread_id"],
        created_at=payload["created_at"],
        lineage=tuple(payload["lineage"]),
        thread_snapshot=payload["thread_snapshot"],
        contract_metadata=payload["contract_metadata"],
        model_context=payload["model_context"],
        memory_summary=payload["memory_summary"],
        last_model_response=payload["last_model_response"],
        last_tool_result=payload["last_tool_result"],
    )
    resumed = framework.resume_thread(restored)

    assert resumed.thread_id == thread.thread_id
    assert resumed.title == thread.title
    assert checkpoint.checkpoint_id in resumed.checkpoints
    assert resumed.mode == "background"


def test_resume_thread_rejects_malformed_snapshot():
    framework = _framework()
    bad_checkpoint = Checkpoint(
        checkpoint_id="checkpoint-bad",
        thread_id="thread-bad",
        created_at="2026-04-21T00:00:00+00:00",
        lineage=("thread-bad",),
        thread_snapshot={"title": "Missing required fields"},
        contract_metadata={
            "title": framework.contract["metadata"]["title"],
            "schema_version": framework.contract["metadata"]["schema_version"],
            "doc_path": framework.contract["metadata"]["doc_path"],
        },
        model_context=framework.model_context,
        memory_summary="summary",
    )

    with pytest.raises(InterfaceFrameworkError, match="missing required thread field"):
        framework.resume_thread(bad_checkpoint)


def test_request_approval_records_structured_decision_payload():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Approval flow",
            mode="code",
            task_prompt="Request approval for a risky write.",
            host="cli",
        )
    )

    approval = framework.request_approval(
        thread,
        category="file write",
        action_summary="Write interface framework module",
        affected_resources=["src/model/interface_framework.py"],
        reason="Modifies runtime behavior",
        reversible=True,
        minimum_scope="allow_once",
    )

    assert isinstance(approval, ApprovalRequest)
    assert approval.thread_id == thread.thread_id
    assert approval.decision == "pending"
    assert approval.affected_resources == ("src/model/interface_framework.py",)


def test_background_job_cancellation_is_explicit_and_idempotent():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Background work",
            mode="background",
            task_prompt="Run detached checks.",
            host="cli",
        )
    )

    job = framework.launch_background_job(
        thread,
        description="Run integration checks in the background.",
    )
    canceled = framework.cancel_background_job(job)
    canceled_again = framework.cancel_background_job(canceled.job_id)

    assert isinstance(job, BackgroundJob)
    assert canceled.status == "canceled"
    assert canceled.cancelable is False
    assert canceled_again.status == "canceled"
    assert canceled_again.job_id == job.job_id


def test_record_tool_call_normalizes_audit_payload_and_rejects_bad_arguments():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Audit tool call",
            mode="debug",
            task_prompt="Record tool observations.",
            host="cli",
        )
    )

    normalized = framework.record_tool_call(
        tool_name="search_repo",
        arguments={"query": "AureliusInterfaceFramework"},
        call_id="call-123",
        host_step_id="step-7",
        status="completed",
        thread=thread,
    )

    assert normalized["tool_name"] == "search_repo"
    assert normalized["arguments"] == {"query": "AureliusInterfaceFramework"}
    assert normalized["call_id"] == "call-123"
    assert normalized["host_step_id"] == "step-7"
    assert normalized["status"] == "completed"
    assert normalized["thread_id"] == thread.thread_id

    with pytest.raises(InterfaceFrameworkError, match="arguments must be a dict"):
        framework.record_tool_call(tool_name="bad", arguments="nope")  # type: ignore[arg-type]


def test_route_channel_returns_message_envelope_and_thread_context():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Route channel",
            mode="chat",
            task_prompt="Route a channel message.",
            host="cli",
            session_id="session-1",
            workstream_id="workstream-1",
            workstream_name="main",
        )
    )

    routed = framework.route_channel(
        host="cli",
        channel="terminal",
        thread=thread,
        recipient="workspace",
        metadata={"kind": "status"},
    )

    assert routed["thread_id"] == thread.thread_id
    assert routed["envelope"]["kind"] == "routing"
    assert routed["envelope"]["session_id"] == "session-1"
    assert routed["envelope"]["workstream_id"] == "workstream-1"


def test_list_tool_observations_returns_audit_trail():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Audit trail",
            mode="debug",
            task_prompt="Record audit trail entries.",
            host="cli",
        )
    )

    framework.record_tool_call(
        tool_name="lookup",
        arguments={"query": "thread"},
        thread=thread,
    )

    observations = framework.list_tool_observations(thread)
    assert len(observations) == 1
    assert observations[0]["tool_name"] == "lookup"
    assert observations[0]["thread_id"] == thread.thread_id


def test_spawn_subagent_preserves_session_and_workstream_lineage():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Parent thread",
            mode="architect",
            task_prompt="Coordinate a subtask.",
            host="cli",
            session_id="session-2",
            workstream_id="workstream-2",
            workstream_name="parallel",
        )
    )

    child = framework.spawn_subagent(
        thread,
        title="Child thread",
        task_prompt="Handle the bounded subtask.",
    )

    assert child.parent_thread_id == thread.thread_id
    assert child.session_id == thread.session_id
    assert child.workstream_id == thread.workstream_id
    assert child.lineage[-1] == thread.thread_id


def test_checkpoint_tracks_session_workstream_and_job_ids():
    framework = _framework()
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Checkpoint state",
            mode="background",
            task_prompt="Capture session metadata.",
            host="cli",
            session_id="session-3",
            workstream_id="workstream-3",
            workstream_name="background",
        )
    )
    job = framework.launch_background_job(thread, description="Audit work")

    checkpoint = framework.checkpoint_thread(
        thread,
        memory_summary="keeping state",
        last_tool_result={"status": "ok"},
    )

    assert checkpoint.session_id == "session-3"
    assert checkpoint.workstream_id == "workstream-3"
    assert checkpoint.workstream_name == "background"
    assert checkpoint.active_job_ids == ()
    assert job.metadata["session_id"] == "session-3"


def test_describe_is_json_safe_and_reports_variant_context():
    framework = AureliusInterfaceFramework.from_repo_root(
        root_dir=_repo_root(),
        variant_id="aurelius/base-1.395b",
    )
    summary = framework.describe()

    json.dumps(summary)
    assert summary["variant_id"] == "aurelius/base-1.395b"
    assert "code" in summary["mode_names"]
    assert "cli" in summary["host_names"]
