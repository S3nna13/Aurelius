"""Integration tests for the Aurelius interface framework surface."""

from __future__ import annotations

from pathlib import Path

import src.model as model_pkg
from src.model import (
    AureliusInterfaceFramework,
    BackgroundJob,
    Checkpoint,
    MessageEnvelope,
    TaskThread,
    TaskThreadSpec,
    Workstream,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_public_src_model_exports_interface_framework_surface():
    for name in (
        "AureliusInterfaceFramework",
        "TaskThreadSpec",
        "TaskThread",
        "ModePolicy",
        "SkillBundle",
        "ApprovalRequest",
        "Checkpoint",
        "BackgroundJob",
        "MessageEnvelope",
        "Workstream",
        "InterfaceFrameworkError",
    ):
        assert hasattr(model_pkg, name)
        assert name in model_pkg.__all__


def test_framework_loads_real_contract_bundle_from_repo_artifacts():
    framework = AureliusInterfaceFramework.from_repo_root(root_dir=_repo_root())
    summary = framework.describe()

    assert framework.bundle.paths.repo_root == _repo_root()
    assert framework.contract["metadata"]["title"] == "Aurelius Canonical Interface Contract"
    assert framework.schema["title"] == "Aurelius Canonical Interface Contract Schema"
    assert "code" in framework.mode_catalog
    assert "review" in framework.mode_catalog
    assert "pending_approval" in framework.contract["canonical_nouns"]["task_thread"]["status_values"]
    assert "cli" in framework.host_catalog
    assert "ide" in framework.host_catalog
    assert "allow" in framework.contract["approval_contract"]["decision_values"]
    assert "deny" in framework.contract["approval_contract"]["decision_values"]
    assert summary["tooling_seams"]["mcp_protocol_version"] == "2024-11-05"


def test_framework_round_trip_with_real_contract_bundle():
    framework = AureliusInterfaceFramework.from_repo_root(
        root_dir=_repo_root(),
        variant_id="aurelius/base-1.395b",
    )

    thread = framework.create_thread(
        TaskThreadSpec(
            title="Exercise the real contract bundle",
            mode="background",
            task_prompt="Track and resume detached work.",
            host="cli",
            workspace=str(_repo_root()),
            attached_skills=("repo/interface-framework",),
        )
    )
    routed = framework.route_channel(
        host="gateway",
        channel="inbox/aurelius",
        thread=thread,
        recipient="user:demo",
    )
    tool_call = framework.record_tool_call(
        tool_name="read_contract_bundle",
        arguments={"path": str(_repo_root())},
        host_step_id="integration-step-1",
        status="completed",
        thread=thread,
    )
    checkpoint = framework.checkpoint_thread(
        thread,
        memory_summary="Waiting on detached work completion.",
        last_model_response="Created background job.",
        last_tool_result={"status": "queued"},
    )
    resumed = framework.resume_thread(checkpoint)
    child = framework.spawn_subagent(
        resumed,
        title="Review the thread state",
        task_prompt="Validate checkpoint lineage.",
    )
    job = framework.launch_background_job(
        resumed,
        description="Continue running after foreground exit.",
    )
    canceled = framework.cancel_background_job(job)

    assert isinstance(thread, TaskThread)
    assert isinstance(resumed, TaskThread)
    assert isinstance(child, TaskThread)
    assert isinstance(checkpoint, Checkpoint)
    assert isinstance(job, BackgroundJob)
    assert canceled.status == "canceled"
    assert routed["thread_id"] == thread.thread_id
    assert routed["host"] == "gateway"
    assert tool_call["tool_name"] == "read_contract_bundle"
    assert tool_call["arguments"] == {"path": str(_repo_root())}
    assert child.parent_thread_id == resumed.thread_id
    assert child.lineage[-1] == resumed.thread_id
