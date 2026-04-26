"""Unit tests for ``src.agent.workflow_shell``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.interface_runtime import AureliusInterfaceRuntime
from src.agent.session_manager import SessionManager
from src.agent.skill_catalog import SkillCatalog
from src.agent.workflow_shell import WorkflowShell
from src.model import AureliusInterfaceFramework, InterfaceFrameworkError


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime(tmp_path: Path) -> AureliusInterfaceRuntime:
    repo_root = _repo_root()
    framework = AureliusInterfaceFramework.from_repo_root(root_dir=repo_root)
    return AureliusInterfaceRuntime(
        framework,
        root_dir=repo_root,
        session_manager=SessionManager(root_dir=tmp_path),
        skill_catalog=SkillCatalog(repo_root, state_dir=tmp_path / ".aurelius" / "skills"),
    )


def test_workflow_shell_runs_approval_gated_steps_and_serializes(tmp_path):
    runtime = _runtime(tmp_path)
    shell = WorkflowShell(runtime)
    session = runtime.create_session(workspace=str(tmp_path))

    workflow = {
        "name": "Parity workflow",
        "thread_spec": {
            "host": "cli",
            "workspace": str(tmp_path),
        },
        "steps": [
            {"kind": "message", "summary": "start", "content": "Begin the workflow."},
            {
                "kind": "approval",
                "summary": "approve write",
                "category": "file write",
                "action_summary": "apply patch",
                "reason": "The workflow needs to record its result.",
                "minimum_scope": "allow_once",
                "reversible": True,
            },
            {
                "kind": "tool_call",
                "summary": "tool call",
                "tool_name": "search_repo",
                "arguments": {"query": "WorkflowShell"},
                "status": "completed",
            },
            {
                "kind": "checkpoint",
                "summary": "checkpoint",
                "memory_summary": "workflow checkpoint",
            },
            {
                "kind": "background_job",
                "summary": "job",
                "description": "run detached validation",
            },
            {
                "kind": "subagent",
                "summary": "subagent",
                "title": "child",
                "task_prompt": "Continue the workflow.",
            },
        ],
    }

    run = shell.run_workflow(
        workflow,
        session_id=session.session_id,
        workstream_name="workflow",
        approval_decisions={"approve write": "allow"},
    )

    assert run.status == "completed"
    assert run.session_id == session.session_id
    assert run.workstream_id
    assert [step.kind for step in run.steps] == [
        "message",
        "approval",
        "tool_call",
        "checkpoint",
        "background_job",
        "subagent",
    ]
    assert run.transcript[1]["result"]["decision"] == "allow"
    assert run.transcript[-1]["result"]["subthread"]["title"] == "child"
    assert run.checkpoint_ids
    assert run.job_ids
    assert run.subthread_ids
    assert run.approval_ids
    stored_approval = runtime.session_manager.get_approval(
        session.session_id,
        run.approval_ids[0],
    )
    assert stored_approval is not None
    assert stored_approval.decision == "allow"
    stored_thread = runtime.session_manager.get_thread(session.session_id, run.thread_id)
    assert stored_thread is not None
    assert stored_thread.status == "completed"
    assert run.approval_ids[0] in stored_thread.approvals
    stored_checkpoint = runtime.session_manager.get_checkpoint(
        session.session_id, run.checkpoint_ids[0]
    )
    assert stored_checkpoint is not None
    assert run.approval_ids[0] in stored_checkpoint.thread_snapshot["approvals"]
    assert run.halted_reason is None
    json.dumps(run.to_dict())


def test_workflow_shell_blocks_on_denied_approval(tmp_path):
    shell = WorkflowShell(_runtime(tmp_path))
    session = shell.runtime.create_session(workspace=str(tmp_path))

    run = shell.run_workflow(
        {
            "name": "Blocked workflow",
            "thread_spec": {"host": "cli", "workspace": str(tmp_path)},
            "steps": [
                {"kind": "message", "content": "start"},
                {
                    "kind": "approval",
                    "category": "file write",
                    "action_summary": "deny the write",
                    "reason": "This should halt execution.",
                    "minimum_scope": "allow_once",
                },
                {"kind": "tool_call", "tool_name": "search_repo", "arguments": {"query": "never"}},
            ],
        },
        session_id=session.session_id,
        workstream_name="blocked",
        approval_decisions={"approval": "deny"},
    )

    assert run.status == "blocked"
    assert run.halted_reason == "approval denied"
    assert len(run.transcript) == 2
    assert run.transcript[1]["result"]["status"] == "denied"
    stored_approval = shell.runtime.session_manager.get_approval(
        session.session_id,
        run.approval_ids[0],
    )
    assert stored_approval is not None
    assert stored_approval.decision == "deny"
    stored_thread = shell.runtime.session_manager.get_thread(session.session_id, run.thread_id)
    assert stored_thread is not None
    assert stored_thread.status == "blocked"
    json.dumps(run.to_dict())


def test_workflow_shell_rejects_malformed_workflow_input(tmp_path):
    shell = WorkflowShell(_runtime(tmp_path))

    with pytest.raises(InterfaceFrameworkError, match="workflow steps must be a sequence"):
        shell.normalize_workflow({"steps": "not a list"})


def test_workflow_shell_macro_expansion_uses_unique_step_ids(tmp_path):
    shell = WorkflowShell(_runtime(tmp_path))

    normalized = shell.normalize_workflow(
        {
            "name": "Macro workflow",
            "thread_spec": {"host": "cli", "workspace": str(tmp_path)},
            "macros": {
                "common": [
                    {"kind": "message", "content": "one"},
                    {"kind": "message", "content": "two"},
                ],
            },
            "steps": [
                {"kind": "macro", "name": "common"},
                {"kind": "message", "content": "three"},
            ],
        }
    )

    assert [step.step_id for step in normalized["steps"]] == ["step-1", "step-2", "step-3"]
