"""Integration tests for ``src.agent.workflow_shell``."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import AureliusInterfaceRuntime, SessionManager, SkillCatalog, WorkflowShell
from src.model import AureliusInterfaceFramework


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_workflow_shell_completes_and_blocks_with_json_safe_round_trips(tmp_path):
    repo_root = _repo_root()
    framework = AureliusInterfaceFramework.from_repo_root(root_dir=repo_root)
    runtime = AureliusInterfaceRuntime(
        framework,
        root_dir=repo_root,
        session_manager=SessionManager(root_dir=tmp_path),
        skill_catalog=SkillCatalog(
            repo_root,
            global_root=tmp_path / "global-skills",
            state_dir=tmp_path / ".aurelius" / "skills",
        ),
    )
    session = runtime.create_session(workspace=tmp_path)
    shell = WorkflowShell(runtime)

    completed = shell.run_workflow(
        {
            "title": "Integration workflow",
            "thread_spec": {
                "host": "cli",
                "workspace": tmp_path,
            },
            "steps": [
                {"kind": "message", "summary": "start", "content": "Begin the workflow."},
                {
                    "kind": "approval",
                    "summary": "approve write",
                    "category": "file write",
                    "action_summary": "apply integration patch",
                    "reason": "The workflow records its own state.",
                    "minimum_scope": "allow_once",
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
                    "memory_summary": "integration checkpoint",
                },
                {
                    "kind": "background_job",
                    "summary": "job",
                    "description": "run detached validation",
                    "result": {"ok": True},
                },
                {
                    "kind": "subagent",
                    "summary": "subagent",
                    "title": "child",
                    "task_prompt": "Continue the workflow.",
                },
            ],
        },
        session_id=session.session_id,
        workstream_name="integration",
        approval_decisions={"approve write": "allow"},
    )

    blocked = shell.run_workflow(
        {
            "title": "Blocked workflow",
            "thread_spec": {
                "host": "cli",
                "workspace": tmp_path,
            },
            "steps": [
                {"kind": "message", "summary": "start", "content": "Begin the workflow."},
                {
                    "kind": "approval",
                    "summary": "deny write",
                    "category": "file write",
                    "action_summary": "deny the write",
                    "reason": "This should halt execution.",
                    "minimum_scope": "allow_once",
                },
                {
                    "kind": "tool_call",
                    "summary": "tool call",
                    "tool_name": "search_repo",
                    "arguments": {"query": "never"},
                },
            ],
        },
        session_id=session.session_id,
        workstream_name="blocked",
        approval_decisions={"deny write": "deny"},
    )

    json.dumps(completed.to_dict())
    json.dumps(blocked.to_dict())

    assert completed.status == "completed"
    assert completed.halted_reason is None
    assert completed.transcript[1]["result"]["decision"] == "allow"
    assert completed.job_ids
    assert completed.checkpoint_ids
    assert completed.subthread_ids
    assert blocked.status == "blocked"
    assert blocked.halted_reason == "approval denied"
    assert len(blocked.transcript) == 2
    assert blocked.transcript[1]["result"]["status"] == "denied"
