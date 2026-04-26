"""Integration tests for ``src.agent.interface_runtime``."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import (
    AureliusInterfaceRuntime,
    SessionManager,
    SkillCatalog,
    WorkflowShell,
    WorkItem,
)
from src.model import AureliusInterfaceFramework


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_agent_package_exports_runtime_surface():
    assert AureliusInterfaceRuntime.__name__ == "AureliusInterfaceRuntime"
    assert SessionManager.__name__ == "SessionManager"
    assert SkillCatalog.__name__ == "SkillCatalog"
    assert WorkflowShell.__name__ == "WorkflowShell"
    assert WorkItem.__name__ == "WorkItem"


def test_interface_runtime_end_to_end_session_workstream_and_workflow(tmp_path):
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

    skill_root = tmp_path / "source-skill"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: integration-skill",
                "name: Integration Skill",
                "scope: workspace",
                "---",
                "Integration skill body.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    installed = runtime.skill_catalog.install(skill_root, scope="repo")

    session = runtime.create_session(workspace=tmp_path, metadata={"integration": True})
    workstream = runtime.create_workstream(session.session_id, "integration", workspace=tmp_path)
    thread = runtime.create_thread(
        {
            "title": "Integration thread",
            "mode": "code",
            "task_prompt": "Exercise the runtime integration path.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path,
            "attached_skills": [installed.skill_id],
        }
    )

    shell = runtime.workflow_shell()
    run = shell.run_workflow(
        {
            "title": "Integration workflow",
            "thread_spec": {
                "host": "cli",
                "workspace": tmp_path,
                "attached_skills": [installed.skill_id],
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
                    "arguments": {"query": "AureliusInterfaceRuntime"},
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
                    "attached_skills": [installed.skill_id],
                },
            ],
        },
        thread=thread,
        approval_decisions={"approve write": "allow"},
    )

    reloaded_runtime = AureliusInterfaceRuntime(
        AureliusInterfaceFramework.from_repo_root(root_dir=repo_root),
        root_dir=repo_root,
        session_manager=SessionManager(root_dir=tmp_path),
        skill_catalog=SkillCatalog(
            repo_root,
            global_root=tmp_path / "global-skills",
            state_dir=tmp_path / ".aurelius" / "skills",
        ),
    )
    status = runtime.session_status(session.session_id)
    thread_status = reloaded_runtime.thread_status(session.session_id, thread.thread_id)
    stored_thread = reloaded_runtime.get_thread(session.session_id, thread.thread_id)
    stored_checkpoint = reloaded_runtime.session_manager.get_checkpoint(
        session.session_id, run.checkpoint_ids[0]
    )
    json.dumps(run.to_dict())
    json.dumps(runtime.describe())

    assert run.status == "completed"
    assert run.session_id == session.session_id
    assert run.workstream_id == workstream.workstream_id
    assert thread.skills[0].skill_id == installed.skill_id
    assert thread_status["tool_calls"][0]["tool_name"] == "search_repo"
    assert stored_thread is not None
    assert run.approval_ids[0] in stored_thread.approvals
    assert stored_checkpoint is not None
    assert run.approval_ids[0] in stored_checkpoint.thread_snapshot["approvals"]
    assert status["counts"]["threads"] >= 1
    assert status["counts"]["jobs"] >= 1
    assert status["journal"]["entries"] >= 1
    assert runtime.get_thread(session.session_id, thread.thread_id).thread_id == thread.thread_id
    assert runtime.get_background_job(session.session_id, run.job_ids[0]).status == "completed"
