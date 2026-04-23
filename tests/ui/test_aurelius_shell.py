"""Unit tests for ``src.ui.aurelius_shell``."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.ui import AureliusShell, AureliusShellError, MessageEnvelope, SkillRecord, WorkflowRun


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _shell() -> AureliusShell:
    return AureliusShell.from_repo_root(root_dir=_repo_root())


def test_shell_is_exported_through_src_ui_package():
    assert AureliusShell.__name__ == "AureliusShell"
    assert MessageEnvelope.__name__ == "MessageEnvelope"
    assert SkillRecord.__name__ == "SkillRecord"
    assert WorkflowRun.__name__ == "WorkflowRun"


def test_shell_describe_create_thread_and_render_status():
    shell = _shell()
    workstream = shell.create_workstream("Parity Harness")
    thread = shell.create_thread(
        title="Implement shell integration",
        task_prompt="Wire the Aurelius terminal shell to the interface framework.",
        mode="code",
        workstream_id=workstream.workstream_id,
        skills=("repo/skill-a",),
    )

    summary = shell.describe()
    json.dumps(summary)

    assert summary["counts"]["threads"] == 1
    assert summary["current_mode"] == shell.current_mode
    assert summary["backend_surface"]["count"] >= 1
    assert "pytorch" in summary["backend_surface"]["names"]
    assert "Implement shell integration" in shell.render_status()
    assert workstream.name in shell.render_status()
    assert thread.thread_id in shell.render_thread_status(thread.thread_id)
    assert "repo/skill-a" in shell.render_thread_status(thread.thread_id)
    assert "Backends:" in shell.render_status()


def test_shell_skill_discovery_attachment_and_execute_command_surface(tmp_path):
    shell = _shell()
    skill_root = tmp_path / "skills" / "demo-skill"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "# Demo Skill\n\nLocal-first skill bundle for terminal workflows.\n",
        encoding="utf-8",
    )

    discovered = shell.discover_skills([tmp_path / "skills"])
    assert len(discovered) == 1
    assert discovered[0].skill_id == "demo-skill"
    assert discovered[0].name == "Demo Skill"

    thread = shell.create_thread(
        title="Skill attach",
        task_prompt="Attach discovered skills.",
        mode="review",
    )
    updated = shell.attach_skill_ids(thread.thread_id, [discovered[0].skill_id], roots=[tmp_path / "skills"])

    assert updated.skills[0].skill_id == "demo-skill"
    assert "Demo Skill" in shell.render_thread_status(updated.thread_id)
    assert "Aurelius Shell" in shell.execute_command("status")

    mode_payload = json.loads(shell.execute_command("mode set review"))
    assert mode_payload["current_mode"] == "review"
    backend_payload = json.loads(shell.execute_command("backend list"))
    assert backend_payload["count"] >= 1
    assert "pytorch" in backend_payload["names"]
    backend_show = json.loads(shell.execute_command("backend show pytorch"))
    assert backend_show["backend"]["backend_name"] == "pytorch"


def test_shell_skill_catalog_commands_route_through_local_catalog(monkeypatch):
    shell = _shell()
    monkeypatch.setattr(shell, "catalog_skill_summary", lambda: {"count": 1, "archived_count": 0})
    monkeypatch.setattr(
        shell,
        "catalog_skill_show",
        lambda skill_id: {"skill_id": skill_id, "name": "Demo Skill"},
    )
    monkeypatch.setattr(
        shell,
        "catalog_skill_search",
        lambda query: [{"skill_id": "demo-skill", "name": "Demo Skill", "query": query}],
    )

    summary_payload = json.loads(shell.execute_command("skill summary"))
    assert summary_payload["summary"]["count"] == 1

    show_payload = json.loads(shell.execute_command("skill show demo-skill"))
    assert show_payload["skill"]["skill_id"] == "demo-skill"

    search_payload = json.loads(shell.execute_command("skill search Demo Skill"))
    assert search_payload["count"] == 1
    assert search_payload["skills"][0]["name"] == "Demo Skill"


def test_shell_capability_and_journal_commands_route_through_runtime_summaries(monkeypatch):
    shell = _shell()
    monkeypatch.setattr(
        shell,
        "capability_summary",
        lambda: {"framework": {"title": "Aurelius"}, "runtime": {"session_bound": False}},
    )
    monkeypatch.setattr(
        shell,
        "capability_summary_schema",
        lambda: {"schema_name": "aurelius.interface.capability-summary", "schema_version": "1.0"},
    )
    monkeypatch.setattr(
        shell,
        "journal_branch_summary",
        lambda branch_id="main": {"branch_id": branch_id, "entry_count": 2},
    )
    monkeypatch.setattr(
        shell,
        "journal_compaction_summary",
        lambda branch_id=None, compaction_id=None: {"branch_id": branch_id, "compaction_id": compaction_id},
    )

    capability_payload = json.loads(shell.execute_command("capability summary"))
    assert capability_payload["capability"]["framework"]["title"] == "Aurelius"

    schema_payload = json.loads(shell.execute_command("capability schema"))
    assert schema_payload["schema"]["schema_version"] == "1.0"

    branch_payload = json.loads(shell.execute_command("journal branch main"))
    assert branch_payload["journal"]["branch_id"] == "main"

    compaction_payload = json.loads(shell.execute_command("journal compaction main"))
    assert compaction_payload["journal"]["branch_id"] == "main"


def test_shell_approval_checkpoint_resume_subagent_job_and_routing():
    shell = _shell()
    thread = shell.create_thread(
        title="Workflow parent",
        task_prompt="Exercise approvals, checkpoints, jobs, and routing.",
        mode="debug",
    )

    approval = shell.request_approval(
        thread,
        category="file write",
        action_summary="update interface shell",
        affected_resources=("src/ui/aurelius_shell.py",),
        reason="This changes runtime behavior.",
        reversible=True,
        minimum_scope="allow_once",
    )
    assert approval.approval_id in shell._threads[thread.thread_id].approvals
    checkpoint = shell.checkpoint_thread(
        thread,
        memory_summary="Focused on shell integration coverage.",
        last_model_response="checkpoint captured",
    )
    resumed = shell.resume_thread(checkpoint)
    child = shell.spawn_subagent(
        resumed,
        title="Subagent child",
        task_prompt="Validate shell rendering.",
    )
    job = shell.launch_background_job(child, description="Run detached validation.")
    canceled = shell.cancel_background_job(job)
    routing = shell.route_channel(
        channel="terminal",
        content="hello from the shell",
        thread=child,
        sender="tester",
    )
    tool_call = shell.record_tool_call(
        tool_name="search_repo",
        arguments={"query": "AureliusShell"},
        thread=child,
    )

    assert approval.thread_id == thread.thread_id
    assert checkpoint.checkpoint_id in shell.render_thread_status(resumed.thread_id)
    assert child.parent_thread_id == thread.thread_id
    assert canceled.status == "canceled"
    assert routing["envelope"]["channel"] == "terminal"
    assert tool_call["tool_name"] == "search_repo"
    assert "canceled" in shell.render_status()
    listed_messages = json.loads(shell.execute_command("channel list"))
    assert listed_messages["count"] >= 1
    assert listed_messages["messages"][0]["channel"] == "terminal"


def test_shell_workflow_execution_and_snapshot_round_trip():
    shell = _shell()
    workstream = shell.create_workstream("Workflow View")
    thread = shell.create_thread(
        title="Workflow root",
        task_prompt="Run a full workflow transcript.",
        mode="code",
        workstream_id=workstream.workstream_id,
    )
    workflow = {
        "name": "Parity workflow",
        "metadata": {"source": "test"},
        "steps": [
            {"kind": "message", "content": "start"},
            {
                "kind": "approval",
                "category": "file write",
                "action_summary": "apply patch",
                "reason": "Needed for parity coverage.",
                "minimum_scope": "allow_once",
                "reversible": True,
                "decision": "allow",
            },
            {"kind": "tool_call", "tool_name": "search_repo", "arguments": {"query": "Aurelius"}},
            {"kind": "checkpoint", "memory_summary": "checkpoint here"},
            {"kind": "background_job", "description": "run detached validation"},
            {"kind": "subagent", "title": "child", "task_prompt": "continue analysis"},
        ],
    }

    run = shell.execute_workflow(thread, workflow)
    json.dumps(asdict(run))

    assert run.status == "completed"
    assert run.workflow_name == "Parity workflow"
    assert run.final_artifact is not None
    assert {entry["kind"] for entry in run.transcript} >= {
        "message",
        "approval",
        "tool_call",
        "checkpoint",
        "background_job",
        "subagent",
    }
    approval = next(
        approval for approval in shell.approvals if approval.action_summary == "apply patch"
    )
    assert approval.decision == "allow"
    assert approval.decided_at is not None
    assert "Status: completed" in shell.render_thread_status(run.thread_id)

    snapshot = shell.snapshot()
    json.dumps(snapshot)
    restored = AureliusShell.restore_snapshot(snapshot, root_dir=_repo_root())

    assert workstream.name in restored.render_status()
    assert thread.title in restored.render_status()
    assert restored.active_thread_id is not None
    assert restored.workflow_runs[0].final_artifact == run.final_artifact
    assert "pytorch" in restored.describe()["backend_surface"]["names"]


def test_shell_rejects_malformed_workflow_input():
    shell = _shell()
    thread = shell.create_thread(
        title="Malformed workflow",
        task_prompt="This should fail loudly.",
        mode="code",
    )

    with pytest.raises(AureliusShellError, match="missing kind"):
        shell.execute_workflow(thread, {"steps": [{"content": "missing kind"}]})
