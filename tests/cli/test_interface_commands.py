"""Unit tests for ``src.cli.interface_commands``."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from src.agent.interface_runtime import AureliusInterfaceRuntime
from src.cli.interface_commands import build_interface_parser, dispatch_interface_command
from src.ui import AureliusShell


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurelius")
    sub = parser.add_subparsers(dest="command")
    build_interface_parser(sub)
    return parser


def test_interface_parser_wires_main_style_subcommands():
    parser = _parser()
    ns = parser.parse_args(["interface", "mode", "set", "review"])

    assert ns.command == "interface"
    assert ns.interface_command == "mode"
    assert ns.mode_command == "set"
    assert ns.mode_name == "review"


def test_interface_describe_command_outputs_json():
    parser = _parser()
    args = parser.parse_args(["interface", "describe"])
    buf = io.StringIO()

    rc = dispatch_interface_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["shell_capabilities"]["workflow_execution"] is True
    assert payload["framework"]["title"] == "Aurelius Canonical Interface Contract"


def test_interface_thread_workstream_job_and_status_commands_share_shell_state():
    parser = _parser()
    shell = AureliusShell.from_repo_root(root_dir=_repo_root())

    workstream_buf = io.StringIO()
    workstream_args = parser.parse_args(
        ["interface", "workstream", "create", "--name", "cli-workstream"]
    )
    rc = dispatch_interface_command(workstream_args, workstream_buf, shell=shell)
    assert rc == 0
    workstream_id = json.loads(workstream_buf.getvalue())["workstream"]["workstream_id"]

    thread_buf = io.StringIO()
    thread_args = parser.parse_args(
        [
            "interface",
            "thread",
            "create",
            "--title",
            "CLI thread",
            "--prompt",
            "Use the Aurelius shell.",
            "--mode",
            "code",
            "--workstream-id",
            workstream_id,
            "--skill",
            "repo/skill-a",
        ]
    )
    rc = dispatch_interface_command(thread_args, thread_buf, shell=shell)
    assert rc == 0
    thread_payload = json.loads(thread_buf.getvalue())["thread"]
    thread_id = thread_payload["thread_id"]

    job_buf = io.StringIO()
    job_args = parser.parse_args(
        [
            "interface",
            "job",
            "launch",
            "--thread-id",
            thread_id,
            "--description",
            "Run detached validation.",
        ]
    )
    rc = dispatch_interface_command(job_args, job_buf, shell=shell)
    assert rc == 0
    job_payload = json.loads(job_buf.getvalue())["job"]
    job_id = job_payload["job_id"]

    status_buf = io.StringIO()
    status_args = parser.parse_args(["interface", "shell", "status"])
    rc = dispatch_interface_command(status_args, status_buf, shell=shell)
    assert rc == 0
    status_text = status_buf.getvalue()
    assert "CLI thread" in status_text
    assert "cli-workstream" in status_text
    assert job_id in status_text

    cancel_buf = io.StringIO()
    cancel_args = parser.parse_args(["interface", "job", "cancel", "--job-id", job_id])
    rc = dispatch_interface_command(cancel_args, cancel_buf, shell=shell)
    assert rc == 0
    canceled_payload = json.loads(cancel_buf.getvalue())["job"]
    assert canceled_payload["status"] == "canceled"

    job_status_buf = io.StringIO()
    job_status_args = parser.parse_args(["interface", "job", "status", "--job-id", job_id])
    rc = dispatch_interface_command(job_status_args, job_status_buf, shell=shell)
    assert rc == 0
    assert "canceled" in job_status_buf.getvalue()

    channel_send_buf = io.StringIO()
    channel_send_args = parser.parse_args(
        [
            "interface",
            "channel",
            "send",
            "--thread-id",
            thread_id,
            "--channel",
            "terminal",
            "--content",
            "hello from cli",
        ]
    )
    rc = dispatch_interface_command(channel_send_args, channel_send_buf, shell=shell)
    assert rc == 0
    channel_send_payload = json.loads(channel_send_buf.getvalue())
    assert channel_send_payload["envelope"]["channel"] == "terminal"

    channel_list_buf = io.StringIO()
    channel_list_args = parser.parse_args(
        ["interface", "channel", "list", "--thread-id", thread_id]
    )
    rc = dispatch_interface_command(channel_list_args, channel_list_buf, shell=shell)
    assert rc == 0
    channel_list_payload = json.loads(channel_list_buf.getvalue())
    assert channel_list_payload["count"] == 1
    assert channel_list_payload["messages"][0]["content"] == "hello from cli"


def test_interface_skill_catalog_and_checkpoint_workflows(tmp_path):
    parser = _parser()
    shell = AureliusShell.from_repo_root(root_dir=_repo_root())

    skill_root = tmp_path / "skills" / "example-skill"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "# Example Skill\n\nReusable instructions for the shell.\n",
        encoding="utf-8",
    )

    skill_list_buf = io.StringIO()
    skill_list_args = parser.parse_args(
        ["interface", "skill", "list", "--root", str(tmp_path / "skills")]
    )
    rc = dispatch_interface_command(skill_list_args, skill_list_buf, shell=shell)
    assert rc == 0
    skills_payload = json.loads(skill_list_buf.getvalue())
    assert skills_payload["count"] == 1
    assert skills_payload["skills"][0]["skill_id"] == "example-skill"

    thread_buf = io.StringIO()
    thread_args = parser.parse_args(
        [
            "interface",
            "thread",
            "create",
            "--title",
            "Skill thread",
            "--prompt",
            "Attach the discovered skill bundle.",
            "--mode",
            "review",
        ]
    )
    rc = dispatch_interface_command(thread_args, thread_buf, shell=shell)
    assert rc == 0
    thread_id = json.loads(thread_buf.getvalue())["thread"]["thread_id"]

    attach_buf = io.StringIO()
    attach_args = parser.parse_args(
        [
            "interface",
            "skill",
            "attach",
            "--thread-id",
            thread_id,
            "--skill",
            "example-skill",
            "--root",
            str(tmp_path / "skills"),
        ]
    )
    rc = dispatch_interface_command(attach_args, attach_buf, shell=shell)
    assert rc == 0
    attached_thread = json.loads(attach_buf.getvalue())["thread"]
    assert attached_thread["skills"][0]["skill_id"] == "example-skill"

    checkpoint_buf = io.StringIO()
    checkpoint_args = parser.parse_args(
        [
            "interface",
            "checkpoint",
            "save",
            "--thread-id",
            thread_id,
            "--memory-summary",
            "Shell checkpoint from CLI.",
            "--last-model-response",
            "ack",
        ]
    )
    rc = dispatch_interface_command(checkpoint_args, checkpoint_buf, shell=shell)
    assert rc == 0
    checkpoint_id = json.loads(checkpoint_buf.getvalue())["checkpoint"]["checkpoint_id"]

    resume_buf = io.StringIO()
    resume_args = parser.parse_args(
        ["interface", "checkpoint", "resume", "--checkpoint-id", checkpoint_id]
    )
    rc = dispatch_interface_command(resume_args, resume_buf, shell=shell)
    assert rc == 0
    resumed_thread = json.loads(resume_buf.getvalue())["thread"]
    assert resumed_thread["thread_id"] == thread_id


def test_interface_skill_catalog_management_commands(tmp_path):
    parser = _parser()
    repo_root = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"
    (repo_root / "skills" / "catalog-skill").mkdir(parents=True)
    (repo_root / "skills" / "catalog-skill" / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: catalog-skill",
                "name: Catalog Skill",
                "scope: repo",
                "---",
                "Catalog skill body.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "AGENTS.md").write_text("Repo instructions.\n", encoding="utf-8")
    workspace_root.mkdir(parents=True)
    (workspace_root / "SOUL.md").write_text("Workspace instructions.\n", encoding="utf-8")

    summary_buf = io.StringIO()
    summary_args = parser.parse_args(
        ["interface", "--repo-root", str(repo_root), "skill", "summary"]
    )
    rc = dispatch_interface_command(summary_args, summary_buf)
    assert rc == 0
    summary_payload = json.loads(summary_buf.getvalue())["summary"]
    assert summary_payload["count"] == 1

    show_buf = io.StringIO()
    show_args = parser.parse_args(
        ["interface", "--repo-root", str(repo_root), "skill", "show", "catalog-skill"]
    )
    rc = dispatch_interface_command(show_args, show_buf)
    assert rc == 0
    assert json.loads(show_buf.getvalue())["skill"]["skill_id"] == "catalog-skill"

    search_buf = io.StringIO()
    search_args = parser.parse_args(
        ["interface", "--repo-root", str(repo_root), "skill", "search", "Catalog Skill"]
    )
    rc = dispatch_interface_command(search_args, search_buf)
    assert rc == 0
    assert json.loads(search_buf.getvalue())["skills"][0]["skill_id"] == "catalog-skill"

    activate_buf = io.StringIO()
    activate_args = parser.parse_args(
        ["interface", "--repo-root", str(repo_root), "skill", "activate", "catalog-skill"]
    )
    rc = dispatch_interface_command(activate_args, activate_buf)
    assert rc == 0
    assert json.loads(activate_buf.getvalue())["skill"]["metadata"]["active"] is True

    deactivate_buf = io.StringIO()
    deactivate_args = parser.parse_args(
        ["interface", "--repo-root", str(repo_root), "skill", "deactivate", "catalog-skill"]
    )
    rc = dispatch_interface_command(deactivate_args, deactivate_buf)
    assert rc == 0
    assert json.loads(deactivate_buf.getvalue())["skill"]["metadata"]["active"] is False

    archive_buf = io.StringIO()
    archive_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(repo_root),
            "skill",
            "archive",
            "catalog-skill",
            "--reason",
            "replaced",
        ]
    )
    rc = dispatch_interface_command(archive_args, archive_buf)
    assert rc == 0
    archive_payload = json.loads(archive_buf.getvalue())["skill"]
    assert archive_payload["metadata"]["archived"] is True
    assert archive_payload["metadata"]["archived_reason"] == "replaced"

    layers_buf = io.StringIO()
    layers_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(repo_root),
            "skill",
            "layers",
            "--workspace",
            str(workspace_root),
            "--mode",
            "review",
            "--skill",
            "catalog-skill",
            "--memory-summary",
            "checkpoint memory",
        ]
    )
    rc = dispatch_interface_command(layers_args, layers_buf)
    assert rc == 0
    layers_payload = json.loads(layers_buf.getvalue())
    assert layers_payload["count"] >= 4
    assert any("Repo instructions." in layer for layer in layers_payload["layers"])
    assert any("Workspace instructions." in layer for layer in layers_payload["layers"])
    assert any("mode instructions: review" == layer for layer in layers_payload["layers"])
    assert any("skill instructions: catalog-skill:" in layer for layer in layers_payload["layers"])


def test_interface_workflow_run_command_executes_approval_gated_workflow():
    parser = _parser()
    shell = AureliusShell.from_repo_root(root_dir=_repo_root())
    thread = shell.create_thread(
        title="Workflow CLI thread",
        task_prompt="Execute a workflow transcript.",
        mode="code",
    )
    workflow = {
        "name": "CLI workflow",
        "steps": [
            {"kind": "message", "content": "start"},
            {
                "kind": "approval",
                "category": "file write",
                "action_summary": "patch the interface shell",
                "reason": "Needed for end-to-end coverage.",
                "minimum_scope": "allow_once",
                "decision": "allow",
            },
            {"kind": "tool_call", "tool_name": "search_repo", "arguments": {"query": "AureliusShell"}},
        ],
    }

    workflow_buf = io.StringIO()
    workflow_args = parser.parse_args(
        [
            "interface",
            "workflow",
            "run",
            "--thread-id",
            thread.thread_id,
            "--workflow-json",
            json.dumps(workflow),
        ]
    )
    rc = dispatch_interface_command(workflow_args, workflow_buf, shell=shell)
    assert rc == 0
    workflow_payload = json.loads(workflow_buf.getvalue())["workflow_run"]
    assert workflow_payload["status"] == "completed"
    assert {step["kind"] for step in workflow_payload["transcript"]} == {
        "message",
        "approval",
        "tool_call",
    }


def test_interface_mode_set_command_updates_shell_state():
    parser = _parser()
    shell = AureliusShell.from_repo_root(root_dir=_repo_root())

    mode_buf = io.StringIO()
    mode_args = parser.parse_args(["interface", "mode", "set", "review"])
    rc = dispatch_interface_command(mode_args, mode_buf, shell=shell)
    assert rc == 0
    payload = json.loads(mode_buf.getvalue())
    assert payload["current_mode"] == "review"
    assert shell.current_mode == "review"


def test_interface_persistent_channel_and_journal_commands(tmp_path):
    parser = _parser()
    runtime = AureliusInterfaceRuntime.from_repo_root(root_dir=_repo_root())
    runtime = AureliusInterfaceRuntime(
        runtime.framework,
        root_dir=_repo_root(),
        session_manager=runtime.session_manager.__class__(state_dir=tmp_path / "sessions", root_dir=_repo_root()),
        skill_catalog=runtime.skill_catalog,
    )
    session = runtime.create_session(session_id="session-cli", workspace=str(tmp_path))
    thread = runtime.create_thread(
        {
            "title": "Persistent CLI thread",
            "mode": "chat",
            "task_prompt": "Persist channel envelopes.",
            "session_id": session.session_id,
            "workspace": str(tmp_path),
        }
    )
    runtime.session_manager.branch_journal(session.session_id, "review")
    runtime.session_manager.compact_journal(session.session_id, branch_id="main", keep_last_n=1)

    send_buf = io.StringIO()
    send_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path / "sessions"),
            "channel",
            "send",
            "--session-id",
            session.session_id,
            "--thread-id",
            thread.thread_id,
            "--channel",
            "terminal",
            "--content",
            "persist me",
        ]
    )
    rc = dispatch_interface_command(send_args, send_buf)
    assert rc == 0
    assert json.loads(send_buf.getvalue())["envelope"]["payload"]["content"] == "persist me"

    list_buf = io.StringIO()
    list_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path / "sessions"),
            "channel",
            "list",
            "--session-id",
            session.session_id,
            "--thread-id",
            thread.thread_id,
        ]
    )
    rc = dispatch_interface_command(list_args, list_buf)
    assert rc == 0
    list_payload = json.loads(list_buf.getvalue())
    assert list_payload["count"] == 1
    assert list_payload["messages"][0]["payload"]["content"] == "persist me"

    branch_buf = io.StringIO()
    branch_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path / "sessions"),
            "journal",
            "branch",
            "--session-id",
            session.session_id,
            "--branch-id",
            "main",
        ]
    )
    rc = dispatch_interface_command(branch_args, branch_buf)
    assert rc == 0
    assert json.loads(branch_buf.getvalue())["journal"]["branch_id"] == "main"

    compaction_buf = io.StringIO()
    compaction_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path / "sessions"),
            "journal",
            "compaction",
            "--session-id",
            session.session_id,
            "--branch-id",
            "main",
        ]
    )
    rc = dispatch_interface_command(compaction_args, compaction_buf)
    assert rc == 0
    assert json.loads(compaction_buf.getvalue())["journal"]["branch_id"] == "main"

    capability_buf = io.StringIO()
    capability_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path / "sessions"),
            "capability",
            "summary",
            "--session-id",
            session.session_id,
        ]
    )
    rc = dispatch_interface_command(capability_args, capability_buf)
    assert rc == 0
    capability_payload = json.loads(capability_buf.getvalue())["capability"]
    assert capability_payload["runtime"]["session_bound"] is True
    assert capability_payload["journal"]["entries"] >= 1


def test_interface_persistent_commands_fail_loudly_for_unknown_session(tmp_path):
    parser = _parser()

    list_buf = io.StringIO()
    list_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path / "sessions"),
            "channel",
            "list",
            "--session-id",
            "missing-session",
        ]
    )
    rc = dispatch_interface_command(list_args, list_buf)
    assert rc == 1
    assert "unknown session" in json.loads(list_buf.getvalue())["error"]

    capability_buf = io.StringIO()
    capability_args = parser.parse_args(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path / "sessions"),
            "capability",
            "summary",
            "--session-id",
            "missing-session",
        ]
    )
    rc = dispatch_interface_command(capability_args, capability_buf)
    assert rc == 1
    assert "unknown session" in json.loads(capability_buf.getvalue())["error"]
