"""Unit tests for ``src.cli.interface_commands``."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

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
