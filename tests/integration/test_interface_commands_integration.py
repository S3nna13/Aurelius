"""Integration tests for the Aurelius interface CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.interface_runtime import AureliusInterfaceRuntime
import src.cli.main as cli_main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_main_parser_includes_interface_group():
    parser = cli_main._build_parser()
    ns = parser.parse_args(["interface", "describe"])

    assert ns.command == "interface"
    assert ns.interface_command == "describe"


def test_main_interface_describe_command_runs_end_to_end(capsys):
    rc = cli_main.main(["interface", "describe"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["shell_capabilities"]["workflow_execution"] is True
    assert payload["framework"]["title"] == "Aurelius Canonical Interface Contract"


def test_main_interface_shell_status_command_runs_end_to_end(capsys):
    rc = cli_main.main(["interface", "shell", "status"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Aurelius Shell" in captured.out
    assert "Workflow runs:" in captured.out


def test_main_interface_channel_commands_run_end_to_end(capsys):
    rc = cli_main.main(
        [
            "interface",
            "channel",
            "send",
            "--channel",
            "terminal",
            "--content",
            "integration hello",
            "--allow-unbound",
        ]
    )
    captured = capsys.readouterr()
    send_payload = json.loads(captured.out)

    assert rc == 0
    assert send_payload["envelope"]["content"] == "integration hello"


def test_main_interface_persistent_channel_and_summary_commands_run_end_to_end(tmp_path, capsys):
    state_dir = tmp_path / "sessions"
    runtime = AureliusInterfaceRuntime.from_repo_root(root_dir=_repo_root())
    runtime = AureliusInterfaceRuntime(
        runtime.framework,
        root_dir=_repo_root(),
        session_manager=runtime.session_manager.__class__(state_dir=state_dir, root_dir=_repo_root()),
        skill_catalog=runtime.skill_catalog,
    )
    session = runtime.create_session(session_id="session-main", workspace=str(tmp_path))
    thread = runtime.create_thread(
        {
            "title": "Main persistent thread",
            "mode": "chat",
            "task_prompt": "Persist interface channel state.",
            "session_id": session.session_id,
            "workspace": str(tmp_path),
        }
    )
    runtime.session_manager.branch_journal(session.session_id, "review")
    runtime.session_manager.compact_journal(session.session_id, branch_id="main", keep_last_n=1)

    rc = cli_main.main(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(state_dir),
            "channel",
            "send",
            "--session-id",
            session.session_id,
            "--thread-id",
            thread.thread_id,
            "--channel",
            "terminal",
            "--content",
            "persistent integration hello",
        ]
    )
    captured = capsys.readouterr()
    send_payload = json.loads(captured.out)

    assert rc == 0
    assert send_payload["envelope"]["payload"]["content"] == "persistent integration hello"

    rc = cli_main.main(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(state_dir),
            "channel",
            "list",
            "--session-id",
            session.session_id,
            "--thread-id",
            thread.thread_id,
        ]
    )
    captured = capsys.readouterr()
    list_payload = json.loads(captured.out)

    assert rc == 0
    assert list_payload["count"] == 1

    rc = cli_main.main(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(state_dir),
            "journal",
            "branch",
            "--session-id",
            session.session_id,
            "--branch-id",
            "main",
        ]
    )
    captured = capsys.readouterr()
    branch_payload = json.loads(captured.out)

    assert rc == 0
    assert branch_payload["journal"]["branch_id"] == "main"

    rc = cli_main.main(
        [
            "interface",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(state_dir),
            "capability",
            "summary",
            "--session-id",
            session.session_id,
        ]
    )
    captured = capsys.readouterr()
    capability_payload = json.loads(captured.out)

    assert rc == 0
    assert capability_payload["capability"]["schema"]["schema_version"] == "1.0"
    assert capability_payload["capability"]["runtime"]["session_bound"] is True


def test_main_interface_capability_schema_command_runs_end_to_end(capsys):
    rc = cli_main.main(["interface", "--repo-root", str(_repo_root()), "capability", "schema"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["schema"]["schema_name"] == "aurelius.interface.capability-summary"
    assert payload["schema"]["schema_version"] == "1.0"


def test_main_interface_surface_commands_run_end_to_end(capsys):
    rc = cli_main.main(["interface", "--repo-root", str(_repo_root()), "surface", "summary"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["surface"]["engine_adapters"]["count"] >= 3
    assert payload["surface"]["ui"]["ui_surfaces"]["count"] >= 1

    rc = cli_main.main(["interface", "--repo-root", str(_repo_root()), "surface", "schema"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["schema"]["schema_name"] == "aurelius.interface.surface-catalog"


def test_main_interface_skill_catalog_management_commands_run_end_to_end(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"
    (repo_root / "skills" / "integration-skill").mkdir(parents=True)
    (repo_root / "skills" / "integration-skill" / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: integration-skill",
                "name: Integration Skill",
                "scope: repo",
                "---",
                "Integration skill body.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "AGENTS.md").write_text("Repo instructions.\n", encoding="utf-8")
    workspace_root.mkdir(parents=True)
    (workspace_root / "TOOLS.md").write_text("Workspace tools.\n", encoding="utf-8")

    rc = cli_main.main(["interface", "--repo-root", str(repo_root), "skill", "summary"])
    captured = capsys.readouterr()
    summary_payload = json.loads(captured.out)

    assert rc == 0
    assert summary_payload["summary"]["count"] == 1

    rc = cli_main.main(
        ["interface", "--repo-root", str(repo_root), "skill", "activate", "integration-skill"]
    )
    captured = capsys.readouterr()
    activate_payload = json.loads(captured.out)

    assert rc == 0
    assert activate_payload["skill"]["metadata"]["active"] is True

    rc = cli_main.main(
        [
            "interface",
            "--repo-root",
            str(repo_root),
            "skill",
            "archive",
            "integration-skill",
            "--reason",
            "deprecated",
        ]
    )
    captured = capsys.readouterr()
    archive_payload = json.loads(captured.out)

    assert rc == 0
    assert archive_payload["skill"]["metadata"]["archived"] is True
    assert archive_payload["skill"]["metadata"]["archived_reason"] == "deprecated"

    rc = cli_main.main(
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
            "integration-skill",
        ]
    )
    captured = capsys.readouterr()
    layers_payload = json.loads(captured.out)

    assert rc == 0
    assert any("Repo instructions." in layer for layer in layers_payload["layers"])
    assert any("Workspace tools." in layer for layer in layers_payload["layers"])
    assert any("mode instructions: review" == layer for layer in layers_payload["layers"])
