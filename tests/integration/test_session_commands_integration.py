"""Integration tests for the session CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

import src.cli.main as cli_main
from src.agent.interface_runtime import AureliusInterfaceRuntime
from src.agent.session_manager import SessionManager
from src.model import AureliusInterfaceFramework


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime(tmp_path: Path) -> AureliusInterfaceRuntime:
    repo_root = _repo_root()
    framework = AureliusInterfaceFramework.from_repo_root(root_dir=repo_root)
    return AureliusInterfaceRuntime(
        framework,
        root_dir=repo_root,
        session_manager=SessionManager(state_dir=tmp_path, root_dir=repo_root),
    )


def test_main_parser_includes_session_group():
    parser = cli_main._build_parser()
    ns = parser.parse_args(["session", "thread", "list", "session-1"])
    journal_ns = parser.parse_args(["session", "journal", "list", "session-1"])
    export_ns = parser.parse_args(["session", "export", "session-1"])
    import_ns = parser.parse_args(["session", "import", "session-export.json"])

    assert ns.command == "session"
    assert ns.session_command == "thread"
    assert ns.thread_command == "list"
    assert journal_ns.command == "session"
    assert journal_ns.session_command == "journal"
    assert journal_ns.journal_command == "list"
    assert export_ns.session_command == "export"
    assert import_ns.session_command == "import"


def test_main_parser_includes_session_workstream_show():
    parser = cli_main._build_parser()
    ns = parser.parse_args(["session", "workstream", "show", "session-1", "workstream-1"])

    assert ns.command == "session"
    assert ns.session_command == "workstream"
    assert ns.workstream_command == "show"


def test_main_session_list_command_runs_end_to_end(tmp_path, capsys):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    runtime.create_workstream(session.session_id, "integration", workspace=tmp_path / "workspace")

    rc = cli_main.main(["session", "list", "--repo-root", str(_repo_root()), "--state-dir", str(tmp_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["count"] == 1
    assert payload["sessions"][0]["session_id"] == session.session_id


def test_main_session_show_command_runs_end_to_end(tmp_path, capsys):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    runtime.create_workstream(session.session_id, "integration", workspace=tmp_path / "workspace")

    rc = cli_main.main(
        ["session", "show", session.session_id, "--repo-root", str(_repo_root()), "--state-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["session"]["session_id"] == session.session_id
    assert payload["counts"]["workstreams"] == 1
    assert payload["journal"]["entries"] >= 1


def test_main_session_thread_list_command_runs_end_to_end(tmp_path, capsys):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(session.session_id, "integration", workspace=tmp_path / "workspace")
    thread = runtime.create_thread(
        {
            "title": "Integration session thread",
            "mode": "code",
            "task_prompt": "Create a thread for session CLI integration.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )

    rc = cli_main.main(
        [
            "session",
            "thread",
            "list",
            session.session_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["session_id"] == session.session_id
    assert payload["count"] == 1
    assert payload["threads"][0]["thread_id"] == thread.thread_id


def test_main_session_workstream_show_command_runs_end_to_end(tmp_path, capsys):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(session.session_id, "integration", workspace=tmp_path / "workspace")
    thread = runtime.create_thread(
        {
            "title": "Integration session workstream",
            "mode": "code",
            "task_prompt": "Create a workstream for session CLI integration.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )

    rc = cli_main.main(
        [
            "session",
            "workstream",
            "show",
            session.session_id,
            workstream.workstream_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["session_id"] == session.session_id
    assert payload["workstream"]["workstream_id"] == workstream.workstream_id
    assert thread.thread_id in payload["workstream"]["thread_ids"]


def test_main_session_export_and_import_commands_run_end_to_end(tmp_path, capsys):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(session.session_id, "integration", workspace=tmp_path / "workspace")
    thread = runtime.create_thread(
        {
            "title": "Integration export thread",
            "mode": "code",
            "task_prompt": "Export and import a session bundle.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )
    runtime.register_message(
        session.session_id,
        channel_id="terminal",
        sender="cli",
        kind="message",
        payload={"content": "export integration"},
        thread_id=thread.thread_id,
        workstream_id=workstream.workstream_id,
    )
    export_path = tmp_path / "exports" / "session-export.json"

    rc = cli_main.main(
        [
            "session",
            "export",
            session.session_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
            "--path",
            str(export_path),
        ]
    )
    captured = capsys.readouterr()
    export_payload = json.loads(captured.out)

    assert rc == 0
    assert export_payload["path"] == str(export_path.resolve())
    assert export_payload["export"]["schema_version"] == "1.0"

    imported_state_dir = tmp_path / "imported"
    rc = cli_main.main(
        [
            "session",
            "import",
            str(export_path),
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(imported_state_dir),
        ]
    )
    captured = capsys.readouterr()
    import_payload = json.loads(captured.out)

    assert rc == 0
    assert import_payload["session_id"] == session.session_id
    assert import_payload["journal"]["entries"] >= 1

    imported_runtime = _runtime(imported_state_dir)
    imported_session = imported_runtime.get_session(session.session_id)
    assert imported_session is not None
    assert imported_session.threads[thread.thread_id].thread_id == thread.thread_id
