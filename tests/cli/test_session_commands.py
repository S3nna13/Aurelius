"""Unit tests for ``src.cli.session_commands``."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from src.agent.interface_runtime import AureliusInterfaceRuntime
from src.agent.session_manager import SessionManager
from src.model import AureliusInterfaceFramework
from src.cli.session_commands import build_session_parser, dispatch_session_command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurelius")
    sub = parser.add_subparsers(dest="command")
    build_session_parser(sub)
    return parser


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


def test_session_parser_wires_subcommands():
    parser = _parser()
    ns = parser.parse_args(["session", "show", "session-1"])

    assert ns.command == "session"
    assert ns.session_command == "show"
    assert ns.session_id == "session-1"


def test_session_parser_wires_nested_thread_and_workstream_subcommands():
    parser = _parser()
    thread_ns = parser.parse_args(["session", "thread", "show", "session-1", "thread-2"])
    workstream_ns = parser.parse_args(
        ["session", "workstream", "show", "session-1", "workstream-2"]
    )
    journal_ns = parser.parse_args(["session", "journal", "list", "session-1"])

    assert thread_ns.session_command == "thread"
    assert thread_ns.thread_command == "show"
    assert thread_ns.session_id == "session-1"
    assert thread_ns.thread_id == "thread-2"
    assert workstream_ns.session_command == "workstream"
    assert workstream_ns.workstream_command == "show"
    assert workstream_ns.session_id == "session-1"
    assert workstream_ns.workstream_id == "workstream-2"
    assert journal_ns.session_command == "journal"
    assert journal_ns.journal_command == "list"
    assert journal_ns.session_id == "session-1"


def test_session_list_outputs_persistent_sessions(tmp_path: Path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    runtime.create_workstream(session.session_id, "cli-workstream", workspace=tmp_path / "workspace")

    parser = _parser()
    args = parser.parse_args(["session", "list", "--repo-root", str(_repo_root()), "--state-dir", str(tmp_path)])
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["count"] == 1
    assert payload["sessions"][0]["session_id"] == session.session_id


def test_session_thread_list_outputs_persistent_threads(tmp_path: Path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(
        session.session_id,
        "cli-workstream",
        workspace=tmp_path / "workspace",
    )
    thread = runtime.create_thread(
        {
            "title": "Thread list",
            "mode": "code",
            "task_prompt": "Create a thread for CLI session inspection.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )

    parser = _parser()
    args = parser.parse_args(
        ["session", "thread", "list", session.session_id, "--repo-root", str(_repo_root()), "--state-dir", str(tmp_path)]
    )
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["session_id"] == session.session_id
    assert payload["count"] == 1
    assert payload["threads"][0]["thread_id"] == thread.thread_id
    assert payload["threads"][0]["workstream_id"] == workstream.workstream_id


def test_session_thread_show_outputs_thread_status_and_tool_calls(tmp_path: Path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(
        session.session_id,
        "cli-workstream",
        workspace=tmp_path / "workspace",
    )
    thread = runtime.create_thread(
        {
            "title": "Thread show",
            "mode": "code",
            "task_prompt": "Create a thread for CLI session status inspection.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )
    runtime.record_tool_call(
        tool_name="search_repo",
        arguments={"query": "Aurelius"},
        thread=thread,
    )

    parser = _parser()
    args = parser.parse_args(
        [
            "session",
            "thread",
            "show",
            session.session_id,
            thread.thread_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
        ]
    )
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["thread"]["thread_id"] == thread.thread_id
    assert payload["thread"]["workstream_id"] == workstream.workstream_id
    assert payload["tool_calls"][0]["tool_name"] == "search_repo"


def test_session_show_outputs_counts_and_snapshot(tmp_path: Path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    runtime.create_workstream(session.session_id, "cli-workstream", workspace=tmp_path / "workspace")

    parser = _parser()
    args = parser.parse_args(
        ["session", "show", session.session_id, "--repo-root", str(_repo_root()), "--state-dir", str(tmp_path)]
    )
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["counts"]["workstreams"] == 1
    assert payload["journal"]["entries"] >= 1
    assert payload["session"]["session_id"] == session.session_id


def test_session_journal_list_outputs_branch_entries(tmp_path: Path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(
        session.session_id,
        "cli-workstream",
        workspace=tmp_path / "workspace",
    )
    thread = runtime.create_thread(
        {
            "title": "Journal thread",
            "mode": "code",
            "task_prompt": "Create a thread for journal inspection.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )
    runtime.session_manager.append_journal_entry(
        session.session_id,
        kind="note",
        summary="Unit test note",
        thread_id=thread.thread_id,
        payload={"source": "unit"},
    )

    parser = _parser()
    args = parser.parse_args(
        [
            "session",
            "journal",
            "list",
            session.session_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
        ]
    )
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["session_id"] == session.session_id
    assert payload["count"] >= 1
    assert payload["entries"][-1]["kind"] == "note"


def test_session_workstream_list_outputs_named_workstreams(tmp_path: Path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    runtime.create_workstream(session.session_id, "cli-workstream", workspace=tmp_path / "workspace")

    parser = _parser()
    args = parser.parse_args(
        [
            "session",
            "workstream",
            "list",
            session.session_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
        ]
    )
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["session_id"] == session.session_id
    assert payload["count"] == 1
    assert payload["workstreams"][0]["name"] == "cli-workstream"


def test_session_workstream_show_outputs_specific_workstream(tmp_path: Path):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(
        session.session_id,
        "cli-workstream",
        workspace=tmp_path / "workspace",
    )
    thread = runtime.create_thread(
        {
            "title": "Workstream show",
            "mode": "code",
            "task_prompt": "Create a thread for workstream inspection.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )

    parser = _parser()
    args = parser.parse_args(
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
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["session_id"] == session.session_id
    assert payload["workstream"]["workstream_id"] == workstream.workstream_id
    assert payload["workstream"]["name"] == "cli-workstream"
    assert thread.thread_id in payload["workstream"]["thread_ids"]


def test_session_show_unknown_session_errors(tmp_path: Path):
    parser = _parser()
    args = parser.parse_args(
        ["session", "show", "missing", "--repo-root", str(_repo_root()), "--state-dir", str(tmp_path)]
    )
    buf = io.StringIO()

    rc = dispatch_session_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 1
    assert "unknown session" in payload["error"]
