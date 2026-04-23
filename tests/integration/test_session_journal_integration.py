"""Integration tests for the session journal CLI surface."""

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


def test_main_session_journal_commands_run_end_to_end(tmp_path, capsys):
    runtime = _runtime(tmp_path)
    session = runtime.create_session(workspace=tmp_path / "workspace")
    workstream = runtime.create_workstream(session.session_id, "integration", workspace=tmp_path / "workspace")
    thread = runtime.create_thread(
        {
            "title": "Integration journal thread",
            "mode": "code",
            "task_prompt": "Create a thread for journal inspection.",
            "host": "cli",
            "session_id": session.session_id,
            "workstream_name": workstream.name,
            "workspace": tmp_path / "workspace",
        }
    )

    rc = cli_main.main(
        [
            "session",
            "journal",
            "append",
            session.session_id,
            "note",
            "CLI note",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
            "--thread-id",
            thread.thread_id,
            "--payload-json",
            json.dumps({"source": "cli"}),
            "--tag",
            "manual",
        ]
    )
    captured = capsys.readouterr()
    append_payload = json.loads(captured.out)

    assert rc == 0
    assert append_payload["entry"]["kind"] == "note"
    assert append_payload["entry"]["thread_id"] == thread.thread_id

    rc = cli_main.main(
        [
            "session",
            "journal",
            "branch",
            session.session_id,
            "review",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
            "--from-entry-id",
            append_payload["entry"]["entry_id"],
        ]
    )
    captured = capsys.readouterr()
    branch_payload = json.loads(captured.out)
    branch_id = branch_payload["branch"]["branch_id"]

    assert rc == 0
    assert branch_payload["branch"]["name"] == "review"
    assert branch_payload["anchor_entry"]["kind"] == "journal.branch.created"

    rc = cli_main.main(
        [
            "session",
            "journal",
            "append",
            session.session_id,
            "branch.note",
            "Branch note",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
            "--branch-id",
            branch_id,
            "--thread-id",
            thread.thread_id,
            "--payload-json",
            json.dumps({"source": "branch"}),
        ]
    )
    captured = capsys.readouterr()
    branch_note_payload = json.loads(captured.out)

    assert rc == 0
    assert branch_note_payload["entry"]["kind"] == "branch.note"

    rc = cli_main.main(
        [
            "session",
            "journal",
            "append",
            session.session_id,
            "branch.review",
            "Branch review",
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
            "--branch-id",
            branch_id,
            "--thread-id",
            thread.thread_id,
            "--payload-json",
            json.dumps({"source": "branch-review"}),
        ]
    )
    captured = capsys.readouterr()
    branch_review_payload = json.loads(captured.out)

    assert rc == 0
    assert branch_review_payload["entry"]["kind"] == "branch.review"

    rc = cli_main.main(
        [
            "session",
            "journal",
            "compact",
            session.session_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
            "--branch-id",
            branch_id,
            "--keep-last-n",
            "1",
        ]
    )
    captured = capsys.readouterr()
    compact_payload = json.loads(captured.out)

    assert rc == 0
    assert compact_payload["compaction"]["branch_id"] == branch_id
    assert compact_payload["compaction"]["summary_entry_id"] is not None

    rc = cli_main.main(
        [
            "session",
            "journal",
            "list",
            session.session_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
            "--branch-id",
            branch_id,
        ]
    )
    captured = capsys.readouterr()
    list_payload = json.loads(captured.out)

    assert rc == 0
    assert list_payload["count"] >= 3
    assert any(entry["kind"] == "journal.compaction" for entry in list_payload["entries"])

    rc = cli_main.main(
        [
            "session",
            "show",
            session.session_id,
            "--repo-root",
            str(_repo_root()),
            "--state-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    session_payload = json.loads(captured.out)

    assert rc == 0
    assert session_payload["journal"]["entries"] >= 1
    assert session_payload["journal"]["branches"] >= 1
