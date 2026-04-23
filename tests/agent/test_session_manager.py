"""Unit tests for ``src.agent.session_manager``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.session_manager import SessionManager, SessionRecord, WorkItem
from src.model import (
    AureliusInterfaceFramework,
    Checkpoint,
    InterfaceFrameworkError,
    MessageEnvelope,
    TaskThread,
    TaskThreadSpec,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _framework() -> AureliusInterfaceFramework:
    return AureliusInterfaceFramework.from_repo_root(root_dir=_repo_root())


def test_session_manager_persists_threads_workstreams_jobs_and_audit_state(tmp_path):
    framework = _framework()
    manager = SessionManager(root_dir=tmp_path)

    session = manager.create_session(workspace=str(tmp_path), metadata={"source": "test"})
    workstream = manager.create_workstream(session.session_id, "main", workspace=str(tmp_path))
    normalized_session = manager.ensure_session(
        session.session_id,
        workspace=tmp_path / "workspace",
        metadata={"normalized": True},
    )
    normalized_workstream = manager.ensure_workstream(
        session.session_id,
        workstream.name,
        workspace=tmp_path / "workspace",
    )

    thread = framework.create_thread(
        TaskThreadSpec(
            title="Session thread",
            mode="code",
            task_prompt="Track a persistent session.",
            host="cli",
            session_id=session.session_id,
            workstream_id=workstream.workstream_id,
            workstream_name=workstream.name,
            workspace=str(tmp_path),
        )
    )
    assert isinstance(thread, TaskThread)
    manager.register_thread(session.session_id, thread, workstream_id=workstream.workstream_id)

    approval = framework.request_approval(
        thread,
        category="file write",
        action_summary="touch session state",
        affected_resources=["src/agent/session_manager.py"],
        reason="The test needs explicit approval records.",
        reversible=True,
        minimum_scope="allow_once",
    )
    manager.register_approval(session.session_id, approval)

    checkpoint = framework.checkpoint_thread(
        thread,
        memory_summary="remember the session state",
        last_model_response="checkpoint captured",
        last_tool_result={"status": "ok"},
    )
    manager.register_checkpoint(session.session_id, checkpoint)

    routing = framework.route_channel(
        host="cli",
        channel="terminal",
        thread=thread,
        recipient="user:demo",
    )
    envelope = MessageEnvelope(**routing["envelope"])
    manager.register_message(session.session_id, envelope)

    job = framework.launch_background_job(thread, description="run detached validation")
    manager.register_background_job(session.session_id, job)
    manager.update_background_job(
        session.session_id,
        job.job_id,
        status="completed",
        result={"ok": True},
    )

    tool_call = framework.record_tool_call(
        tool_name="search_repo",
        arguments={"query": "SessionManager"},
        thread=thread,
        status="completed",
    )
    manager.register_tool_call(session.session_id, thread.thread_id, tool_call)

    work_item = manager.queue_work_item(
        session.session_id,
        workstream.workstream_id,
        kind="command",
        title="Run follow-up",
        payload={"thread_id": thread.thread_id},
        thread_id=thread.thread_id,
    )
    queued_default = manager.queue_work_item(
        session.session_id,
        workstream.name,
        kind="review",
        payload={"thread_id": thread.thread_id},
        thread_id=thread.thread_id,
    )
    updated_item = manager.update_work_item(
        session.session_id,
        work_item.item_id,
        status="completed",
        result={"done": True},
    )
    canceled_item = manager.cancel_work_item(session.session_id, queued_default.item_id)

    blocked_workstream = manager.set_workstream_status(session.session_id, workstream.workstream_id, "blocked")
    active_workstream = manager.set_workstream_status(session.session_id, workstream.name, "active")
    manager.set_session_status(session.session_id, "paused")
    resumed = manager.resume_session(session.session_id)
    status = manager.status(session.session_id)
    journal = manager.get_journal(session.session_id)
    journal_summary = manager.journal_summary(session.session_id)
    listed_messages = manager.list_messages(session.session_id, thread_id=thread.thread_id)
    snapshot = manager.snapshot(manager.get_session(session.session_id))
    json.dumps(snapshot)
    export_payload = manager.export_session(session.session_id)
    export_path = tmp_path / "exports" / "session-export.json"
    written_export_path = manager.write_session_export(session.session_id, export_path)
    imported_manager = SessionManager(state_dir=tmp_path / "imported", root_dir=tmp_path)
    imported_session = imported_manager.import_session_export(written_export_path)
    imported_journal = imported_manager.get_journal(session.session_id)
    json.dumps(export_payload)

    reloaded = SessionManager(root_dir=tmp_path)
    reloaded_session = reloaded.get_session(session.session_id)
    reloaded_journal = reloaded.get_journal(session.session_id)

    assert isinstance(session, SessionRecord)
    assert manager.session_count() == 1
    assert normalized_session.workspace == str(tmp_path / "workspace")
    assert normalized_session.metadata["normalized"] is True
    assert normalized_workstream.workspace == str(tmp_path / "workspace")
    assert blocked_workstream.status == "blocked"
    assert active_workstream.status == "active"
    assert manager.get_session(session.session_id).status == "active"
    assert resumed.status == "active"
    assert status["counts"]["threads"] == 1
    assert status["counts"]["jobs"] == 1
    assert status["counts"]["tool_calls"] == 1
    assert status["counts"]["journal_entries"] >= 1
    assert status["journal"]["entries"] >= 1
    assert status["journal"]["branches"] >= 1
    assert journal_summary["entries"] >= 1
    assert journal_summary["branches"][0]["entry_count"] >= 1
    assert export_payload["schema_version"] == "1.0"
    assert export_payload["session"]["session_id"] == session.session_id
    assert written_export_path == export_path.resolve()
    assert imported_session.session_id == session.session_id
    assert imported_journal is not None
    assert imported_journal.describe()["entries"] == journal.describe()["entries"]
    assert workstream.workstream_id in status["workstream_ids"]
    assert thread.thread_id in status["thread_ids"]
    assert updated_item.status == "completed"
    assert updated_item.result == {"done": True}
    assert canceled_item.status == "canceled"
    assert queued_default.title == "review"
    assert manager.list_tool_calls(session.session_id, thread.thread_id)[0]["tool_name"] == "search_repo"
    assert listed_messages[0].envelope_id == envelope.envelope_id
    assert manager.list_background_jobs(session.session_id, workstream.workstream_id)[0].job_id == job.job_id
    assert reloaded_session is not None
    assert reloaded_session.session_id == session.session_id
    assert reloaded_session.messages[envelope.envelope_id].kind == "routing"
    assert reloaded_session.tool_calls[thread.thread_id][0]["tool_name"] == "search_repo"
    assert reloaded_session.threads[thread.thread_id].approvals == (approval.approval_id,)
    assert reloaded_session.checkpoints[checkpoint.checkpoint_id].thread_id == thread.thread_id
    assert reloaded_session.jobs[job.job_id].status == "completed"
    assert reloaded_session.queue[0].item_id == work_item.item_id
    assert reloaded_journal is not None
    assert reloaded_journal.describe()["entries"] == journal.describe()["entries"]
    assert reloaded_journal.describe()["latest_entry_kind"] == "session.status.changed"
    assert thread.thread_id in reloaded_session.workstreams[workstream.workstream_id].thread_ids
    assert reloaded.get_workstream(session.session_id, workstream.name).workstream_id == workstream.workstream_id


def test_session_manager_rejects_unknown_background_job(tmp_path):
    manager = SessionManager(root_dir=tmp_path)
    session = manager.create_session(workspace=str(tmp_path))

    with pytest.raises(InterfaceFrameworkError, match="unknown background job"):
        manager.update_background_job(session.session_id, "job-missing", status="completed")
