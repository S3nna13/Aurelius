"""Integration tests for ``src.agent.session_manager``."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import SessionManager
from src.model import (
    AureliusInterfaceFramework,
    MessageEnvelope,
    SessionRecord,
    TaskThreadSpec,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_session_manager_persists_session_workstream_and_queue_state(tmp_path):
    framework = AureliusInterfaceFramework.from_repo_root(root_dir=_repo_root())
    manager = SessionManager(root_dir=tmp_path)

    session = manager.ensure_session(workspace=tmp_path / "workspace", metadata={"integration": True})
    workstream = manager.ensure_workstream(session.session_id, "integration", workspace=tmp_path / "workspace")
    thread = framework.create_thread(
        TaskThreadSpec(
            title="Integration thread",
            mode="code",
            task_prompt="Exercise the persistent session layer.",
            host="cli",
            session_id=session.session_id,
            workstream_id=workstream.workstream_id,
            workstream_name=workstream.name,
            workspace=str(tmp_path / "workspace"),
        )
    )
    manager.register_thread(session.session_id, thread, workstream_id=workstream.workstream_id)

    approval = framework.request_approval(
        thread,
        category="file write",
        action_summary="touch session state",
        affected_resources=["src/agent/session_manager.py"],
        reason="The integration test needs an approval record.",
        reversible=True,
        minimum_scope="allow_once",
    )
    manager.register_approval(session.session_id, approval)

    checkpoint = framework.checkpoint_thread(
        thread,
        memory_summary="remember integration state",
        last_model_response="checkpoint captured",
        last_tool_result={"status": "ok"},
    )
    manager.register_checkpoint(session.session_id, checkpoint)

    routing = framework.route_channel(host="cli", channel="terminal", thread=thread, recipient="user:demo")
    manager.register_message(session.session_id, MessageEnvelope(**routing["envelope"]))

    job = framework.launch_background_job(thread, description="run detached validation")
    manager.register_background_job(session.session_id, job)
    canceled_job = manager.cancel_background_job(session.session_id, job.job_id)

    work_item = manager.queue_work_item(
        session.session_id,
        workstream.name,
        kind="cleanup",
        payload={"thread_id": thread.thread_id},
        thread_id=thread.thread_id,
    )
    canceled_item = manager.cancel_work_item(session.session_id, work_item.item_id)

    manager.set_session_status(session.session_id, "paused")
    manager.resume_session(session.session_id)
    manager.set_workstream_status(session.session_id, workstream.name, "blocked")
    manager.set_workstream_status(session.session_id, workstream.workstream_id, "active")

    snapshot = manager.snapshot(manager.get_session(session.session_id))
    json.dumps(snapshot)
    journal = manager.get_journal(session.session_id)

    reloaded = SessionManager(root_dir=tmp_path)
    reloaded_session = reloaded.get_session(session.session_id)
    reloaded_journal = reloaded.get_journal(session.session_id)

    assert isinstance(reloaded_session, SessionRecord)
    assert reloaded_session.session_id == session.session_id
    assert reloaded_session.status == "active"
    assert reloaded_session.workspace == str(tmp_path / "workspace")
    assert reloaded_session.threads[thread.thread_id].approvals == (approval.approval_id,)
    assert reloaded_session.checkpoints[checkpoint.checkpoint_id].thread_id == thread.thread_id
    assert reloaded_session.jobs[job.job_id].status == "canceled"
    assert canceled_job.status == "canceled"
    assert canceled_item.status == "canceled"
    assert reloaded_session.queue[0].item_id == work_item.item_id
    assert journal.describe()["entries"] >= 1
    assert reloaded_journal.describe()["entries"] == journal.describe()["entries"]
    assert reloaded.get_workstream(session.session_id, workstream.name).workstream_id == workstream.workstream_id
    assert reloaded.list_sessions()[0].session_id == session.session_id
