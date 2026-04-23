"""Unit tests for ``src.agent.interface_runtime``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.interface_runtime import AureliusInterfaceRuntime
from src.agent.session_manager import SessionManager
from src.agent.skill_catalog import SkillCatalog
from src.model import (
    ApprovalRequest,
    AureliusInterfaceFramework,
    BackgroundJob,
    Checkpoint,
    InterfaceFrameworkError,
    TaskThread,
    TaskThreadSpec,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime(tmp_path: Path) -> AureliusInterfaceRuntime:
    repo_root = _repo_root()
    framework = AureliusInterfaceFramework.from_repo_root(root_dir=repo_root)
    session_manager = SessionManager(root_dir=tmp_path)
    skill_catalog = SkillCatalog(repo_root, state_dir=tmp_path / ".aurelius" / "skills")
    return AureliusInterfaceRuntime(
        framework,
        root_dir=repo_root,
        session_manager=session_manager,
        skill_catalog=skill_catalog,
    )


def test_runtime_drives_the_full_thread_session_and_audit_lifecycle(tmp_path):
    runtime = _runtime(tmp_path)

    skill_root = tmp_path / "source-skill"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: runtime-skill",
                "name: Runtime Skill",
                "scope: workspace",
                "---",
                "Runtime skill body.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    installed = runtime.skill_catalog.install(skill_root, scope="repo")
    activated = runtime.activate_skill(installed.skill_id)

    session = runtime.create_session(workspace=str(tmp_path), metadata={"source": "runtime"})
    workstream = runtime.create_workstream(session.session_id, "runtime-workstream", workspace=str(tmp_path))
    thread = runtime.create_thread(
        TaskThreadSpec(
            title="Runtime thread",
            mode="code",
            task_prompt="Exercise the Aurelius runtime facade.",
            host="cli",
            session_id=session.session_id,
            workstream_name=workstream.name,
            workspace=str(tmp_path),
            attached_skills=(installed.skill_id,),
            metadata={"memory_summary": "remember this"},
        )
    )

    assert thread.skills[0].skill_id == installed.skill_id
    assert thread.skills[0].scope == "thread"

    approval = runtime.request_approval(
        thread,
        category="file write",
        action_summary="record runtime state",
        affected_resources=["src/agent/interface_runtime.py"],
        reason="The runtime should create structured approvals.",
        reversible=True,
        minimum_scope="allow_once",
    )
    checkpoint = runtime.checkpoint_thread(
        thread,
        memory_summary="checkpoint the runtime thread",
        last_model_response="checkpointed",
        last_tool_result={"status": "ok"},
    )
    resumed = runtime.resume_thread(checkpoint)
    child = runtime.spawn_subagent(
        resumed,
        title="Runtime child",
        task_prompt="Validate the inherited runtime state.",
    )
    job = runtime.launch_background_job(resumed, description="Detached runtime work")
    canceled = runtime.cancel_background_job(job)
    routing = runtime.route_channel(
        host="cli",
        channel="terminal",
        thread=resumed,
        recipient="user:demo",
    )
    tool_call = runtime.record_tool_call(
        tool_name="search_repo",
        arguments={"query": "AureliusInterfaceRuntime"},
        call_id="call-1",
        host_step_id="step-1",
        status="completed",
        thread=resumed,
    )
    runtime.register_message(
        session.session_id,
        channel_id="gateway",
        sender="cli",
        kind="message",
        payload={"content": "persisted envelope"},
        thread_id=resumed.thread_id,
        recipient="user:demo",
    )
    runtime.session_manager.branch_journal(session.session_id, "review")
    runtime.session_manager.compact_journal(session.session_id, branch_id="main", keep_last_n=1)
    export_payload = runtime.export_session(session.session_id)
    imported_runtime = _runtime(tmp_path / "imported")
    imported_session = imported_runtime.import_session(export_payload, replace=True)
    reloaded_runtime = _runtime(tmp_path)
    thread_status = reloaded_runtime.thread_status(session.session_id, thread.thread_id)
    session_status = runtime.session_status(session.session_id)
    messages = reloaded_runtime.list_messages(session.session_id, thread_id=resumed.thread_id)
    branch_summary = runtime.journal_branch_summary(session.session_id, "main")
    compaction_summary = runtime.journal_compaction_summary(session.session_id, branch_id="main")
    capability_schema = runtime.capability_summary_schema()
    capability = runtime.capability_summary(session.session_id)
    summary = runtime.describe()
    json.dumps(summary)
    json.dumps(thread_status)
    json.dumps(export_payload)

    assert isinstance(approval, ApprovalRequest)
    assert isinstance(checkpoint, Checkpoint)
    assert isinstance(resumed, TaskThread)
    assert isinstance(child, TaskThread)
    assert isinstance(job, BackgroundJob)
    assert canceled.status == "canceled"
    assert approval.thread_id == thread.thread_id
    assert checkpoint.thread_id == thread.thread_id
    assert child.parent_thread_id == thread.thread_id
    assert activated.metadata["active"] is True
    assert routing["envelope"]["kind"] == "routing"
    assert tool_call["call_id"] == "call-1"
    assert messages[-1]["payload"]["content"] == "persisted envelope"
    assert branch_summary["branch_id"] == "main"
    assert compaction_summary["branch_id"] == "main"
    assert export_payload["schema_version"] == "1.0"
    assert imported_session.session_id == session.session_id
    assert capability_schema["schema_version"] == "1.0"
    assert capability["schema"]["schema_name"] == capability_schema["schema_name"]
    assert capability["runtime"]["session_bound"] is True
    assert capability["journal"]["entries"] >= 1
    assert thread_status["thread"]["thread_id"] == thread.thread_id
    assert thread_status["tool_calls"][0]["tool_name"] == "search_repo"
    assert reloaded_runtime.get_thread(session.session_id, thread.thread_id).thread_id == thread.thread_id
    assert session_status["counts"]["threads"] >= 2
    assert session_status["counts"]["jobs"] == 1
    assert session_status["journal"]["entries"] >= 1
    assert summary["session_count"] == 1
    assert summary["session_ids"] == [session.session_id]
    assert runtime.get_background_job(session.session_id, job.job_id).status == "canceled"
    assert runtime.get_thread(session.session_id, thread.thread_id).thread_id == thread.thread_id
    assert runtime.list_skills(active=True)[0].skill_id == installed.skill_id
    assert runtime.search_skills("Runtime Skill")[0].skill_id == installed.skill_id
    assert runtime.describe()["skill_catalog"]["active_skill_ids"] == [installed.skill_id]
    runtime.validate()


def test_runtime_normalizes_json_style_thread_specs_and_rejects_bare_skill_strings(tmp_path):
    runtime = _runtime(tmp_path)

    skill_root = tmp_path / "source-skill"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: json-skill",
                "name: JSON Skill",
                "scope: workspace",
                "---",
                "JSON skill body.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    installed = runtime.skill_catalog.install(skill_root, scope="repo")

    thread = runtime.create_thread(
        {
            "title": "JSON thread",
            "mode": "code",
            "task_prompt": "Normalize JSON-style thread specs.",
            "host": "cli",
            "workspace": tmp_path,
            "attached_skills": [installed.skill_id],
        }
    )

    assert thread.workspace == str(tmp_path)
    assert thread.skills[0].skill_id == installed.skill_id

    with pytest.raises(InterfaceFrameworkError, match="skill_ids must be a sequence"):
        runtime.attach_skills(thread, installed.skill_id)


def test_runtime_archives_skills_and_reports_provenance_summary(tmp_path):
    runtime = _runtime(tmp_path)

    skill_root = tmp_path / "archive-source"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "skill_id: archive-skill",
                "name: Archive Skill",
                "scope: repo",
                "---",
                "Archive from runtime.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    installed = runtime.skill_catalog.install(skill_root, scope="repo")
    runtime.activate_skill(installed.skill_id)
    archived = runtime.archive_skill(installed.skill_id, reason="superseded")
    summary = runtime.skill_provenance_summary()

    assert archived.skill_id == installed.skill_id
    assert archived.metadata["archived"] is True
    assert archived.metadata["archived_reason"] == "superseded"
    assert summary["archived_count"] == 1
    assert summary["active_count"] == 0


def test_runtime_rejects_unknown_modes_loudly(tmp_path):
    runtime = _runtime(tmp_path)

    with pytest.raises(InterfaceFrameworkError, match="unknown mode"):
        runtime.framework.select_mode("not-a-mode")
