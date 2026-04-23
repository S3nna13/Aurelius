"""Integration tests for the Aurelius terminal shell surface."""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_src_ui_package_exports_shell_surface():
    from src.ui import (
        AureliusShell,
        MessageEnvelope,
        SkillRecord,
        Workstream,
        WorkflowRun,
        WorkflowStep,
    )

    assert AureliusShell.__name__ == "AureliusShell"
    assert MessageEnvelope.__name__ == "MessageEnvelope"
    assert SkillRecord.__name__ == "SkillRecord"
    assert Workstream.__name__ == "Workstream"
    assert WorkflowRun.__name__ == "WorkflowRun"
    assert WorkflowStep.__name__ == "WorkflowStep"


def test_shell_status_snapshot_and_restore_are_json_safe():
    from src.ui import AureliusShell

    shell = AureliusShell.from_repo_root(root_dir=_repo_root())
    workstream = shell.create_workstream("Integration workstream")
    thread = shell.create_thread(
        title="Integration thread",
        task_prompt="Cover the shell end to end.",
        mode="code",
        workstream_id=workstream.workstream_id,
    )
    job = shell.launch_background_job(thread, description="background validation")
    shell.route_channel(channel="integration", content="hello", thread=thread)

    snapshot = shell.snapshot()
    json.dumps(snapshot)
    restored = AureliusShell.restore_snapshot(snapshot, root_dir=_repo_root())

    assert "Integration thread" in restored.render_status()
    assert "Integration workstream" in restored.render_status()
    assert job.job_id in restored.render_status()
    assert thread.thread_id in restored.render_thread_status(thread.thread_id)
    assert "Backends:" in restored.render_status()


def test_shell_backend_commands_are_reachable_from_execute_command():
    from src.ui import AureliusShell

    shell = AureliusShell.from_repo_root(root_dir=_repo_root())
    backend_list = json.loads(shell.execute_command("backend list"))

    assert backend_list["count"] >= 1
    assert "pytorch" in backend_list["names"]

    backend_show = json.loads(shell.execute_command("backend show pytorch"))
    assert backend_show["backend"]["backend_name"] == "pytorch"

    engine_list = json.loads(shell.execute_command("backend engine list"))
    assert engine_list["engine_surface"]["count"] >= 3
    assert "gguf" in engine_list["engine_surface"]["names"]

    engine_show = json.loads(shell.execute_command("backend engine show vllm"))
    assert engine_show["engine"]["backend_name"] == "vllm"


def test_shell_capability_and_journal_commands_are_reachable_from_execute_command():
    from src.ui import AureliusShell

    shell = AureliusShell.from_repo_root(root_dir=_repo_root())
    capability = json.loads(shell.execute_command("capability summary"))
    schema = json.loads(shell.execute_command("capability schema"))
    journal_branch = json.loads(shell.execute_command("journal branch main"))
    surface = json.loads(shell.execute_command("surface summary"))
    surface_schema = json.loads(shell.execute_command("surface schema"))

    assert capability["capability"]["framework"]["title"] == "Aurelius Canonical Interface Contract"
    assert "skills" in capability["capability"]
    assert schema["schema"]["schema_version"] == "1.0"
    assert journal_branch["journal"]["branch_id"] == "main"
    assert surface["surface"]["engine_adapters"]["count"] >= 3
    assert surface_schema["schema"]["schema_name"] == "aurelius.interface.surface-catalog"
