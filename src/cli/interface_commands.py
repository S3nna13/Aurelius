"""CLI command group for the Aurelius interface shell.

The handlers are intentionally thin: they construct or reuse a
``AureliusShell`` instance, delegate real work to the shell/framework,
and print JSON or plain-text summaries suitable for terminal use.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TextIO

from src.ui.aurelius_shell import (
    AureliusShell,
    AureliusShellError,
)
from src.model.interface_framework import InterfaceFrameworkError

__all__ = [
    "INTERFACE_COMMAND_HANDLERS",
    "build_interface_parser",
    "dispatch_interface_command",
    "handle_interface_approval_request",
    "handle_interface_checkpoint_resume",
    "handle_interface_checkpoint_save",
    "handle_interface_describe",
    "handle_interface_job_cancel",
    "handle_interface_job_launch",
    "handle_interface_job_status",
    "handle_interface_mode_list",
    "handle_interface_mode_set",
    "handle_interface_shell_status",
    "handle_interface_skill_attach",
    "handle_interface_skill_list",
    "handle_interface_thread_create",
    "handle_interface_thread_status",
    "handle_interface_workflow_run",
    "handle_interface_workstream_create",
    "handle_interface_workstream_list",
    "handle_interface_workstream_status",
]


def _json_write(out_stream: TextIO, payload: Any) -> None:
    out_stream.write(json.dumps(payload, sort_keys=True))
    out_stream.write("\n")


def _build_shell(args: argparse.Namespace, shell: AureliusShell | None = None) -> AureliusShell:
    if shell is not None:
        return shell
    repo_root = getattr(args, "repo_root", None)
    variant_id = getattr(args, "variant_id", None)
    return AureliusShell.from_repo_root(root_dir=repo_root, variant_id=variant_id)


def _resolve_thread(shell: AureliusShell, thread_id: str | None) -> str:
    if thread_id:
        return thread_id
    if shell.active_thread_id is not None:
        return shell.active_thread_id
    raise AureliusShellError("no thread id supplied and no active thread is selected")


def _resolve_workstream(shell: AureliusShell, workstream_id: str | None) -> str:
    if workstream_id:
        return workstream_id
    if shell.active_workstream_id is not None:
        return shell.active_workstream_id
    raise AureliusShellError("no workstream id supplied and no active workstream is selected")


def _load_workflow_argument(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    if getattr(args, "workflow_json", None) is not None:
        try:
            payload = json.loads(args.workflow_json)
        except json.JSONDecodeError as exc:
            raise AureliusShellError(f"failed to parse workflow JSON: {exc.msg}") from exc
        if not isinstance(payload, (dict, list)):
            raise AureliusShellError("workflow JSON must decode to an object or array")
        return payload
    workflow_file = getattr(args, "workflow_file", None)
    if workflow_file is None:
        raise AureliusShellError("workflow run requires --workflow-json or --workflow-file")
    try:
        text = Path(workflow_file).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise AureliusShellError(f"failed to read workflow file {workflow_file!r}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AureliusShellError(f"failed to parse workflow file as JSON: {exc.msg}") from exc
    if not isinstance(payload, (dict, list)):
        raise AureliusShellError("workflow file must decode to an object or array")
    return payload


def handle_interface_describe(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    _json_write(stream, active_shell.describe())
    return 0


def handle_interface_mode_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    _json_write(
        stream,
        {
            "current_mode": active_shell.current_mode,
            "modes": [asdict(active_shell.framework.select_mode(name)) for name in active_shell.list_modes()],
        },
    )
    return 0


def handle_interface_mode_set(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    policy = active_shell.set_mode(args.mode_name)
    _json_write(
        stream,
        {
            "current_mode": active_shell.current_mode,
            "policy": asdict(policy),
        },
    )
    return 0


def handle_interface_shell_status(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    stream.write(active_shell.render_status())
    stream.write("\n")
    return 0


def handle_interface_workstream_create(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    workstream = active_shell.create_workstream(
        args.name,
        workspace=args.workspace,
    )
    _json_write(stream, {"workstream": asdict(workstream)})
    return 0


def handle_interface_workstream_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    _json_write(
        stream,
        {
            "workstreams": [asdict(workstream) for workstream in active_shell.workstreams],
        },
    )
    return 0


def handle_interface_workstream_status(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    workstream_id = _resolve_workstream(active_shell, getattr(args, "workstream_id", None))
    stream.write(active_shell.render_workstream_status(workstream_id))
    stream.write("\n")
    return 0


def handle_interface_thread_create(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    if getattr(args, "workstream_name", None) and getattr(args, "workstream_id", None):
        raise AureliusShellError("pass either --workstream-name or --workstream-id, not both")
    workstream_id = getattr(args, "workstream_id", None)
    if getattr(args, "workstream_name", None):
        workstream = active_shell.create_workstream(args.workstream_name, workspace=args.workspace)
        workstream_id = workstream.workstream_id
    thread = active_shell.create_thread(
        title=args.title,
        task_prompt=args.prompt,
        mode=getattr(args, "mode", None),
        host=getattr(args, "host", "cli"),
        workspace=getattr(args, "workspace", None),
        workstream_id=workstream_id,
        channel=getattr(args, "channel", None),
        skills=getattr(args, "skills", None) or (),
        repo_instructions=getattr(args, "repo_instructions", None),
        workspace_instructions=getattr(args, "workspace_instructions", None),
    )
    _json_write(
        stream,
        {
            "thread": asdict(thread),
            "active_thread_id": active_shell.active_thread_id,
            "active_workstream_id": active_shell.active_workstream_id,
        },
    )
    return 0


def handle_interface_thread_status(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    thread_id = _resolve_thread(active_shell, getattr(args, "thread_id", None))
    stream.write(active_shell.render_thread_status(thread_id))
    stream.write("\n")
    return 0


def handle_interface_skill_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    roots = getattr(args, "roots", None)
    skills = active_shell.discover_skills(roots=roots)
    _json_write(
        stream,
        {
            "count": len(skills),
            "skills": [asdict(skill) for skill in skills],
        },
    )
    return 0


def handle_interface_skill_attach(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    thread_id = _resolve_thread(active_shell, getattr(args, "thread_id", None))
    roots = getattr(args, "roots", None)
    updated = active_shell.attach_skill_ids(thread_id, list(args.skills), roots=roots)
    _json_write(stream, {"thread": asdict(updated)})
    return 0


def handle_interface_approval_request(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    thread_id = _resolve_thread(active_shell, getattr(args, "thread_id", None))
    approval = active_shell.request_approval(
        thread_id,
        category=args.category,
        action_summary=args.action_summary,
        affected_resources=tuple(args.resource or ()),
        reason=args.reason,
        reversible=bool(args.reversible),
        minimum_scope=args.minimum_scope,
    )
    _json_write(stream, {"approval": asdict(approval)})
    return 0


def handle_interface_checkpoint_save(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    thread_id = _resolve_thread(active_shell, getattr(args, "thread_id", None))
    checkpoint = active_shell.checkpoint_thread(
        thread_id,
        memory_summary=args.memory_summary,
        last_model_response=getattr(args, "last_model_response", None),
    )
    _json_write(stream, {"checkpoint": asdict(checkpoint)})
    return 0


def handle_interface_checkpoint_resume(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    resumed = active_shell.resume_thread(args.checkpoint_id)
    _json_write(stream, {"thread": asdict(resumed)})
    return 0


def handle_interface_job_launch(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    thread_id = _resolve_thread(active_shell, getattr(args, "thread_id", None))
    job = active_shell.launch_background_job(
        thread_id,
        description=args.description,
    )
    _json_write(stream, {"job": asdict(job)})
    return 0


def handle_interface_job_status(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    stream.write(active_shell.render_job_status(args.job_id))
    stream.write("\n")
    return 0


def handle_interface_job_cancel(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    canceled = active_shell.cancel_background_job(args.job_id)
    _json_write(stream, {"job": asdict(canceled)})
    return 0


def handle_interface_workflow_run(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_shell = _build_shell(args, shell)
    thread_id = _resolve_thread(active_shell, getattr(args, "thread_id", None))
    workflow = _load_workflow_argument(args)
    run = active_shell.execute_workflow(thread_id, workflow)
    _json_write(stream, {"workflow_run": asdict(run)})
    return 0


INTERFACE_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace, TextIO, AureliusShell | None], int]] = {
    "describe": handle_interface_describe,
    "mode_list": handle_interface_mode_list,
    "mode_set": handle_interface_mode_set,
    "shell_status": handle_interface_shell_status,
    "workstream_create": handle_interface_workstream_create,
    "workstream_list": handle_interface_workstream_list,
    "workstream_status": handle_interface_workstream_status,
    "thread_create": handle_interface_thread_create,
    "thread_status": handle_interface_thread_status,
    "skill_list": handle_interface_skill_list,
    "skill_attach": handle_interface_skill_attach,
    "approval_request": handle_interface_approval_request,
    "checkpoint_save": handle_interface_checkpoint_save,
    "checkpoint_resume": handle_interface_checkpoint_resume,
    "job_launch": handle_interface_job_launch,
    "job_status": handle_interface_job_status,
    "job_cancel": handle_interface_job_cancel,
    "workflow_run": handle_interface_workflow_run,
}


def build_interface_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Attach the ``interface`` command group to ``subparsers``."""
    interface_parser = subparsers.add_parser(
        "interface",
        help="inspect and operate the Aurelius interface shell",
        description="Aurelius-native interface shell and workflow command group.",
    )
    interface_parser.add_argument(
        "--repo-root",
        default=None,
        help="repository root used to load the canonical interface contract",
    )
    interface_parser.add_argument(
        "--variant-id",
        default=None,
        help="optional model variant identifier used for framework context",
    )
    interface_sub = interface_parser.add_subparsers(
        dest="interface_command",
        metavar="interface_command",
    )

    describe_p = interface_sub.add_parser("describe", help="describe the active shell state")
    describe_p.set_defaults(interface_handler=handle_interface_describe)

    mode_p = interface_sub.add_parser("mode", help="list and select Aurelius modes")
    mode_sub = mode_p.add_subparsers(dest="mode_command", metavar="mode_command")
    mode_list_p = mode_sub.add_parser("list", help="list available modes")
    mode_list_p.set_defaults(interface_handler=handle_interface_mode_list)
    mode_set_p = mode_sub.add_parser("set", help="select the active mode")
    mode_set_p.add_argument("mode_name", help="mode name from the canonical contract")
    mode_set_p.set_defaults(interface_handler=handle_interface_mode_set)

    shell_p = interface_sub.add_parser("shell", help="render the shell status view")
    shell_sub = shell_p.add_subparsers(dest="shell_command", metavar="shell_command")
    shell_status_p = shell_sub.add_parser("status", help="render the current shell status")
    shell_status_p.set_defaults(interface_handler=handle_interface_shell_status)

    workstream_p = interface_sub.add_parser("workstream", help="manage named workstreams")
    workstream_sub = workstream_p.add_subparsers(
        dest="workstream_command",
        metavar="workstream_command",
    )
    workstream_create_p = workstream_sub.add_parser("create", help="create a named workstream")
    workstream_create_p.add_argument("--name", required=True, help="workstream name")
    workstream_create_p.add_argument("--workspace", default=None, help="workspace path")
    workstream_create_p.set_defaults(interface_handler=handle_interface_workstream_create)
    workstream_list_p = workstream_sub.add_parser("list", help="list workstreams")
    workstream_list_p.set_defaults(interface_handler=handle_interface_workstream_list)
    workstream_status_p = workstream_sub.add_parser("status", help="show a workstream")
    workstream_status_p.add_argument("--workstream-id", default=None, help="workstream id")
    workstream_status_p.set_defaults(interface_handler=handle_interface_workstream_status)

    thread_p = interface_sub.add_parser("thread", help="create and inspect threads")
    thread_sub = thread_p.add_subparsers(dest="thread_command", metavar="thread_command")
    thread_create_p = thread_sub.add_parser("create", help="create a new thread")
    thread_create_p.add_argument("--title", required=True, help="thread title")
    thread_create_p.add_argument("--prompt", required=True, help="task prompt")
    thread_create_p.add_argument("--mode", default=None, help="mode name")
    thread_create_p.add_argument("--host", default="cli", help="host adapter name")
    thread_create_p.add_argument("--workspace", default=None, help="workspace path")
    thread_create_p.add_argument("--workstream-id", default=None, help="attach to an existing workstream")
    thread_create_p.add_argument("--workstream-name", default=None, help="create and attach a new workstream")
    thread_create_p.add_argument("--channel", default=None, help="channel identifier")
    thread_create_p.add_argument("--skill", dest="skills", action="append", default=[], help="skill id to attach")
    thread_create_p.add_argument("--repo-instructions", default=None, help="repo-local instructions text")
    thread_create_p.add_argument("--workspace-instructions", default=None, help="workspace instructions text")
    thread_create_p.set_defaults(interface_handler=handle_interface_thread_create)
    thread_status_p = thread_sub.add_parser("status", help="render thread status")
    thread_status_p.add_argument("--thread-id", default=None, help="thread id")
    thread_status_p.set_defaults(interface_handler=handle_interface_thread_status)

    skill_p = interface_sub.add_parser("skill", help="discover and attach skill bundles")
    skill_sub = skill_p.add_subparsers(dest="skill_command", metavar="skill_command")
    skill_list_p = skill_sub.add_parser("list", help="list discovered skills")
    skill_list_p.add_argument(
        "--root",
        dest="roots",
        action="append",
        default=None,
        help="skill root to scan (may be supplied multiple times)",
    )
    skill_list_p.set_defaults(interface_handler=handle_interface_skill_list)
    skill_attach_p = skill_sub.add_parser("attach", help="attach skills to a thread")
    skill_attach_p.add_argument("--thread-id", default=None, help="thread id")
    skill_attach_p.add_argument("--skill", dest="skills", action="append", required=True, help="skill id")
    skill_attach_p.add_argument(
        "--root",
        dest="roots",
        action="append",
        default=None,
        help="skill root to scan for provenance",
    )
    skill_attach_p.set_defaults(interface_handler=handle_interface_skill_attach)

    approval_p = interface_sub.add_parser("approval", help="create approval records")
    approval_sub = approval_p.add_subparsers(dest="approval_command", metavar="approval_command")
    approval_req_p = approval_sub.add_parser("request", help="request explicit approval")
    approval_req_p.add_argument("--thread-id", default=None, help="thread id")
    approval_req_p.add_argument("--category", required=True, help="approval category")
    approval_req_p.add_argument("--action-summary", required=True, help="summary of the action")
    approval_req_p.add_argument("--reason", required=True, help="why approval is needed")
    approval_req_p.add_argument("--minimum-scope", required=True, help="minimum approval scope")
    approval_req_p.add_argument("--resource", action="append", default=[], help="affected resource path")
    approval_req_p.add_argument("--reversible", action="store_true", help="mark the action reversible")
    approval_req_p.set_defaults(interface_handler=handle_interface_approval_request)

    checkpoint_p = interface_sub.add_parser("checkpoint", help="checkpoint and resume threads")
    checkpoint_sub = checkpoint_p.add_subparsers(dest="checkpoint_command", metavar="checkpoint_command")
    checkpoint_save_p = checkpoint_sub.add_parser("save", help="save a checkpoint")
    checkpoint_save_p.add_argument("--thread-id", default=None, help="thread id")
    checkpoint_save_p.add_argument("--memory-summary", required=True, help="checkpoint memory summary")
    checkpoint_save_p.add_argument("--last-model-response", default=None, help="most recent model response")
    checkpoint_save_p.set_defaults(interface_handler=handle_interface_checkpoint_save)
    checkpoint_resume_p = checkpoint_sub.add_parser("resume", help="resume a checkpoint")
    checkpoint_resume_p.add_argument("--checkpoint-id", required=True, help="checkpoint id")
    checkpoint_resume_p.set_defaults(interface_handler=handle_interface_checkpoint_resume)

    job_p = interface_sub.add_parser("job", help="launch and inspect background jobs")
    job_sub = job_p.add_subparsers(dest="job_command", metavar="job_command")
    job_launch_p = job_sub.add_parser("launch", help="launch a background job")
    job_launch_p.add_argument("--thread-id", default=None, help="thread id")
    job_launch_p.add_argument("--description", required=True, help="job description")
    job_launch_p.set_defaults(interface_handler=handle_interface_job_launch)
    job_status_p = job_sub.add_parser("status", help="show a job status")
    job_status_p.add_argument("--job-id", required=True, help="job id")
    job_status_p.set_defaults(interface_handler=handle_interface_job_status)
    job_cancel_p = job_sub.add_parser("cancel", help="cancel a job")
    job_cancel_p.add_argument("--job-id", required=True, help="job id")
    job_cancel_p.set_defaults(interface_handler=handle_interface_job_cancel)

    workflow_p = interface_sub.add_parser("workflow", help="execute JSON workflow specs")
    workflow_sub = workflow_p.add_subparsers(dest="workflow_command", metavar="workflow_command")
    workflow_run_p = workflow_sub.add_parser("run", help="run a workflow spec")
    workflow_run_p.add_argument("--thread-id", default=None, help="thread id")
    workflow_run_p.add_argument("--workflow-json", default=None, help="workflow JSON string")
    workflow_run_p.add_argument("--workflow-file", default=None, help="path to a workflow JSON file")
    workflow_run_p.set_defaults(interface_handler=handle_interface_workflow_run)

    return interface_parser


def dispatch_interface_command(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    shell: AureliusShell | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    handler = getattr(args, "interface_handler", None)
    if handler is None:
        _json_write(stream, {"error": "missing interface command"})
        return 1
    active_shell = _build_shell(args, shell)
    try:
        return handler(args, stream, active_shell)
    except (AureliusShellError, InterfaceFrameworkError, ValueError) as exc:
        _json_write(stream, {"error": str(exc)})
        return 1
