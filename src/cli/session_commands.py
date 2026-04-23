"""CLI command group for persistent Aurelius sessions and workstreams.

This stays local-first and thin: the command handlers delegate to
``AureliusInterfaceRuntime`` / ``SessionManager`` rather than duplicating
session persistence logic in the CLI layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TextIO

from src.agent.interface_runtime import AureliusInterfaceRuntime
from src.agent.session_manager import SessionManager
from src.model.interface_framework import AureliusInterfaceFramework
from src.model.interface_framework import InterfaceFrameworkError

__all__ = [
    "SESSION_COMMAND_HANDLERS",
    "build_session_parser",
    "dispatch_session_command",
    "handle_session_export",
    "handle_session_import",
    "handle_session_list",
    "handle_session_show",
    "handle_session_journal_append",
    "handle_session_journal_branch",
    "handle_session_journal_compact",
    "handle_session_journal_list",
    "handle_session_journal_show",
    "handle_session_thread_list",
    "handle_session_thread_show",
    "handle_session_workstream_list",
    "handle_session_workstream_show",
]


_SESSION_TOP_LEVEL_COMMANDS = ("list", "show", "export", "import", "journal", "thread", "workstream")
_SESSION_THREAD_COMMANDS = ("list", "show")
_SESSION_JOURNAL_COMMANDS = ("list", "show", "append", "branch", "compact")
_SESSION_WORKSTREAM_COMMANDS = ("list", "show")
_VALID_JOURNAL_POLICIES = ("oldest_first", "middle_biased", "tool_output_aggregated")


def _json_write(out_stream: TextIO, payload: Any) -> None:
    out_stream.write(json.dumps(payload, sort_keys=True))
    out_stream.write("\n")


def _error(out_stream: TextIO, message: str, **extra: Any) -> int:
    payload = {"error": message}
    payload.update(extra)
    _json_write(out_stream, payload)
    return 1


def _parse_json_payload(value: str | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InterfaceFrameworkError(f"{field_name} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise InterfaceFrameworkError(f"{field_name} must decode to a JSON object")
    return payload


def _build_runtime(args: argparse.Namespace, runtime: AureliusInterfaceRuntime | None = None) -> AureliusInterfaceRuntime:
    if runtime is not None:
        return runtime
    repo_root = getattr(args, "repo_root", None)
    variant_id = getattr(args, "variant_id", None)
    state_dir = getattr(args, "state_dir", None)
    if state_dir is None:
        return AureliusInterfaceRuntime.from_repo_root(root_dir=repo_root, variant_id=variant_id)
    framework = AureliusInterfaceFramework.from_repo_root(root_dir=repo_root, variant_id=variant_id)
    resolved_root = Path(repo_root).expanduser().resolve() if repo_root is not None else framework.paths.repo_root
    return AureliusInterfaceRuntime(
        framework,
        root_dir=resolved_root,
        variant_id=variant_id,
        session_manager=SessionManager(state_dir=state_dir, root_dir=resolved_root),
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        default=None,
        help="repository root used to locate the canonical interface contract",
    )
    parser.add_argument(
        "--variant-id",
        default=None,
        help="optional model variant identifier used for framework context",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="directory used to store persistent session state",
    )


def handle_session_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    active_runtime = _build_runtime(args, runtime)
    sessions = active_runtime.session_manager.list_sessions()
    _json_write(
        stream,
        {
            "count": len(sessions),
            "sessions": [active_runtime.session_manager.snapshot(session) for session in sessions],
        },
    )
    return 0


def handle_session_show(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    if not session_id:
        return _error(stream, "session show requires a session_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        payload = active_runtime.session_status(session_id)
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(stream, payload)
    return 0


def handle_session_export(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    if not session_id:
        return _error(stream, "session export requires a session_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        export = active_runtime.export_session(session_id)
        path = getattr(args, "path", None)
        if path:
            target = active_runtime.export_session_to_path(session_id, path)
            _json_write(
                stream,
                {
                    "session_id": session_id,
                    "path": str(target),
                    "export": export,
                },
            )
            return 0
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session_id,
            "export": export,
        },
    )
    return 0


def handle_session_import(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    path = getattr(args, "path", None)
    if not path:
        return _error(stream, "session import requires a path")
    active_runtime = _build_runtime(args, runtime)
    try:
        session = active_runtime.import_session(path, replace=getattr(args, "replace", False))
        journal = active_runtime.session_manager.journal_summary(session.session_id)
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session.session_id,
            "session": active_runtime.session_manager.snapshot(session),
            "journal": journal,
        },
    )
    return 0


def handle_session_journal_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    if not session_id:
        return _error(stream, "session journal list requires a session_id")
    branch_id = getattr(args, "branch_id", None)
    active_runtime = _build_runtime(args, runtime)
    try:
        journal = active_runtime.session_manager.get_journal(session_id)
        entries = active_runtime.session_manager.list_journal_entries(
            session_id,
            branch_id=branch_id,
        )
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session_id,
            "branch_id": branch_id,
            "count": len(entries),
            "journal": journal.describe() if journal is not None else None,
            "entries": [asdict(entry) for entry in entries],
        },
    )
    return 0


def handle_session_journal_show(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    entry_id = getattr(args, "entry_id", None)
    if not session_id:
        return _error(stream, "session journal show requires a session_id")
    if not entry_id:
        return _error(stream, "session journal show requires an entry_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        journal = active_runtime.session_manager.get_journal(session_id)
        entry = active_runtime.session_manager.get_journal_entry(session_id, entry_id)
        if entry is None:
            raise InterfaceFrameworkError(f"unknown journal entry: {entry_id!r}")
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session_id,
            "journal": journal.describe() if journal is not None else None,
            "entry": asdict(entry),
        },
    )
    return 0


def handle_session_journal_append(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    kind = getattr(args, "kind", None)
    summary = getattr(args, "summary", None)
    if not session_id:
        return _error(stream, "session journal append requires a session_id")
    if not kind:
        return _error(stream, "session journal append requires a kind")
    if not summary:
        return _error(stream, "session journal append requires a summary")
    active_runtime = _build_runtime(args, runtime)
    try:
        payload = _parse_json_payload(getattr(args, "payload_json", None), "payload_json")
        metadata = _parse_json_payload(getattr(args, "metadata_json", None), "metadata_json")
        tags = tuple(getattr(args, "tag", ()) or ())
        entry = active_runtime.session_manager.append_journal_entry(
            session_id,
            kind=kind,
            summary=summary,
            branch_id=getattr(args, "branch_id", "main") or "main",
            thread_id=getattr(args, "thread_id", None),
            workstream_id=getattr(args, "workstream_id", None),
            severity=getattr(args, "severity", "info"),
            payload=payload,
            metadata=metadata,
            tags=tags,
        )
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session_id,
            "entry": asdict(entry),
        },
    )
    return 0


def handle_session_journal_branch(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    name = getattr(args, "name", None)
    if not session_id:
        return _error(stream, "session journal branch requires a session_id")
    if not name:
        return _error(stream, "session journal branch requires a name")
    active_runtime = _build_runtime(args, runtime)
    try:
        result = active_runtime.session_manager.branch_journal(
            session_id,
            name,
            from_entry_id=getattr(args, "from_entry_id", None),
            source_branch_id=getattr(args, "source_branch_id", "main") or "main",
            metadata={"reason": getattr(args, "reason", None)} if getattr(args, "reason", None) else None,
        )
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(stream, {"session_id": session_id, **result})
    return 0


def handle_session_journal_compact(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    if not session_id:
        return _error(stream, "session journal compact requires a session_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        result = active_runtime.session_manager.compact_journal(
            session_id,
            branch_id=getattr(args, "branch_id", "main") or "main",
            keep_last_n=getattr(args, "keep_last_n", 4),
            policy=getattr(args, "policy", "oldest_first"),
            metadata={"reason": getattr(args, "reason", None)} if getattr(args, "reason", None) else None,
        )
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(stream, {"session_id": session_id, **result})
    return 0


def handle_session_thread_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    if not session_id:
        return _error(stream, "session thread list requires a session_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        active_runtime.session_status(session_id)
        threads = active_runtime.session_manager.list_threads(session_id)
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session_id,
            "count": len(threads),
            "threads": [asdict(thread) for thread in threads],
        },
    )
    return 0


def handle_session_thread_show(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    thread_id = getattr(args, "thread_id", None)
    if not session_id:
        return _error(stream, "session thread show requires a session_id")
    if not thread_id:
        return _error(stream, "session thread show requires a thread_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        payload = active_runtime.thread_status(session_id, thread_id)
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(stream, payload)
    return 0


def handle_session_workstream_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    if not session_id:
        return _error(stream, "session workstream list requires a session_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        workstreams = active_runtime.session_manager.list_workstreams(session_id)
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session_id,
            "count": len(workstreams),
            "workstreams": [asdict(workstream) for workstream in workstreams],
        },
    )
    return 0


def handle_session_workstream_show(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    session_id = getattr(args, "session_id", None)
    workstream_id = getattr(args, "workstream_id", None)
    if not session_id:
        return _error(stream, "session workstream show requires a session_id")
    if not workstream_id:
        return _error(stream, "session workstream show requires a workstream_id")
    active_runtime = _build_runtime(args, runtime)
    try:
        active_runtime.session_status(session_id)
        workstream = active_runtime.session_manager.get_workstream(session_id, workstream_id)
        if workstream is None:
            raise InterfaceFrameworkError(f"unknown workstream: {workstream_id!r}")
    except InterfaceFrameworkError as exc:
        return _error(stream, str(exc))
    _json_write(
        stream,
        {
            "session_id": session_id,
            "workstream": asdict(workstream),
        },
    )
    return 0


SESSION_COMMAND_HANDLERS: dict[
    str, Callable[[argparse.Namespace, TextIO, AureliusInterfaceRuntime | None], int]
] = {
    "list": handle_session_list,
    "show": handle_session_show,
    "export": handle_session_export,
    "import": handle_session_import,
    "workstream_list": handle_session_workstream_list,
}


def build_session_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Attach the ``session`` command group to an existing subparser tree."""
    session_parser = subparsers.add_parser(
        "session",
        help="inspect persistent Aurelius sessions and workstreams",
        description="Local-first session and workstream inspection commands.",
    )
    _add_runtime_args(session_parser)
    session_sub = session_parser.add_subparsers(
        dest="session_command",
        metavar="session_command",
    )

    list_p = session_sub.add_parser("list", help="list persisted sessions")
    _add_runtime_args(list_p)

    show_p = session_sub.add_parser("show", help="show a session summary")
    _add_runtime_args(show_p)
    show_p.add_argument("session_id", help="session id")

    export_p = session_sub.add_parser("export", help="export a portable session bundle")
    _add_runtime_args(export_p)
    export_p.add_argument("session_id", help="session id")
    export_p.add_argument("--path", default=None, help="optional export path")

    import_p = session_sub.add_parser("import", help="import a portable session bundle")
    _add_runtime_args(import_p)
    import_p.add_argument("path", help="session export path")
    import_p.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing session if the export session id already exists",
    )

    journal_p = session_sub.add_parser(
        "journal",
        help="inspect and compact the session journal",
    )
    _add_runtime_args(journal_p)
    journal_sub = journal_p.add_subparsers(
        dest="journal_command",
        metavar="journal_command",
    )

    journal_list_p = journal_sub.add_parser("list", help="list journal entries")
    _add_runtime_args(journal_list_p)
    journal_list_p.add_argument("session_id", help="session id")
    journal_list_p.add_argument("--branch-id", default=None, help="optional branch id")

    journal_show_p = journal_sub.add_parser("show", help="show a journal entry")
    _add_runtime_args(journal_show_p)
    journal_show_p.add_argument("session_id", help="session id")
    journal_show_p.add_argument("entry_id", help="journal entry id")

    journal_append_p = journal_sub.add_parser("append", help="append a journal entry")
    _add_runtime_args(journal_append_p)
    journal_append_p.add_argument("session_id", help="session id")
    journal_append_p.add_argument("kind", help="journal entry kind")
    journal_append_p.add_argument("summary", help="human-readable summary")
    journal_append_p.add_argument("--branch-id", default="main", help="branch id")
    journal_append_p.add_argument("--thread-id", default=None, help="optional thread id")
    journal_append_p.add_argument("--workstream-id", default=None, help="optional workstream id")
    journal_append_p.add_argument(
        "--severity",
        default="info",
        choices=("info", "notice", "warning", "error", "critical"),
        help="journal severity",
    )
    journal_append_p.add_argument("--payload-json", default=None, help="JSON object payload")
    journal_append_p.add_argument("--metadata-json", default=None, help="JSON object metadata")
    journal_append_p.add_argument(
        "--tag",
        action="append",
        default=[],
        help="journal tag (repeatable)",
    )

    journal_branch_p = journal_sub.add_parser("branch", help="create a journal branch")
    _add_runtime_args(journal_branch_p)
    journal_branch_p.add_argument("session_id", help="session id")
    journal_branch_p.add_argument("name", help="branch name")
    journal_branch_p.add_argument("--from-entry-id", default=None, help="branch base entry id")
    journal_branch_p.add_argument(
        "--source-branch-id",
        default="main",
        help="source branch id",
    )
    journal_branch_p.add_argument("--reason", default=None, help="optional reason")

    journal_compact_p = journal_sub.add_parser("compact", help="compact a journal branch")
    _add_runtime_args(journal_compact_p)
    journal_compact_p.add_argument("session_id", help="session id")
    journal_compact_p.add_argument("--branch-id", default="main", help="branch id")
    journal_compact_p.add_argument(
        "--keep-last-n",
        type=int,
        default=4,
        help="number of tail entries to keep verbatim",
    )
    journal_compact_p.add_argument(
        "--policy",
        default="oldest_first",
        choices=_VALID_JOURNAL_POLICIES,
        help="compaction policy",
    )
    journal_compact_p.add_argument("--reason", default=None, help="optional reason")

    thread_p = session_sub.add_parser(
        "thread",
        help="inspect threads inside a session",
    )
    thread_sub = thread_p.add_subparsers(
        dest="thread_command",
        metavar="thread_command",
    )
    thread_list_p = thread_sub.add_parser(
        "list",
        help="list threads in a session",
    )
    _add_runtime_args(thread_list_p)
    thread_list_p.add_argument("session_id", help="session id")
    thread_show_p = thread_sub.add_parser(
        "show",
        help="show a thread inside a session",
    )
    _add_runtime_args(thread_show_p)
    thread_show_p.add_argument("session_id", help="session id")
    thread_show_p.add_argument("thread_id", help="thread id")

    workstream_p = session_sub.add_parser(
        "workstream",
        help="inspect workstreams inside a session",
    )
    workstream_sub = workstream_p.add_subparsers(
        dest="workstream_command",
        metavar="workstream_command",
    )
    workstream_list_p = workstream_sub.add_parser(
        "list",
        help="list named workstreams in a session",
    )
    _add_runtime_args(workstream_list_p)
    workstream_list_p.add_argument("session_id", help="session id")
    workstream_show_p = workstream_sub.add_parser(
        "show",
        help="show a workstream inside a session",
    )
    _add_runtime_args(workstream_show_p)
    workstream_show_p.add_argument("session_id", help="session id")
    workstream_show_p.add_argument("workstream_id", help="workstream id")

    return session_parser


def dispatch_session_command(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
    runtime: AureliusInterfaceRuntime | None = None,
) -> int:
    """Route a parsed session command to its handler."""
    stream = out_stream if out_stream is not None else sys.stdout
    name = getattr(args, "session_command", None)
    if name is None:
        return _error(
            stream,
            "missing session command",
            known_subcommands=list(_SESSION_TOP_LEVEL_COMMANDS),
        )
    if name == "thread":
        thread_command = getattr(args, "thread_command", None)
        if thread_command == "list":
            return handle_session_thread_list(args, stream, runtime)
        if thread_command == "show":
            return handle_session_thread_show(args, stream, runtime)
        return _error(
            stream,
            f"unknown session thread command: {thread_command!r}",
            known_subcommands=list(_SESSION_THREAD_COMMANDS),
        )
    if name == "journal":
        journal_command = getattr(args, "journal_command", None)
        if journal_command == "list":
            return handle_session_journal_list(args, stream, runtime)
        if journal_command == "show":
            return handle_session_journal_show(args, stream, runtime)
        if journal_command == "append":
            return handle_session_journal_append(args, stream, runtime)
        if journal_command == "branch":
            return handle_session_journal_branch(args, stream, runtime)
        if journal_command == "compact":
            return handle_session_journal_compact(args, stream, runtime)
        return _error(
            stream,
            f"unknown session journal command: {journal_command!r}",
            known_subcommands=list(_SESSION_JOURNAL_COMMANDS),
        )
    if name == "export":
        return handle_session_export(args, stream, runtime)
    if name == "import":
        return handle_session_import(args, stream, runtime)
    if name == "workstream":
        workstream_command = getattr(args, "workstream_command", None)
        if workstream_command == "list":
            return handle_session_workstream_list(args, stream, runtime)
        if workstream_command == "show":
            return handle_session_workstream_show(args, stream, runtime)
        return _error(
            stream,
            f"unknown session workstream command: {workstream_command!r}",
            known_subcommands=list(_SESSION_WORKSTREAM_COMMANDS),
        )
    handler = SESSION_COMMAND_HANDLERS.get(name)
    if handler is None:
        return _error(
            stream,
            f"unknown session command: {name!r}",
            known_subcommands=list(_SESSION_TOP_LEVEL_COMMANDS),
        )
    return handler(args, stream, runtime)
