"""CLI command group for backend registry inspection and checkpoint checks.

This module stays intentionally small:

* ``backend list`` enumerates registered backend adapters.
* ``backend show <backend_name>`` prints one adapter contract.
* ``backend check <variant_id> --checkpoint-json ...`` validates a
  checkpoint payload against a manifest and surfaces backend-aware
  compatibility verdicts.

The imports that touch :mod:`src.backends` or :mod:`src.model` are kept
inside the handlers so CLI startup stays cheap and the command group can
be imported without immediately pulling the full model surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from src.agent.surface_catalog import describe_backend_surface, describe_engine_surface
from src.backends.base import BackendAdapterError

__all__ = [
    "BACKEND_COMMAND_HANDLERS",
    "build_backend_parser",
    "dispatch_backend_command",
    "handle_backend_check",
    "handle_backend_list",
    "handle_backend_engine_list",
    "handle_backend_engine_show",
    "handle_backend_show",
]


def _json_write(out_stream: TextIO, payload: Any) -> None:
    out_stream.write(json.dumps(payload, sort_keys=True))
    out_stream.write("\n")


def _error(out_stream: TextIO, message: str, **extra: Any) -> int:
    payload = {"error": message}
    payload.update(extra)
    _json_write(out_stream, payload)
    return 1


def _load_backend_surface():
    import src.backends as backends

    return backends


def _load_checkpoint_meta(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    checkpoint_json = getattr(args, "checkpoint_json", None)
    checkpoint_file = getattr(args, "checkpoint_file", None)

    if checkpoint_json is None and checkpoint_file is None:
        raise ValueError("backend check requires --checkpoint-json or --checkpoint-file")

    if checkpoint_json is not None:
        payload = json.loads(checkpoint_json)
        source = "inline-json"
    else:
        path = Path(checkpoint_file).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)

    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata must decode to a JSON object")
    return payload, source


def _backend_record(adapter: Any) -> dict[str, Any]:
    return {
        "backend_name": adapter.contract.backend_name,
        "adapter_class": type(adapter).__name__,
        "contract": asdict(adapter.contract),
        "runtime_info": adapter.runtime_info(),
    }


def _engine_record_by_name(engine_name: str) -> dict[str, Any] | None:
    surface = describe_engine_surface()
    for record in surface["engine_adapters"]:
        if record["backend_name"] == engine_name:
            return record
    return None


def handle_backend_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    try:
        payload = describe_backend_surface()
    except Exception as exc:
        return _error(stream, f"failed to load backend registry: {exc}")

    if not payload:
        return _error(stream, "failed to describe backend registry")
    _json_write(stream, payload)
    return 0


def handle_backend_engine_list(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    try:
        payload = describe_engine_surface()
    except Exception as exc:
        return _error(stream, f"failed to load engine adapter registry: {exc}")
    _json_write(stream, {"engine_adapters": payload})
    return 0


def handle_backend_engine_show(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    engine_name = getattr(args, "engine_name", None)
    if not engine_name:
        return _error(stream, "backend engine show requires an engine_name")
    try:
        record = _engine_record_by_name(engine_name)
    except Exception as exc:
        return _error(stream, f"failed to load engine adapter registry: {exc}")
    if record is None:
        surface = describe_engine_surface()
        return _error(
            stream,
            f"unknown engine adapter: {engine_name!r}",
            known_engines=surface["names"],
        )
    _json_write(stream, {"engine": record})
    return 0


def handle_backend_show(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    backend_name = getattr(args, "backend_name", None)
    if not backend_name:
        return _error(stream, "backend show requires a backend_name")

    try:
        backends = _load_backend_surface()
        adapter = backends.get_backend(backend_name)
    except BackendAdapterError as exc:
        return _error(stream, str(exc))
    except Exception as exc:
        return _error(stream, f"failed to load backend registry: {exc}")

    _json_write(stream, {"backend": _backend_record(adapter)})
    return 0


def handle_backend_check(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
) -> int:
    stream = out_stream if out_stream is not None else sys.stdout
    variant_id = getattr(args, "variant_id", None)
    if not variant_id:
        return _error(stream, "backend check requires a variant_id")

    try:
        checkpoint_meta, checkpoint_source = _load_checkpoint_meta(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _error(stream, f"failed to load checkpoint metadata: {exc}")

    try:
        from src.model.compatibility import (
            CompatibilityError,
            check_checkpoint_compatibility,
        )
        from src.model.family import MODEL_VARIANT_REGISTRY, get_variant_by_id
        from src.model.manifest import dump_manifest
    except Exception as exc:
        return _error(stream, f"failed to load model compatibility helpers: {exc}")

    try:
        variant = get_variant_by_id(variant_id)
    except KeyError as exc:
        return _error(
            stream,
            str(exc),
            known_variants=sorted(MODEL_VARIANT_REGISTRY.keys()),
        )

    try:
        verdict = check_checkpoint_compatibility(variant.manifest, checkpoint_meta)
    except CompatibilityError as exc:
        return _error(stream, str(exc))

    _json_write(
        stream,
        {
            "variant_id": variant_id,
            "checkpoint_source": checkpoint_source,
            "manifest": dump_manifest(variant.manifest),
            "checkpoint_meta": checkpoint_meta,
            "compatible": verdict.compatible,
            "severity": verdict.severity,
            "reasons": list(verdict.reasons),
        },
    )
    return 0 if verdict.compatible else 1


BACKEND_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace, TextIO], int]] = {
    "list": handle_backend_list,
    "show": handle_backend_show,
    "engine_list": handle_backend_engine_list,
    "engine_show": handle_backend_engine_show,
    "check": handle_backend_check,
}


def build_backend_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Attach the ``backend`` command group to an existing subparser tree."""
    backend_parser = subparsers.add_parser(
        "backend",
        help="inspect backend adapters and checkpoint compatibility",
        description="Backend registry inspection and checkpoint validation.",
    )
    backend_sub = backend_parser.add_subparsers(
        dest="backend_command",
        metavar="backend_command",
    )

    backend_sub.add_parser("list", help="list registered backends")

    show_p = backend_sub.add_parser("show", help="show one registered backend")
    show_p.add_argument("backend_name", help="backend name, e.g. 'pytorch'")

    engine_p = backend_sub.add_parser("engine", help="inspect inference engine adapters")
    engine_sub = engine_p.add_subparsers(
        dest="backend_engine_command", metavar="backend_engine_command"
    )
    engine_list_p = engine_sub.add_parser("list", help="list known engine adapters")
    engine_list_p.set_defaults(backend_command="engine_list")
    engine_show_p = engine_sub.add_parser("show", help="show one known engine adapter")
    engine_show_p.add_argument("engine_name", help="engine adapter name, e.g. 'vllm'")
    engine_show_p.set_defaults(backend_command="engine_show")

    check_p = backend_sub.add_parser("check", help="validate checkpoint metadata against a variant")
    check_p.add_argument(
        "variant_id",
        help="variant id in the form 'family/variant'",
    )
    check_group = check_p.add_mutually_exclusive_group(required=True)
    check_group.add_argument(
        "--checkpoint-json",
        default=None,
        help="checkpoint metadata encoded as a JSON object",
    )
    check_group.add_argument(
        "--checkpoint-file",
        default=None,
        help="path to a JSON file containing checkpoint metadata",
    )

    return backend_parser


def dispatch_backend_command(
    args: argparse.Namespace,
    out_stream: TextIO | None = None,
) -> int:
    """Route a parsed backend command to its handler."""
    stream = out_stream if out_stream is not None else sys.stdout
    name = getattr(args, "backend_command", None)
    if name is None:
        return _error(
            stream,
            "missing backend command",
            known_subcommands=sorted(BACKEND_COMMAND_HANDLERS.keys()),
        )
    handler = BACKEND_COMMAND_HANDLERS.get(name)
    if handler is None:
        return _error(
            stream,
            f"unknown backend command: {name!r}",
            known_subcommands=sorted(BACKEND_COMMAND_HANDLERS.keys()),
        )
    return handler(args, stream)
