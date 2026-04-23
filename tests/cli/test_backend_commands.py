"""Unit tests for ``src.cli.backend_commands``."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from src.cli.backend_commands import build_backend_parser, dispatch_backend_command
from src.model.manifest import AURELIUS_REFERENCE_MANIFEST


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurelius")
    sub = parser.add_subparsers(dest="command")
    build_backend_parser(sub)
    return parser


def test_backend_parser_wires_subcommands():
    parser = _parser()
    ns = parser.parse_args(["backend", "show", "pytorch"])

    assert ns.command == "backend"
    assert ns.backend_command == "show"
    assert ns.backend_name == "pytorch"


def test_backend_list_outputs_registered_backend_metadata():
    parser = _parser()
    args = parser.parse_args(["backend", "list"])
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["count"] >= 1
    assert any(item["backend_name"] == "pytorch" for item in payload["backends"])


def test_backend_show_outputs_contract_and_runtime_info():
    parser = _parser()
    args = parser.parse_args(["backend", "show", "pytorch"])
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["backend"]["backend_name"] == "pytorch"
    assert payload["backend"]["contract"]["engine_contract"] == "1.0.0"
    assert payload["backend"]["runtime_info"]["available"] is True


def test_backend_engine_list_outputs_known_engine_adapter_metadata():
    parser = _parser()
    args = parser.parse_args(["backend", "engine", "list"])
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["engine_adapters"]["count"] >= 3
    assert "gguf" in payload["engine_adapters"]["names"]


def test_backend_engine_show_outputs_engine_adapter_summary():
    parser = _parser()
    args = parser.parse_args(["backend", "engine", "show", "sglang"])
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["engine"]["backend_name"] == "sglang"
    assert payload["engine"]["registered"] in {True, False}


def test_backend_show_unknown_backend_returns_error():
    parser = _parser()
    args = parser.parse_args(["backend", "show", "no-such-backend"])
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 1
    assert "error" in payload


def test_backend_check_exact_checkpoint_reports_success():
    parser = _parser()
    checkpoint = {
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.0.0",
        "tokenizer_hash": None,
    }
    args = parser.parse_args(
        [
            "backend",
            "check",
            AURELIUS_REFERENCE_MANIFEST.registry_key,
            "--checkpoint-json",
            json.dumps(checkpoint),
        ]
    )
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["variant_id"] == AURELIUS_REFERENCE_MANIFEST.registry_key
    assert payload["severity"] == "exact"
    assert payload["compatible"] is True


def test_backend_check_backend_mismatch_is_major_break():
    parser = _parser()
    checkpoint = {
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.0.0",
        "tokenizer_hash": None,
        "backend_name": "jax",
        "engine_contract": "1.0.0",
        "adapter_contract": "1.0.0",
    }
    args = parser.parse_args(
        [
            "backend",
            "check",
            AURELIUS_REFERENCE_MANIFEST.registry_key,
            "--checkpoint-json",
            json.dumps(checkpoint),
        ]
    )
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 1
    assert payload["severity"] == "major_break"
    assert payload["compatible"] is False
    assert any("checkpoint backend contract" in reason for reason in payload["reasons"])


def test_backend_check_rejects_non_object_json():
    parser = _parser()
    args = parser.parse_args(
        [
            "backend",
            "check",
            AURELIUS_REFERENCE_MANIFEST.registry_key,
            "--checkpoint-json",
            "[]",
        ]
    )
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 1
    assert "checkpoint metadata must decode to a JSON object" in payload["error"]


def test_backend_check_file_round_trip(tmp_path: Path):
    parser = _parser()
    checkpoint_file = tmp_path / "checkpoint.json"
    checkpoint_file.write_text(
        json.dumps(
            {
                "checkpoint_format_version": "1.0.0",
                "config_version": "1.0.0",
                "tokenizer_hash": None,
            }
        ),
        encoding="utf-8",
    )
    args = parser.parse_args(
        [
            "backend",
            "check",
            AURELIUS_REFERENCE_MANIFEST.registry_key,
            "--checkpoint-file",
            str(checkpoint_file),
        ]
    )
    buf = io.StringIO()

    rc = dispatch_backend_command(args, buf)
    payload = json.loads(buf.getvalue())

    assert rc == 0
    assert payload["checkpoint_source"] == str(checkpoint_file)
