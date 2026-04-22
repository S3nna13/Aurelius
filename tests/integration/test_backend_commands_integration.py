"""Integration tests for the backend CLI command group."""

from __future__ import annotations

import json

import src.cli.main as cli_main
from src.model.manifest import AURELIUS_REFERENCE_MANIFEST


def test_main_parser_includes_backend_group():
    parser = cli_main._build_parser()
    ns = parser.parse_args(["backend", "list"])

    assert ns.command == "backend"
    assert ns.backend_command == "list"


def test_main_backend_list_command_runs_end_to_end(capsys):
    rc = cli_main.main(["backend", "list"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert any(item["backend_name"] == "pytorch" for item in payload["backends"])


def test_main_backend_check_command_runs_end_to_end(capsys):
    rc = cli_main.main(
        [
            "backend",
            "check",
            AURELIUS_REFERENCE_MANIFEST.registry_key,
            "--checkpoint-json",
            json.dumps(
                {
                    "checkpoint_format_version": "1.0.0",
                    "config_version": "1.0.0",
                    "tokenizer_hash": None,
                }
            ),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["severity"] == "exact"
    assert payload["compatible"] is True
