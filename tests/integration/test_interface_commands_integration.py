"""Integration tests for the Aurelius interface CLI command group."""

from __future__ import annotations

import json

import src.cli.main as cli_main


def test_main_parser_includes_interface_group():
    parser = cli_main._build_parser()
    ns = parser.parse_args(["interface", "describe"])

    assert ns.command == "interface"
    assert ns.interface_command == "describe"


def test_main_interface_describe_command_runs_end_to_end(capsys):
    rc = cli_main.main(["interface", "describe"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["shell_capabilities"]["workflow_execution"] is True
    assert payload["framework"]["title"] == "Aurelius Canonical Interface Contract"


def test_main_interface_shell_status_command_runs_end_to_end(capsys):
    rc = cli_main.main(["interface", "shell", "status"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Aurelius Shell" in captured.out
    assert "Workflow runs:" in captured.out
