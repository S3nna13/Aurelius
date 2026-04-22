"""Integration tests for the family CLI subcommand group."""

from __future__ import annotations

import argparse
import io
import json

from src.cli import family_commands as fc


def test_build_family_parser_attaches_subparsers():
    parser = argparse.ArgumentParser(prog="aurelius")
    sub = parser.add_subparsers(dest="command")
    fc.build_family_parser(sub)
    ns = parser.parse_args(["family", "list"])
    assert ns.command == "family"
    assert ns.family_command == "list"


def test_family_list_integration_runs_end_to_end():
    parser = argparse.ArgumentParser(prog="aurelius")
    sub = parser.add_subparsers(dest="command")
    fc.build_family_parser(sub)
    ns = parser.parse_args(["family", "list"])
    buf = io.StringIO()
    rc = fc.dispatch_family_command(ns, buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "aurelius" in payload["families"]


def test_family_check_integration_research_denied_in_production():
    parser = argparse.ArgumentParser(prog="aurelius")
    sub = parser.add_subparsers(dest="command")
    fc.build_family_parser(sub)
    ns = parser.parse_args(
        [
            "family",
            "check",
            "aurelius/base-1.395b",
            "--policy",
            "production",
        ]
    )
    buf = io.StringIO()
    rc = fc.dispatch_family_command(ns, buf)
    assert rc == 1
    payload = json.loads(buf.getvalue())
    assert payload["allowed"] is False


def test_family_compare_integration_reports_manifest_compatibility():
    parser = argparse.ArgumentParser(prog="aurelius")
    sub = parser.add_subparsers(dest="command")
    fc.build_family_parser(sub)
    ns = parser.parse_args(
        [
            "family",
            "compare",
            "aurelius/base-1.395b",
            "aurelius/base-1.395b",
        ]
    )
    buf = io.StringIO()
    rc = fc.dispatch_family_command(ns, buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["compatible"] is True
    assert payload["severity"] == "exact"
    assert payload["backend_contract_verdict"] == "exact"


def test_main_cli_still_starts_with_help():
    """Verify existing CLI startup path still parses (additive rule)."""
    from src.cli import main as cli_main

    parser = cli_main._build_parser()
    # Should still have the pre-existing commands at minimum.
    ns = parser.parse_args(["chat"])
    assert ns.command == "chat"
