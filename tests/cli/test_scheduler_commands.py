"""Tests for aurelius_cli.scheduler_commands argument parsing and dispatch.

Unit tests only — no scheduler execution is started.
"""

import argparse

from aurelius_cli.scheduler_commands import build_schedule_parser, handle_schedule


def build_parser() -> argparse.ArgumentParser:
    """Helper: build a top-level parser with schedule subparser attached."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    build_schedule_parser(sub)
    return parser


class TestScheduleParser:
    def test_schedule_cron_parses_correctly(self):
        parser = build_parser()
        args = parser.parse_args(["schedule", "cron", "0 2 * * *", "--", "python", "backup.py"])
        assert args.command == "schedule"
        assert args.schedule_cmd == "cron"
        assert args.cron_expr == "0 2 * * *"
        assert args.shell_cmd == ["python", "backup.py"]

    def test_schedule_interval_parses(self):
        parser = build_parser()
        args = parser.parse_args(["schedule", "interval", "60", "--", "echo", "hello"])
        assert args.command == "schedule"
        assert args.schedule_cmd == "interval"
        assert args.seconds == 60.0
        assert args.shell_cmd == ["echo", "hello"]

    def test_schedule_once_parses(self):
        parser = build_parser()
        args = parser.parse_args(["schedule", "once", "300", "--", "say", "done"])
        assert args.command == "schedule"
        assert args.schedule_cmd == "once"
        assert args.delay == 300.0
        assert args.shell_cmd == ["say", "done"]

    def test_handle_schedule_missing_shell_cmd_returns_error(self):
        ns = argparse.Namespace(schedule_cmd="cron", cron_expr="* * * * *", shell_cmd=[])
        rc = handle_schedule(ns)
        assert rc == 2


class TestSchedulerDispatch:
    """Ensure the command integrates with main() via parser build only."""

    def test_parser_build_does_not_raise(self):
        """build_schedule_parser should complete without exception."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        build_schedule_parser(sub)
