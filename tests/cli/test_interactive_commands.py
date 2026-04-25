"""Tests for interactive CLI commands."""
from __future__ import annotations

import pytest

from src.cli.interactive_commands import InteractiveCLI, CommandResult


class TestInteractiveCLI:
    def test_register_and_execute(self):
        cli = InteractiveCLI()
        cli.register("ping", lambda args: CommandResult(True, "pong"))
        result = cli.execute("ping")
        assert result.success is True
        assert result.output == "pong"

    def test_unknown_command(self):
        cli = InteractiveCLI()
        result = cli.execute("bogus")
        assert result.success is False
        assert "unknown" in result.output.lower()

    def test_aliases(self):
        cli = InteractiveCLI()
        cli.register("help", lambda args: CommandResult(True, "help text"), aliases=["h", "?"])
        assert cli.execute("h").output == "help text"
        assert cli.execute("?").output == "help text"

    def test_commands_with_args(self):
        cli = InteractiveCLI()
        cli.register("echo", lambda args: CommandResult(True, " ".join(args)))
        result = cli.execute("echo hello world")
        assert result.output == "hello world"

    def test_list_commands(self):
        cli = InteractiveCLI()
        cli.register("a", lambda args: CommandResult(True, ""))
        cli.register("b", lambda args: CommandResult(True, ""))
        cmds = cli.list_commands()
        assert sorted(c.name for c in cmds) == ["a", "b"]

    def test_error_handling(self):
        cli = InteractiveCLI()
        cli.register("fail", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
        result = cli.execute("fail")
        assert result.success is False
        assert "boom" in result.output