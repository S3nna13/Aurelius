"""Tests for agent CLI commands."""
from __future__ import annotations

import pytest

from src.cli.agent_commands import AgentCLICommand, AgentCLIDispatcher


class TestAgentCLIDispatcher:
    def test_register_and_dispatch(self):
        dispatcher = AgentCLIDispatcher()
        cmd = AgentCLICommand(name="greet", description="say hello",
                               handler=lambda args: f"hello {' '.join(args)}")
        dispatcher.register(cmd)
        
        result = dispatcher.dispatch("greet world")
        assert result == "hello world"
    
    def test_alias_dispatches_same_handler(self):
        dispatcher = AgentCLIDispatcher()
        cmd = AgentCLICommand(name="help", description="show help",
                               handler=lambda _: "help text", aliases=["h", "?"])
        dispatcher.register(cmd)
        
        assert dispatcher.dispatch("h") == "help text"
        assert dispatcher.dispatch("?") == "help text"
    
    def test_empty_string_returns_no_command_message(self):
        dispatcher = AgentCLIDispatcher()
        assert dispatcher.dispatch("") == "no command"
    
    def test_unknown_command(self):
        dispatcher = AgentCLIDispatcher()
        assert dispatcher.dispatch("bogus") == "unknown command: bogus"
    
    def test_list_commands_no_duplicates(self):
        dispatcher = AgentCLIDispatcher()
        cmd1 = AgentCLICommand(name="a", description="first",
                                handler=lambda _: "a", aliases=["a1"])
        cmd2 = AgentCLICommand(name="b", description="second",
                                handler=lambda _: "b")
        dispatcher.register(cmd1)
        dispatcher.register(cmd2)
        
        names = [c.name for c in dispatcher.list_commands()]
        assert sorted(names) == ["a", "b"]