"""Tests for completion engine."""
from __future__ import annotations

import pytest

from src.cli.completion_engine import CompletionEngine


class TestCompletionEngine:
    def test_complete_command(self):
        eng = CompletionEngine()
        eng.register("deploy", ["start", "stop", "status"])
        candidates = eng.complete("dep")
        assert "deploy" in candidates

    def test_complete_subcommand(self):
        eng = CompletionEngine()
        eng.register("deploy", ["start", "stop", "status"])
        candidates = eng.complete("deploy st")
        assert "start" in candidates
        assert "stop" in candidates

    def test_no_match(self):
        eng = CompletionEngine()
        eng.register("deploy", ["start"])
        candidates = eng.complete("unknown")
        assert candidates == []

    def test_empty_line(self):
        eng = CompletionEngine()
        eng.register("deploy", ["start"])
        candidates = eng.complete("")
        assert "deploy" in candidates

    def test_registered_commands(self):
        eng = CompletionEngine()
        eng.register("a", ["1"])
        eng.register("b", ["2"])
        assert sorted(eng.registered_commands()) == ["a", "b"]