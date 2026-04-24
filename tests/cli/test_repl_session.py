"""Tests for src/cli/repl_session.py."""

import pytest

from src.cli.repl_session import ReplConfig, ReplMode, ReplSession


# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------

def test_repl_mode_values():
    assert ReplMode.INTERACTIVE == "interactive"
    assert ReplMode.PIPE == "pipe"
    assert len(ReplMode) == 3


# ---------------------------------------------------------------------------
# ReplConfig defaults
# ---------------------------------------------------------------------------

def test_repl_config_defaults():
    cfg = ReplConfig(mode=ReplMode.INTERACTIVE)
    assert cfg.prompt == "aurelius> "
    assert cfg.history_file == ".aurelius_history"
    assert cfg.max_history == 1000


# ---------------------------------------------------------------------------
# add_to_history / get_history
# ---------------------------------------------------------------------------

def test_add_and_get_history():
    session = ReplSession(ReplConfig(mode=ReplMode.INTERACTIVE))
    session.add_to_history("chat --model base")
    session.add_to_history("eval --dataset bench")
    hist = session.get_history()
    assert hist == ["chat --model base", "eval --dataset bench"]


def test_history_max_capped():
    cfg = ReplConfig(mode=ReplMode.INTERACTIVE, max_history=3)
    session = ReplSession(cfg)
    for i in range(5):
        session.add_to_history(f"cmd{i}")
    hist = session.get_history()
    assert len(hist) == 3
    assert hist == ["cmd2", "cmd3", "cmd4"]


def test_get_history_returns_copy():
    session = ReplSession(ReplConfig(mode=ReplMode.INTERACTIVE))
    session.add_to_history("hello")
    h1 = session.get_history()
    h1.append("mutated")
    assert len(session.get_history()) == 1


# ---------------------------------------------------------------------------
# save_history / load_history
# ---------------------------------------------------------------------------

def test_save_and_load_history(tmp_path):
    session = ReplSession(ReplConfig(mode=ReplMode.INTERACTIVE))
    session.add_to_history("train --epochs 5")
    session.add_to_history("serve --port 8080")
    path = str(tmp_path / "history.txt")
    session.save_history(path)

    session2 = ReplSession(ReplConfig(mode=ReplMode.INTERACTIVE))
    count = session2.load_history(path)
    assert count == 2
    assert session2.get_history() == ["train --epochs 5", "serve --port 8080"]


def test_load_history_missing_file(tmp_path):
    session = ReplSession(ReplConfig(mode=ReplMode.INTERACTIVE))
    count = session.load_history(str(tmp_path / "nonexistent.txt"))
    assert count == 0


def test_save_history_creates_file(tmp_path):
    session = ReplSession(ReplConfig(mode=ReplMode.INTERACTIVE))
    session.add_to_history("version")
    path = tmp_path / "hist.txt"
    session.save_history(str(path))
    assert path.exists()
    assert "version" in path.read_text()


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------

def test_format_prompt_no_context():
    cfg = ReplConfig(mode=ReplMode.INTERACTIVE, prompt="aurelius> ")
    session = ReplSession(cfg)
    assert session.format_prompt() == "aurelius> "


def test_format_prompt_with_model_name():
    cfg = ReplConfig(mode=ReplMode.INTERACTIVE, prompt="({model_name})> ")
    session = ReplSession(cfg)
    result = session.format_prompt({"model_name": "aurelius-1.4B"})
    assert result == "(aurelius-1.4B)> "


def test_format_prompt_missing_key_no_error():
    cfg = ReplConfig(mode=ReplMode.INTERACTIVE, prompt="({model_name})> ")
    session = ReplSession(cfg)
    # Unknown key — should not raise
    result = session.format_prompt({})
    assert isinstance(result, str)


def test_format_prompt_empty_context():
    cfg = ReplConfig(mode=ReplMode.INTERACTIVE, prompt="aurelius> ")
    session = ReplSession(cfg)
    assert session.format_prompt({}) == "aurelius> "
