"""Tests for src/cli/completion_engine.py."""

from src.cli.completion_engine import Completion, CompletionEngine, CompletionKind

# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------


def test_completion_kind_values():
    assert CompletionKind.COMMAND == "command"
    assert CompletionKind.FILEPATH == "filepath"
    assert len(CompletionKind) == 5


# ---------------------------------------------------------------------------
# Completion dataclass
# ---------------------------------------------------------------------------


def test_completion_dataclass_defaults():
    c = Completion(text="/chat", kind=CompletionKind.COMMAND)
    assert c.description == ""


def test_completion_dataclass_fields():
    c = Completion(text="/help", kind=CompletionKind.COMMAND, description="Show help")
    assert c.text == "/help"
    assert c.kind == CompletionKind.COMMAND
    assert c.description == "Show help"


# ---------------------------------------------------------------------------
# complete() — COMMAND completions
# ---------------------------------------------------------------------------


def test_complete_slash_returns_commands():
    engine = CompletionEngine()
    results = engine.complete("/")
    kinds = {r.kind for r in results}
    assert CompletionKind.COMMAND in kinds


def test_complete_slash_chat_prefix():
    engine = CompletionEngine()
    results = engine.complete("/ch")
    texts = [r.text for r in results]
    assert "/chat" in texts


def test_complete_slash_all_commands_present():
    engine = CompletionEngine()
    results = engine.complete("/")
    texts = {r.text for r in results}
    for cmd in ["chat", "eval", "train", "serve", "export", "version", "help", "quit"]:
        assert f"/{cmd}" in texts


# ---------------------------------------------------------------------------
# complete() — FILEPATH completions
# ---------------------------------------------------------------------------


def test_complete_dot_slash_returns_filepaths():
    engine = CompletionEngine()
    results = engine.complete("./")
    kinds = {r.kind for r in results}
    assert CompletionKind.FILEPATH in kinds


def test_complete_filepath_returns_three_stubs():
    engine = CompletionEngine()
    results = engine.complete("./models")
    assert len(results) == 3


def test_complete_absolute_path_returns_filepaths():
    engine = CompletionEngine()
    results = engine.complete("/usr/local/")
    kinds = {r.kind for r in results}
    assert CompletionKind.FILEPATH in kinds


# ---------------------------------------------------------------------------
# complete() — HISTORY completions
# ---------------------------------------------------------------------------


def test_complete_exact_history_match():
    engine = CompletionEngine()
    engine.feed_history(["chat --model base", "eval --metric bleu"])
    results = engine.complete("chat --model base")
    assert any(r.kind == CompletionKind.HISTORY for r in results)


def test_complete_history_prefix_match():
    engine = CompletionEngine()
    engine.feed_history(["chat --model base", "chat --persona coding"])
    results = engine.complete("chat")
    assert len(results) >= 1
    assert all(r.kind == CompletionKind.HISTORY for r in results)


def test_complete_no_match_returns_empty():
    engine = CompletionEngine()
    results = engine.complete("zzznomatch")
    assert results == []


# ---------------------------------------------------------------------------
# register_command
# ---------------------------------------------------------------------------


def test_register_command_adds_to_completions():
    engine = CompletionEngine()
    engine.register_command("custom", "My custom command")
    results = engine.complete("/custom")
    texts = [r.text for r in results]
    assert "/custom" in texts


def test_register_command_no_duplicate():
    engine = CompletionEngine()
    engine.register_command("chat", "Already exists")
    results = engine.complete("/chat")
    assert sum(1 for r in results if r.text == "/chat") == 1


# ---------------------------------------------------------------------------
# get_completions_for
# ---------------------------------------------------------------------------


def test_get_completions_for_chat():
    engine = CompletionEngine()
    args = engine.get_completions_for("chat")
    assert "--model" in args


def test_get_completions_for_unknown_returns_empty():
    engine = CompletionEngine()
    assert engine.get_completions_for("nonexistent") == []


def test_get_completions_for_version_is_empty():
    engine = CompletionEngine()
    assert engine.get_completions_for("version") == []
