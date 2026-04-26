"""Tests for the interactive terminal chat REPL."""

from src.serving.terminal_chat import TerminalChat, build_chatml_prompt

_MOCK_RESPONSE = "[model not loaded — mock response]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat(**kwargs) -> TerminalChat:
    """Return a TerminalChat instance using the default mock generate_fn."""
    return TerminalChat(**kwargs)


# ---------------------------------------------------------------------------
# Test 1: TerminalChat instantiates with default args
# ---------------------------------------------------------------------------


def test_instantiation_defaults():
    tc = _make_chat()
    assert isinstance(tc, TerminalChat)
    assert tc.history == []
    assert tc.system_prompt != ""
    assert tc.use_colors is True
    assert callable(tc.generate_fn)


# ---------------------------------------------------------------------------
# Test 2: add_message increases history length
# ---------------------------------------------------------------------------


def test_add_message_increases_history():
    tc = _make_chat()
    assert len(tc.history) == 0
    tc.add_message("user", "Hello")
    assert len(tc.history) == 1
    tc.add_message("assistant", "Hi there")
    assert len(tc.history) == 2


# ---------------------------------------------------------------------------
# Test 3: build_prompt returns a non-empty string
# ---------------------------------------------------------------------------


def test_build_prompt_nonempty():
    tc = _make_chat()
    tc.add_message("user", "What is 2 + 2?")
    prompt = tc.build_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Test 4: build_prompt includes the user message
# ---------------------------------------------------------------------------


def test_build_prompt_includes_user_message():
    tc = _make_chat()
    tc.add_message("user", "tell me about entropy")
    prompt = tc.build_prompt()
    assert "tell me about entropy" in prompt


# ---------------------------------------------------------------------------
# Test 5: reset clears history back to 0 user/assistant messages
# ---------------------------------------------------------------------------


def test_reset_clears_history():
    tc = _make_chat()
    tc.add_message("user", "Hello")
    tc.add_message("assistant", "Hi")
    tc.add_message("user", "How are you?")
    tc.reset()
    assert tc.history == []
    assert len(tc.history) == 0


# ---------------------------------------------------------------------------
# Test 6: chat returns a string
# ---------------------------------------------------------------------------


def test_chat_returns_string():
    tc = _make_chat()
    result = tc.chat("Hello, Aurelius!")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test 7: chat adds both user and assistant messages to history
# ---------------------------------------------------------------------------


def test_chat_adds_both_messages():
    tc = _make_chat()
    before = len(tc.history)
    tc.chat("What is the meaning of life?")
    assert len(tc.history) == before + 2
    roles = [m["role"] for m in tc.history]
    assert "user" in roles
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# Test 8: _colorize with use_colors=True wraps in ANSI codes
# ---------------------------------------------------------------------------


def test_colorize_with_colors_contains_ansi():
    tc = _make_chat(use_colors=True)
    result = tc._colorize("hello", "blue")
    assert "\033[" in result


# ---------------------------------------------------------------------------
# Test 9: _colorize with use_colors=False returns plain text
# ---------------------------------------------------------------------------


def test_colorize_without_colors_returns_plain():
    tc = _make_chat(use_colors=False)
    text = "hello world"
    result = tc._colorize(text, "blue")
    assert result == text
    assert "\033[" not in result


# ---------------------------------------------------------------------------
# Test 10: build_chatml_prompt returns string containing system content
# ---------------------------------------------------------------------------


def test_build_chatml_prompt_contains_system():
    messages = [{"role": "user", "content": "Hi"}]
    prompt = build_chatml_prompt(messages, "You are a helpful assistant.")
    assert "You are a helpful assistant." in prompt


# ---------------------------------------------------------------------------
# Test 11: build_chatml_prompt returns string containing user content
# ---------------------------------------------------------------------------


def test_build_chatml_prompt_contains_user():
    messages = [{"role": "user", "content": "What is quantum computing?"}]
    prompt = build_chatml_prompt(messages, "Be concise.")
    assert "What is quantum computing?" in prompt


# ---------------------------------------------------------------------------
# Test 12: Special command /reset clears history (via reset() directly)
# ---------------------------------------------------------------------------


def test_reset_command_clears_history_direct():
    tc = _make_chat()
    tc.chat("First message")
    tc.chat("Second message")
    assert len(tc.history) == 4
    tc.reset()
    assert len(tc.history) == 0


# ---------------------------------------------------------------------------
# Test 13: chat with mock generate_fn returns the mock string
# ---------------------------------------------------------------------------


def test_chat_mock_generate_fn_returns_mock_string():
    tc = _make_chat()
    result = tc.chat("Anything at all")
    assert result == _MOCK_RESPONSE


# ---------------------------------------------------------------------------
# Test 14: chat with custom generate_fn uses that callable
# ---------------------------------------------------------------------------


def test_chat_custom_generate_fn():
    def custom_fn(prompt):
        return "custom response"

    tc = TerminalChat(generate_fn=custom_fn)
    result = tc.chat("Hello")
    assert result == "custom response"
