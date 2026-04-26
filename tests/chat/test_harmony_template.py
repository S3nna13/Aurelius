"""Unit tests for the Harmony Response Format (GPT-OSS-120B, arXiv:2508.10925).

Covers the HarmonyTemplate.render() / parse_roles() surface defined in
src/chat/harmony_template.py.  All tests use plain dicts as the message
representation so the API contract is clear.
"""

from __future__ import annotations

import pytest

from src.chat.harmony_template import VALID_ROLES, HarmonyTemplate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tpl() -> HarmonyTemplate:
    return HarmonyTemplate()


# Convenience message dicts used across multiple tests.
_SYS = {"role": "system", "content": "You are a helpful assistant."}
_USER = {"role": "user", "content": "Hello, world!"}
_ASST = {"role": "assistant", "content": "Hi there!"}
_TOOL = {"role": "tool", "content": '{"result": 42}'}


# ---------------------------------------------------------------------------
# Test 1 — starts with BOS token
# ---------------------------------------------------------------------------


def test_starts_with_bos(tpl: HarmonyTemplate) -> None:
    out = tpl.render([_USER])
    assert out.startswith("<|begin_of_text|>"), (
        f"rendered string should begin with BOS, got: {out[:50]!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — system delimiters
# ---------------------------------------------------------------------------


def test_system_delimiters(tpl: HarmonyTemplate) -> None:
    out = tpl.render([_SYS])
    assert "<|system|>" in out
    assert "<|end_system|>" in out


# ---------------------------------------------------------------------------
# Test 3 — user delimiters
# ---------------------------------------------------------------------------


def test_user_delimiters(tpl: HarmonyTemplate) -> None:
    out = tpl.render([_USER])
    assert "<|user|>" in out
    assert "<|end_user|>" in out


# ---------------------------------------------------------------------------
# Test 4 — assistant delimiters
# ---------------------------------------------------------------------------


def test_assistant_delimiters(tpl: HarmonyTemplate) -> None:
    out = tpl.render([_ASST])
    assert "<|assistant|>" in out
    assert "<|end_assistant|>" in out


# ---------------------------------------------------------------------------
# Test 5 — tool-result delimiters
# ---------------------------------------------------------------------------


def test_tool_result_delimiters(tpl: HarmonyTemplate) -> None:
    out = tpl.render([_TOOL])
    assert "<tool_result>" in out
    assert "</tool_result>" in out


# ---------------------------------------------------------------------------
# Test 6 — thinking wrapped in <think>...</think>
# ---------------------------------------------------------------------------


def test_thinking_wrapped(tpl: HarmonyTemplate) -> None:
    msg = {"role": "assistant", "content": "answer", "thinking": "step-by-step"}
    out = tpl.render([msg])
    assert "<think>step-by-step</think>" in out


# ---------------------------------------------------------------------------
# Test 7 — no thinking field → no <think> tag
# ---------------------------------------------------------------------------


def test_no_thinking_no_think_tag(tpl: HarmonyTemplate) -> None:
    out = tpl.render([_ASST])
    assert "<think>" not in out


# ---------------------------------------------------------------------------
# Test 8 — tool_calls in assistant turn
# ---------------------------------------------------------------------------


def test_tool_call_in_assistant(tpl: HarmonyTemplate) -> None:
    msg = {"role": "assistant", "content": "", "tool_calls": ["fn()"]}
    out = tpl.render([msg])
    assert "<tool_call>fn()</tool_call>" in out


# ---------------------------------------------------------------------------
# Test 9 — unknown role raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_role_raises(tpl: HarmonyTemplate) -> None:
    with pytest.raises(ValueError, match="Unknown role"):
        tpl.render([{"role": "invalid", "content": "oops"}])


# ---------------------------------------------------------------------------
# Test 10 — empty message list does not crash
# ---------------------------------------------------------------------------


def test_empty_messages_no_crash(tpl: HarmonyTemplate) -> None:
    out = tpl.render([])
    # Should still start with BOS and not raise.
    assert out == tpl.bos


# ---------------------------------------------------------------------------
# Test 11 — role order is preserved (system → user → assistant)
# ---------------------------------------------------------------------------


def test_role_order_preserved(tpl: HarmonyTemplate) -> None:
    msgs = [_SYS, _USER, _ASST]
    out = tpl.render(msgs)
    sys_pos = out.index("<|system|>")
    user_pos = out.index("<|user|>")
    asst_pos = out.index("<|assistant|>")
    assert sys_pos < user_pos < asst_pos


# ---------------------------------------------------------------------------
# Test 12 — add_eos=True appends <|end_of_text|>
# ---------------------------------------------------------------------------


def test_add_eos_flag() -> None:
    tpl_eos = HarmonyTemplate(add_eos=True)
    out = tpl_eos.render([_USER])
    assert out.endswith("<|end_of_text|>")


# ---------------------------------------------------------------------------
# Test 13 — add_eos=False (default) does NOT append EOS token
# ---------------------------------------------------------------------------


def test_add_eos_false(tpl: HarmonyTemplate) -> None:
    assert not tpl.add_eos
    out = tpl.render([_USER])
    assert not out.endswith("<|end_of_text|>")


# ---------------------------------------------------------------------------
# Test 14 — content appears verbatim in output
# ---------------------------------------------------------------------------


def test_content_preserved(tpl: HarmonyTemplate) -> None:
    payload = "verbatim content with punctuation: 2 + 2 == 4!"
    out = tpl.render([{"role": "user", "content": payload}])
    assert payload in out


# ---------------------------------------------------------------------------
# Test 15 — parse_roles round-trip
# ---------------------------------------------------------------------------


def test_parse_roles_round_trip(tpl: HarmonyTemplate) -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    rendered = tpl.render(msgs)
    roles = tpl.parse_roles(rendered)
    expected = [m["role"] for m in msgs]
    assert roles == expected


# ---------------------------------------------------------------------------
# Additional tests (beyond minimum 14+1)
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_in_one_assistant_turn(tpl: HarmonyTemplate) -> None:
    msg = {
        "role": "assistant",
        "content": "done",
        "tool_calls": ["read_file()", "write_file()"],
    }
    out = tpl.render([msg])
    assert "<tool_call>read_file()</tool_call>" in out
    assert "<tool_call>write_file()</tool_call>" in out


def test_thinking_appears_before_content(tpl: HarmonyTemplate) -> None:
    msg = {"role": "assistant", "content": "final answer", "thinking": "my reasoning"}
    out = tpl.render([msg])
    think_pos = out.index("<think>")
    content_pos = out.index("final answer")
    assert think_pos < content_pos


def test_bos_is_customisable() -> None:
    custom = HarmonyTemplate(bos="<BOS>")
    out = custom.render([_USER])
    assert out.startswith("<BOS>")


def test_eos_token_customisable() -> None:
    custom = HarmonyTemplate(eos="<EOS>", add_eos=True)
    out = custom.render([_USER])
    assert out.endswith("<EOS>")


def test_render_only_system(tpl: HarmonyTemplate) -> None:
    out = tpl.render([{"role": "system", "content": "You are Aurelius."}])
    assert "You are Aurelius." in out
    assert "<|system|>" in out
    assert "<|end_system|>" in out
    assert "<|user|>" not in out


def test_tool_result_content_preserved(tpl: HarmonyTemplate) -> None:
    payload = '{"status": "ok", "value": 99}'
    out = tpl.render([{"role": "tool", "content": payload}])
    assert payload in out


@pytest.mark.parametrize("role", sorted(VALID_ROLES))
def test_all_valid_roles_accepted(tpl: HarmonyTemplate, role: str) -> None:
    """Every member of VALID_ROLES must not raise on render."""
    tpl.render([{"role": role, "content": "test"}])


def test_empty_thinking_does_not_emit_think_tag(tpl: HarmonyTemplate) -> None:
    msg = {"role": "assistant", "content": "answer", "thinking": ""}
    out = tpl.render([msg])
    assert "<think>" not in out


def test_parse_roles_empty_string(tpl: HarmonyTemplate) -> None:
    assert tpl.parse_roles("") == []


def test_render_returns_str(tpl: HarmonyTemplate) -> None:
    assert isinstance(tpl.render([_USER]), str)
