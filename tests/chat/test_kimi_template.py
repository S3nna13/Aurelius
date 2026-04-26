"""Tests for src/chat/kimi_template.py."""

from __future__ import annotations

import json

import pytest

from src.chat.kimi_template import (
    CHAT_TEMPLATE_REGISTRY,
    IM_ASSISTANT,
    IM_END,
    IM_MIDDLE,
    IM_SYSTEM,
    IM_USER,
    TOOL_CALL_BEGIN,
    TOOL_CALLS_BEGIN,
    TOOL_CALLS_END,
    KimiChatTemplate,
    render_content,
    set_role,
)


@pytest.fixture()
def tmpl() -> KimiChatTemplate:
    return KimiChatTemplate()


# ---------------------------------------------------------------------------
# render_content
# ---------------------------------------------------------------------------


class TestRenderContent:
    def test_plain_string(self):
        assert render_content({"role": "user", "content": "hello"}) == "hello"

    def test_empty_string(self):
        assert render_content({"role": "user", "content": ""}) == ""

    def test_list_content(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert render_content(msg) == "hi"

    def test_list_filters_non_text(self):
        msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "ok"}]}
        assert render_content(msg) == "ok"

    def test_missing_content_key(self):
        assert render_content({"role": "user"}) == ""


# ---------------------------------------------------------------------------
# set_role
# ---------------------------------------------------------------------------


class TestSetRole:
    def test_user_role(self):
        result = set_role({"role": "user"})
        assert result.startswith(IM_USER)
        assert IM_MIDDLE in result

    def test_assistant_role(self):
        result = set_role({"role": "assistant"})
        assert result.startswith(IM_ASSISTANT)

    def test_system_role(self):
        result = set_role({"role": "system"})
        assert result.startswith(IM_SYSTEM)

    def test_custom_name(self):
        result = set_role({"role": "user", "name": "Alice"})
        assert "Alice" in result

    def test_default_name_is_role(self):
        result = set_role({"role": "user"})
        assert "user" in result


# ---------------------------------------------------------------------------
# KimiChatTemplate.encode
# ---------------------------------------------------------------------------


class TestKimiChatTemplateEncode:
    def test_empty_messages(self, tmpl):
        result = tmpl.encode([])
        assert result == ""

    def test_single_user_message(self, tmpl):
        msgs = [{"role": "user", "content": "Hello"}]
        result = tmpl.encode(msgs)
        assert IM_USER in result
        assert "Hello" in result
        assert IM_END in result

    def test_system_message(self, tmpl):
        msgs = [{"role": "system", "content": "Be concise."}]
        result = tmpl.encode(msgs)
        assert IM_SYSTEM in result
        assert "Be concise." in result

    def test_assistant_message(self, tmpl):
        msgs = [{"role": "assistant", "content": "Sure!"}]
        result = tmpl.encode(msgs)
        assert IM_ASSISTANT in result
        assert "Sure!" in result

    def test_multi_turn(self, tmpl):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = tmpl.encode(msgs)
        assert IM_USER in result
        assert IM_ASSISTANT in result

    def test_tools_prepended(self, tmpl):
        tools = [{"name": "search", "description": "Search"}]
        msgs = [{"role": "user", "content": "Hi"}]
        result = tmpl.encode(msgs, tools=tools)
        tool_pos = result.index("tool_declare")
        user_pos = result.index("Hi")
        assert tool_pos < user_pos

    def test_tools_ts_str_overrides_tools(self, tmpl):
        tools = [{"name": "search"}]
        msgs = [{"role": "user", "content": "Hi"}]
        result = tmpl.encode(msgs, tools=tools, tools_ts_str="CUSTOM_TOOLS")
        assert "CUSTOM_TOOLS" in result

    def test_tool_calls_encoded(self, tmpl):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": '{"q": "ai"}'}}
                ],
            }
        ]
        result = tmpl.encode(msgs)
        assert TOOL_CALLS_BEGIN in result
        assert TOOL_CALLS_END in result
        assert TOOL_CALL_BEGIN in result
        assert "call_1" in result

    def test_no_tools_no_tool_declare(self, tmpl):
        msgs = [{"role": "user", "content": "Hi"}]
        result = tmpl.encode(msgs)
        assert "tool_declare" not in result


# ---------------------------------------------------------------------------
# KimiChatTemplate.split_hist_suffix
# ---------------------------------------------------------------------------


class TestSplitHistSuffix:
    def test_splits_at_last_non_tool_call_assistant(self, tmpl):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        hist, suffix = tmpl.split_hist_suffix(msgs)
        assert hist == [{"role": "user", "content": "Q"}]
        assert suffix == [{"role": "assistant", "content": "A"}]

    def test_tool_call_assistant_is_skipped(self, tmpl):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "tool_calls": [{}], "content": ""},
            {"role": "assistant", "content": "Final"},
        ]
        hist, suffix = tmpl.split_hist_suffix(msgs)
        assert suffix[0] == {"role": "assistant", "content": "Final"}

    def test_no_assistant_message(self, tmpl):
        msgs = [{"role": "user", "content": "Hi"}]
        hist, suffix = tmpl.split_hist_suffix(msgs)
        assert hist == msgs
        assert suffix == []

    def test_empty_messages(self, tmpl):
        hist, suffix = tmpl.split_hist_suffix([])
        assert hist == []
        assert suffix == []


# ---------------------------------------------------------------------------
# KimiChatTemplate.render_tools_as_json
# ---------------------------------------------------------------------------


def test_render_tools_as_json(tmpl):
    tools = [{"name": "foo", "description": "bar"}]
    result = tmpl.render_tools_as_json(tools)
    parsed = json.loads(result)
    assert parsed[0]["name"] == "foo"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_entry():
    assert "kimi" in CHAT_TEMPLATE_REGISTRY
    assert isinstance(CHAT_TEMPLATE_REGISTRY["kimi"], KimiChatTemplate)
