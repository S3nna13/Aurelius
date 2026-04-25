"""Tests for src/chat/glm_moe_template.py."""

from __future__ import annotations

import pytest

from src.chat.glm_moe_template import (
    GMASK_TOKEN,
    SOP_TOKEN,
    ROLES,
    GlmMoeTemplate,
    visible_text,
    CHAT_TEMPLATE_REGISTRY,
)


@pytest.fixture()
def tmpl() -> GlmMoeTemplate:
    return GlmMoeTemplate()


# ---------------------------------------------------------------------------
# visible_text
# ---------------------------------------------------------------------------


class TestVisibleText:
    def test_plain_string_passthrough(self):
        assert visible_text("hello world") == "hello world"

    def test_empty_string(self):
        assert visible_text("") == ""

    def test_list_with_text_blocks(self):
        blocks = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
        assert visible_text(blocks) == "foobar"

    def test_list_filters_non_text(self):
        blocks = [{"type": "image", "url": "x"}, {"type": "text", "text": "hi"}]
        assert visible_text(blocks) == "hi"

    def test_empty_list(self):
        assert visible_text([]) == ""


# ---------------------------------------------------------------------------
# GlmMoeTemplate.encode
# ---------------------------------------------------------------------------


class TestGlmMoeTemplateEncode:
    def test_empty_messages_returns_prefix(self, tmpl):
        result = tmpl.encode([])
        assert result == GMASK_TOKEN + SOP_TOKEN

    def test_single_user_message(self, tmpl):
        msgs = [{"role": "user", "content": "Hello"}]
        result = tmpl.encode(msgs)
        assert result.startswith(GMASK_TOKEN + SOP_TOKEN)
        assert ROLES["user"] + "\nHello" in result

    def test_system_message(self, tmpl):
        msgs = [{"role": "system", "content": "Be helpful."}]
        result = tmpl.encode(msgs)
        assert ROLES["system"] + "\nBe helpful." in result

    def test_assistant_message_no_thinking(self, tmpl):
        msgs = [{"role": "assistant", "content": "Sure!"}]
        result = tmpl.encode(msgs, enable_thinking=False)
        assert ROLES["assistant"] + "\nSure!" in result
        assert "<think>" not in result

    def test_assistant_with_reasoning_content_thinking_enabled(self, tmpl):
        msgs = [{"role": "assistant", "content": "42", "reasoning_content": "Let me think..."}]
        result = tmpl.encode(msgs, enable_thinking=True)
        assert "<think>\nLet me think...\n</think>\n42" in result

    def test_assistant_with_reasoning_content_thinking_disabled(self, tmpl):
        msgs = [{"role": "assistant", "content": "42", "reasoning_content": "Let me think..."}]
        result = tmpl.encode(msgs, enable_thinking=False)
        assert "<think>" not in result
        assert "42" in result

    def test_observation_role(self, tmpl):
        msgs = [{"role": "observation", "content": "Tool result here"}]
        result = tmpl.encode(msgs)
        assert ROLES["observation"] + "\nTool result here" in result

    def test_multi_turn_conversation(self, tmpl):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        result = tmpl.encode(msgs)
        assert ROLES["user"] + "\nHi" in result
        assert ROLES["assistant"] + "\nHello" in result
        assert ROLES["user"] + "\nBye" in result

    def test_tools_prepended_as_system(self, tmpl):
        tools = [{"name": "search", "description": "Search the web", "parameters": {"properties": {}}}]
        msgs = [{"role": "user", "content": "Find stuff"}]
        result = tmpl.encode(msgs, tools=tools)
        # System block appears before user block.
        system_pos = result.index(ROLES["system"])
        user_pos = result.index(ROLES["user"])
        assert system_pos < user_pos
        assert "search" in result

    def test_defer_loading_skips_tool_system_message(self, tmpl):
        tools = [{"name": "search", "description": "Search", "parameters": {"properties": {}}}]
        msgs = [{"role": "user", "content": "Hi"}]
        result = tmpl.encode(msgs, tools=tools, defer_loading=True)
        assert "search" not in result

    def test_tool_call_encoding(self, tmpl):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": {"expression": "2+2"},
                        }
                    }
                ],
            }
        ]
        result = tmpl.encode(msgs)
        assert "<tool_call>calculator" in result
        assert "<arg_key>expression</arg_key>" in result
        assert "<arg_value>2+2</arg_value>" in result

    def test_list_content_in_user_message(self, tmpl):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "from list"}]}]
        result = tmpl.encode(msgs)
        assert "from list" in result


# ---------------------------------------------------------------------------
# GlmMoeTemplate.decode_tool_call
# ---------------------------------------------------------------------------


class TestGlmMoeTemplateDecode:
    def test_decode_simple_tool_call(self, tmpl):
        text = "<tool_call>myFunc<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        parsed = tmpl.decode_tool_call(text)
        assert parsed is not None
        assert parsed["name"] == "myFunc"
        assert parsed["arguments"]["x"] == "1"

    def test_decode_returns_none_for_no_tool_call(self, tmpl):
        assert tmpl.decode_tool_call("no tool call here") is None

    def test_decode_multiple_args(self, tmpl):
        text = (
            "<tool_call>fn"
            "<arg_key>a</arg_key><arg_value>1</arg_value>"
            "<arg_key>b</arg_key><arg_value>2</arg_value>"
            "</tool_call>"
        )
        parsed = tmpl.decode_tool_call(text)
        assert parsed["arguments"] == {"a": "1", "b": "2"}


# ---------------------------------------------------------------------------
# GlmMoeTemplate.render_tool_schema
# ---------------------------------------------------------------------------


class TestRenderToolSchema:
    def test_basic_schema(self, tmpl):
        tool = {
            "name": "search",
            "description": "Search the web",
            "parameters": {"properties": {"query": {}, "limit": {}}},
        }
        rendered = tmpl.render_tool_schema(tool)
        assert "search" in rendered
        assert "Search the web" in rendered
        assert "query" in rendered
        assert "limit" in rendered

    def test_no_parameters(self, tmpl):
        tool = {"name": "ping", "description": "Ping", "parameters": {}}
        rendered = tmpl.render_tool_schema(tool)
        assert "(none)" in rendered


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_entry():
    assert "glm_moe" in CHAT_TEMPLATE_REGISTRY
    assert isinstance(CHAT_TEMPLATE_REGISTRY["glm_moe"], GlmMoeTemplate)
