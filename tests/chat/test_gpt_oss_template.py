"""Tests for src/chat/gpt_oss_template.py."""

from __future__ import annotations

import json

import pytest

from src.chat.gpt_oss_template import (
    ANALYSIS_CHANNEL,
    CALL,
    CHANNEL,
    CHAT_TEMPLATE_REGISTRY,
    END,
    FINAL_CHANNEL,
    MESSAGE,
    RETURN,
    START,
    GptOssTemplate,
)


@pytest.fixture()
def tmpl() -> GptOssTemplate:
    return GptOssTemplate()


# ---------------------------------------------------------------------------
# encode — basic cases
# ---------------------------------------------------------------------------


class TestGptOssTemplateEncodeBasic:
    def test_empty_messages_no_generation_prompt(self, tmpl):
        result = tmpl.encode([], add_generation_prompt=False)
        assert result == ""

    def test_empty_messages_with_generation_prompt(self, tmpl):
        result = tmpl.encode([], add_generation_prompt=True)
        assert result.startswith(START + "assistant")

    def test_user_message(self, tmpl):
        msgs = [{"role": "user", "content": "Hello"}]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert f"{START}user{CHANNEL}{FINAL_CHANNEL}{MESSAGE}Hello{END}" in result

    def test_developer_message(self, tmpl):
        msgs = [{"role": "developer", "content": "You are helpful."}]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert f"{START}developer{CHANNEL}{FINAL_CHANNEL}{MESSAGE}You are helpful.{END}" in result

    def test_system_treated_as_developer(self, tmpl):
        msgs = [{"role": "system", "content": "Be concise."}]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert f"{START}developer{CHANNEL}" in result

    def test_assistant_final_channel(self, tmpl):
        msgs = [{"role": "assistant", "content": "42"}]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert f"{START}assistant{CHANNEL}{FINAL_CHANNEL}{MESSAGE}42{END}" in result

    def test_multi_turn(self, tmpl):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert "user" in result
        assert "assistant" in result


# ---------------------------------------------------------------------------
# encode — reasoning / analysis channel
# ---------------------------------------------------------------------------


class TestGptOssTemplateReasoning:
    def test_last_assistant_with_reasoning_emits_analysis(self, tmpl):
        msgs = [{"role": "assistant", "content": "answer", "reasoning_content": "my thinking"}]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert f"{START}assistant{CHANNEL}{ANALYSIS_CHANNEL}{MESSAGE}my thinking{END}" in result

    def test_non_last_assistant_reasoning_suppressed(self, tmpl):
        msgs = [
            {"role": "assistant", "content": "first", "reasoning_content": "think1"},
            {"role": "assistant", "content": "second"},
        ]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        # first assistant should NOT have analysis block since there is a later assistant
        # second assistant should have final block
        assert f"{START}assistant{CHANNEL}{FINAL_CHANNEL}{MESSAGE}second{END}" in result
        # Exactly one analysis block for the last assistant only — here last has no reasoning
        # so no analysis at all expected.
        assert ANALYSIS_CHANNEL not in result

    def test_thinking_field_used_as_reasoning(self, tmpl):
        msgs = [{"role": "assistant", "content": "ans", "thinking": "deep thought"}]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert "deep thought" in result
        assert ANALYSIS_CHANNEL in result


# ---------------------------------------------------------------------------
# encode — tool calls
# ---------------------------------------------------------------------------


class TestGptOssTemplateToolCalls:
    def test_tool_call_uses_call_delimiter(self, tmpl):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "search", "arguments": {"q": "ai"}}}],
                "content": "",
            }
        ]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert "to=functions.search" in result
        assert CALL in result

    def test_tool_result_uses_return_delimiter(self, tmpl):
        msgs = [{"role": "tool", "name": "search", "content": "results here"}]
        result = tmpl.encode(msgs, add_generation_prompt=False)
        assert "functions.search" in result
        assert RETURN in result

    def test_tools_namespace_prepended(self, tmpl):
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        ]
        msgs = [{"role": "user", "content": "hi"}]
        result = tmpl.encode(msgs, tools=tools, add_generation_prompt=False)
        assert "namespace functions" in result
        assert "search" in result


# ---------------------------------------------------------------------------
# decode_tool_call
# ---------------------------------------------------------------------------


class TestGptOssTemplateDecode:
    def test_decode_valid_tool_call(self, tmpl):
        args = json.dumps({"q": "test"})
        text = f"{START}assistant to=functions.search{CHANNEL}commentary json{MESSAGE}{args}{CALL}"
        parsed = tmpl.decode_tool_call(text)
        assert parsed is not None
        assert parsed["name"] == "search"
        assert parsed["arguments"]["q"] == "test"

    def test_decode_returns_none_when_no_match(self, tmpl):
        assert tmpl.decode_tool_call("no tool call") is None

    def test_decode_invalid_json_returns_raw_string(self, tmpl):
        text = f"{START}assistant to=functions.foo{CHANNEL}commentary json{MESSAGE}not_json{CALL}"
        parsed = tmpl.decode_tool_call(text)
        assert parsed is not None
        assert isinstance(parsed["arguments"], str)


# ---------------------------------------------------------------------------
# List content
# ---------------------------------------------------------------------------


def test_list_content_in_user_message(tmpl):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "from list"}]}]
    result = tmpl.encode(msgs, add_generation_prompt=False)
    assert "from list" in result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_entry():
    assert "gpt_oss" in CHAT_TEMPLATE_REGISTRY
    assert isinstance(CHAT_TEMPLATE_REGISTRY["gpt_oss"], GptOssTemplate)
