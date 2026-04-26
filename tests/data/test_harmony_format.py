"""Tests for src/data/harmony_format.py — harmony response format parser/formatter."""

from __future__ import annotations

import json

from src.data.harmony_format import (
    HARMONY_TOKENS,
    Conversation,
    Message,
    MessageRole,
    ToolCall,
    build_sft_labels_harmony,
    deserialize_conversation,
    deserialize_message,
    extract_tool_calls_from_text,
    format_conversation_for_training,
    parse_harmony_response,
    serialize_conversation,
    serialize_message,
    validate_conversation,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _simple_tokenize(s: str) -> list[int]:
    """Deterministic char-level tokenizer: each char → ord(c) % 256."""
    return [ord(c) % 256 for c in s]


def _make_simple_conv() -> Conversation:
    return Conversation(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="Hi there, how can I help?"),
        ]
    )


def _make_tool_conv() -> Conversation:
    tc = ToolCall(
        id="call_abc123", function_name="get_weather", function_args='{"location": "NYC"}'
    )
    return Conversation(
        messages=[
            Message(role=MessageRole.USER, content="What's the weather?"),
            Message(role=MessageRole.ASSISTANT, content=None, tool_calls=[tc]),
            Message(
                role=MessageRole.TOOL,
                content='{"temp": 72, "condition": "sunny"}',
                tool_call_id="call_abc123",
                name="get_weather",
            ),
        ]
    )


# ── 1. MessageRole values ─────────────────────────────────────────────────────


def test_message_role_values():
    assert MessageRole.SYSTEM == "system"
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.TOOL == "tool"


# ── 2. Serialize/deserialize message round-trip ───────────────────────────────


def test_serialize_deserialize_message_roundtrip():
    msg = Message(role=MessageRole.USER, content="What is 2+2?")
    data = serialize_message(msg)
    assert isinstance(data, dict)
    restored = deserialize_message(data)
    assert restored.role == msg.role
    assert restored.content == msg.content
    assert restored.tool_calls == []
    assert restored.tool_call_id is None


# ── 3. Serialize message that has tool_calls ──────────────────────────────────


def test_serialize_message_with_tool_calls():
    tc = ToolCall(id="call_1", function_name="calculator", function_args='{"expr": "1+1"}')
    msg = Message(role=MessageRole.ASSISTANT, content=None, tool_calls=[tc])
    data = serialize_message(msg)
    assert "tool_calls" in data
    assert isinstance(data["tool_calls"], list)
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["id"] == "call_1"


# ── 4. Deserialize tool-call message ─────────────────────────────────────────


def test_deserialize_tool_call_message():
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "python"}'},
            }
        ],
    }
    msg = deserialize_message(raw)
    assert msg.role == MessageRole.ASSISTANT
    assert msg.content is None
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.function_name == "search"
    assert tc.function_args == '{"q": "python"}'


# ── 5. Conversation.system_message ───────────────────────────────────────────


def test_conversation_system_message():
    conv = _make_simple_conv()
    sys_msg = conv.system_message()
    assert sys_msg == "You are a helpful assistant."


def test_conversation_system_message_none_when_absent():
    conv = Conversation(messages=[Message(role=MessageRole.USER, content="hi")])
    assert conv.system_message() is None


# ── 6. Conversation.user_messages count ──────────────────────────────────────


def test_conversation_user_messages_count():
    conv = _make_simple_conv()
    user_msgs = conv.user_messages()
    assert len(user_msgs) == 1
    assert all(m.role == MessageRole.USER for m in user_msgs)


# ── 7. Conversation.last_assistant_message ───────────────────────────────────


def test_conversation_last_assistant():
    conv = _make_simple_conv()
    last = conv.last_assistant_message()
    assert last is not None
    assert last.role == MessageRole.ASSISTANT
    assert "help" in last.content.lower()


def test_conversation_last_assistant_none_when_absent():
    conv = Conversation(messages=[Message(role=MessageRole.USER, content="hi")])
    assert conv.last_assistant_message() is None


# ── 8. Serialize/deserialize conversation round-trip ─────────────────────────


def test_serialize_deserialize_conversation_roundtrip():
    conv = _make_tool_conv()
    data = serialize_conversation(conv)
    assert isinstance(data, dict)
    assert "messages" in data
    restored = deserialize_conversation(data)
    assert len(restored.messages) == len(conv.messages)
    assert restored.messages[0].role == MessageRole.USER
    assert restored.messages[1].role == MessageRole.ASSISTANT
    assert len(restored.messages[1].tool_calls) == 1
    assert restored.messages[2].role == MessageRole.TOOL
    assert restored.messages[2].tool_call_id == "call_abc123"


# ── 9. format_conversation_for_training contains role tokens ─────────────────


def test_format_conversation_contains_role_tokens():
    conv = _make_simple_conv()
    text = format_conversation_for_training(conv)
    assert HARMONY_TOKENS["system_start"] in text
    assert HARMONY_TOKENS["user_start"] in text
    assert HARMONY_TOKENS["assistant_start"] in text
    assert "You are a helpful assistant." in text
    assert "Hello!" in text
    assert "Hi there" in text


# ── 10. format_conversation_for_training has end_of_turn tokens ──────────────


def test_format_conversation_end_of_turn():
    conv = _make_simple_conv()
    text = format_conversation_for_training(conv)
    eot = HARMONY_TOKENS["end_of_turn"]
    # Each message should produce one end_of_turn
    assert text.count(eot) == len(conv.messages)


# ── 11. format_conversation add_generation_prompt ────────────────────────────


def test_format_conversation_add_generation_prompt():
    conv = Conversation(
        messages=[
            Message(role=MessageRole.USER, content="Say hi"),
        ]
    )
    text_no_prompt = format_conversation_for_training(conv, add_generation_prompt=False)
    text_with_prompt = format_conversation_for_training(conv, add_generation_prompt=True)
    assert not text_no_prompt.endswith(HARMONY_TOKENS["assistant_start"] + "\n")
    assert text_with_prompt.endswith(HARMONY_TOKENS["assistant_start"] + "\n")


# ── 12. build_sft_labels_harmony shape ───────────────────────────────────────


def test_build_sft_labels_shape():
    conv = _make_simple_conv()
    text = format_conversation_for_training(conv)
    input_ids, labels = build_sft_labels_harmony(text, _simple_tokenize, max_seq_len=2048)
    assert len(input_ids) == len(labels)
    assert len(input_ids) > 0


# ── 13. build_sft_labels_harmony: system tokens are -100 ─────────────────────


def test_build_sft_labels_minus100_for_system():
    conv = _make_simple_conv()
    text = format_conversation_for_training(conv)
    input_ids, labels = build_sft_labels_harmony(text, _simple_tokenize, max_seq_len=2048)
    # At least some labels should be -100 (system and user turns masked)
    assert -100 in labels
    # Not ALL should be -100 — assistant turn should be unmasked
    assert any(line != -100 for line in labels)


# ── 14. validate_conversation: valid conv → empty errors ─────────────────────


def test_validate_conversation_valid():
    conv = _make_simple_conv()
    errors = validate_conversation(conv)
    assert isinstance(errors, list)
    assert len(errors) == 0


# ── 15. validate_conversation: bad ordering → non-empty errors ───────────────


def test_validate_conversation_invalid():
    # Two user messages in a row — bad alternation
    conv = Conversation(
        messages=[
            Message(role=MessageRole.USER, content="First"),
            Message(role=MessageRole.USER, content="Second"),
            Message(role=MessageRole.ASSISTANT, content="Reply"),
        ]
    )
    errors = validate_conversation(conv)
    assert isinstance(errors, list)
    assert len(errors) > 0


# ── 16. extract_tool_calls_from_text ─────────────────────────────────────────


def test_extract_tool_calls_from_text():
    tool_json = json.dumps(
        {
            "id": "call_test1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 3, "b": 4}'},
        }
    )
    text = f"<|tool_call|>{tool_json}<|end_of_turn|>"
    calls = extract_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].id == "call_test1"
    assert calls[0].function_name == "multiply"
    assert calls[0].function_args == '{"a": 3, "b": 4}'


# ── 17. parse_harmony_response: role is extracted ────────────────────────────


def test_parse_harmony_response_role():
    text = "<|assistant|>\nThis is my response.<|end_of_turn|>"
    result = parse_harmony_response(text)
    assert isinstance(result, dict)
    assert result["role"] == "assistant"
    assert result["content"] is not None
    assert "This is my response." in result["content"]
    assert isinstance(result["tool_calls"], list)
