"""Unit tests for src.serving.openai_api_validator."""

from __future__ import annotations

import copy

from src.serving.openai_api_validator import (
    OpenAIChatRequestValidator,
    OpenAIChatResponseValidator,
)


def _codes(errors):
    return sorted({e.code for e in errors})


def _fields(errors):
    return sorted({e.field for e in errors})


# --------------------------------------------------------------------- requests


def test_minimal_valid_request_passes():
    v = OpenAIChatRequestValidator()
    payload = {"model": "aurelius-1.4b", "messages": [{"role": "user", "content": "hello"}]}
    assert v.validate(payload) == []


def test_missing_model_field_errors():
    v = OpenAIChatRequestValidator()
    errs = v.validate({"messages": [{"role": "user", "content": "x"}]})
    assert any(e.field == "model" and e.code == "missing_field" for e in errs)


def test_missing_messages_errors():
    v = OpenAIChatRequestValidator()
    errs = v.validate({"model": "m"})
    assert any(e.field == "messages" and e.code == "missing_field" for e in errs)


def test_message_without_role_errors():
    v = OpenAIChatRequestValidator()
    errs = v.validate({"model": "m", "messages": [{"content": "hi"}]})
    assert any(e.field == "messages[0].role" and e.code == "missing_field" for e in errs)


def test_message_without_content_errors_for_user_role():
    v = OpenAIChatRequestValidator()
    errs = v.validate({"model": "m", "messages": [{"role": "user"}]})
    assert any(e.field == "messages[0].content" and e.code == "missing_field" for e in errs)


def test_assistant_message_with_tool_calls_only_is_valid():
    # Spec carve-out: assistant may omit content if tool_calls is present.
    v = OpenAIChatRequestValidator()
    payload = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "call the tool"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ],
    }
    assert v.validate(payload) == []


def test_temperature_out_of_range_errors():
    v = OpenAIChatRequestValidator()
    errs = v.validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 2.5}
    )
    assert any(e.field == "temperature" and e.code == "out_of_range" for e in errs)
    errs = v.validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": -0.1}
    )
    assert any(e.field == "temperature" and e.code == "out_of_range" for e in errs)


def test_top_p_out_of_range_errors():
    v = OpenAIChatRequestValidator()
    errs = v.validate({"model": "m", "messages": [{"role": "user", "content": "x"}], "top_p": 1.5})
    assert any(e.field == "top_p" and e.code == "out_of_range" for e in errs)


def test_negative_max_tokens_errors():
    v = OpenAIChatRequestValidator()
    errs = v.validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": -1}
    )
    assert any(e.field == "max_tokens" and e.code == "out_of_range" for e in errs)


def test_tools_with_invalid_schema_errors():
    v = OpenAIChatRequestValidator()
    errs = v.validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "not_a_function"}],  # missing function, wrong type
        }
    )
    assert any("tools[0]" in e.field for e in errs)
    assert "invalid_value" in _codes(errs) or "missing_field" in _codes(errs)


def test_n_must_be_positive_integer():
    v = OpenAIChatRequestValidator()
    errs = v.validate({"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 0})
    assert any(e.field == "n" and e.code == "out_of_range" for e in errs)


def test_stop_accepts_string_list_and_null():
    v = OpenAIChatRequestValidator()
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    assert v.validate({**base, "stop": None}) == []
    assert v.validate({**base, "stop": "END"}) == []
    assert v.validate({**base, "stop": ["END", "STOP"]}) == []
    errs = v.validate({**base, "stop": 42})
    assert any(e.field == "stop" for e in errs)


def test_strict_false_allows_unknown_top_level_fields():
    v = OpenAIChatRequestValidator(strict=False)
    errs = v.validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "custom_future_field": True,
        }
    )
    assert errs == []


def test_unknown_field_in_strict_mode_errors():
    v = OpenAIChatRequestValidator(strict=True)
    errs = v.validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "custom_future_field": True,
        }
    )
    assert any(e.code == "unknown_field" for e in errs)


def test_determinism():
    v = OpenAIChatRequestValidator()
    payload = {
        "model": "m",
        "messages": [{"role": "user"}],  # missing content
        "temperature": 3,  # out of range
        "top_p": -1,  # out of range
    }
    a = v.validate(copy.deepcopy(payload))
    b = v.validate(copy.deepcopy(payload))
    assert a == b
    assert [e.field for e in a] == [e.field for e in b]


def test_tool_call_in_request_requires_function_name():
    v = OpenAIChatRequestValidator()
    errs = v.validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {}}],
                }
            ],
        }
    )
    assert any(e.field.endswith("function.name") and e.code == "missing_field" for e in errs)


# -------------------------------------------------------------------- responses


def _valid_response():
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "aurelius-1.4b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }


def test_response_valid_streaming_chunk():
    v = OpenAIChatResponseValidator()
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "aurelius-1.4b",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "hel"}, "finish_reason": None}
        ],
    }
    assert v.validate(chunk) == []


def test_response_invalid_finish_reason_errors():
    v = OpenAIChatResponseValidator()
    bad = _valid_response()
    bad["choices"][0]["finish_reason"] = "exploded"
    errs = v.validate(bad)
    assert any(e.field == "choices[0].finish_reason" and e.code == "invalid_value" for e in errs)


def test_response_tool_calls_in_assistant_valid():
    v = OpenAIChatResponseValidator()
    payload = _valid_response()
    payload["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
            }
        ],
    }
    payload["choices"][0]["finish_reason"] = "tool_calls"
    assert v.validate(payload) == []
