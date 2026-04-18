"""Tests for response_formatter module."""

from src.serving.response_formatter import (
    ResponseFormatter,
    extract_code_blocks,
    format_for_terminal,
    strip_special_tokens,
)


def test_strip_special_tokens_removes_assistant_token():
    result = strip_special_tokens("<|assistant|>Hello")
    assert "<|assistant|>" not in result
    assert "Hello" in result


def test_strip_special_tokens_leaves_regular_text_unchanged():
    text = "Just a normal sentence."
    assert strip_special_tokens(text) == text


def test_extract_code_blocks_finds_python_block():
    text = "Some text\n```python\nprint('hi')\n```\nmore"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0][0] == "python"
    assert "print('hi')" in blocks[0][1]


def test_extract_code_blocks_returns_empty_for_no_fences():
    assert extract_code_blocks("No code here.") == []


def test_format_for_terminal_returns_string():
    result = format_for_terminal("Hello world")
    assert isinstance(result, str)


def test_response_formatter_instantiates():
    rf = ResponseFormatter()
    assert rf.max_length == 2048
    assert rf.strip_tokens is True


def test_format_strips_special_tokens_when_enabled():
    rf = ResponseFormatter(strip_tokens=True)
    result = rf.format("<|user|>Tell me a joke<|end|>")
    assert "<|user|>" not in result
    assert "<|end|>" not in result


def test_format_truncates_at_max_length():
    rf = ResponseFormatter(max_length=10)
    result = rf.format("A" * 100)
    assert len(result) == 10


def test_has_code_true_for_text_with_fence():
    rf = ResponseFormatter()
    assert rf.has_code("Here is code:\n```python\npass\n```") is True


def test_estimate_tokens_returns_int_greater_than_zero():
    rf = ResponseFormatter()
    result = rf.estimate_tokens("The quick brown fox jumps")
    assert isinstance(result, int)
    assert result > 0
