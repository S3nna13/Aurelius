import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.serving.tool_executor import ToolExecutor, ToolResult


@pytest.fixture
def executor():
    return ToolExecutor()


# 1. Instantiates with default tools
def test_instantiates_with_default_tools(executor):
    assert isinstance(executor, ToolExecutor)


# 2. list_tools contains the 4 built-ins
def test_list_tools_contains_builtins(executor):
    tools = executor.list_tools()
    for name in ("calculator", "word_count", "current_time", "echo"):
        assert name in tools


# 3. calculator '2+2' -> output '4'
def test_calculator_addition(executor):
    result = executor.execute("calculator", {"expression": "2+2"})
    assert result.success is True
    assert result.output == "4"


# 4. word_count 'hello world' -> contains '2 words'
def test_word_count(executor):
    result = executor.execute("word_count", {"text": "hello world"})
    assert result.success is True
    assert "2 words" in result.output


# 5. echo 'test' -> output 'test'
def test_echo(executor):
    result = executor.execute("echo", {"message": "test"})
    assert result.success is True
    assert result.output == "test"


# 6. current_time -> success=True
def test_current_time(executor):
    result = executor.execute("current_time", {})
    assert result.success is True
    assert result.output  # non-empty


# 7. Unknown tool -> success=False
def test_unknown_tool(executor):
    result = executor.execute("nonexistent_tool", {})
    assert result.success is False
    assert result.error is not None


# 8. Bad calculator expression -> success=False
def test_bad_calculator_expression(executor):
    result = executor.execute("calculator", {"expression": "__import__('os')"})
    assert result.success is False


# 9. parse_tool_call extracts dict from valid block
def test_parse_tool_call_valid(executor):
    text = 'Some text <tool_call>{"name": "echo", "args": {"message": "hi"}}</tool_call>'
    parsed = executor.parse_tool_call(text)
    assert parsed is not None
    assert parsed["name"] == "echo"
    assert parsed["args"] == {"message": "hi"}


# 10. parse_tool_call returns None for text without block
def test_parse_tool_call_no_block(executor):
    assert executor.parse_tool_call("no tool call here") is None


# 11. parse_tool_call returns None for malformed JSON
def test_parse_tool_call_malformed_json(executor):
    assert executor.parse_tool_call("<tool_call>{bad json}</tool_call>") is None


# 12. process with tool_call returns (ToolResult, str)
def test_process_with_tool_call(executor):
    text = 'Result: <tool_call>{"name": "echo", "args": {"message": "hello"}}</tool_call>'
    result, cleaned = executor.process(text)
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert "<tool_call>" not in cleaned


# 13. process with no tool_call returns (None, original_text)
def test_process_no_tool_call(executor):
    text = "Just plain text."
    result, cleaned = executor.process(text)
    assert result is None
    assert cleaned == text
