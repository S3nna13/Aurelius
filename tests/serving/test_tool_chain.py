"""Tests for src/serving/tool_chain.py."""

import pytest
from src.serving.tool_executor import ToolExecutor
from src.serving.tool_chain import ChainStep, ToolChain

TWO_TOOL_TEXT = (
    'First <tool_call>{"name": "echo", "args": {"message": "hello"}}</tool_call> '
    'then <tool_call>{"name": "word_count", "args": {"text": "one two three"}}</tool_call>'
)

NO_TOOL_TEXT = "There are no tool calls here at all."


@pytest.fixture
def chain():
    executor = ToolExecutor()
    return ToolChain(executor)


def test_tool_chain_instantiates():
    executor = ToolExecutor()
    tc = ToolChain(executor)
    assert tc is not None


def test_parse_chain_with_two_calls_returns_list_of_length_two(chain):
    steps = chain.parse_chain(TWO_TOOL_TEXT)
    assert len(steps) == 2


def test_parse_chain_with_no_calls_returns_empty_list(chain):
    steps = chain.parse_chain(NO_TOOL_TEXT)
    assert steps == []


def test_each_chain_step_has_correct_tool_name(chain):
    steps = chain.parse_chain(TWO_TOOL_TEXT)
    assert steps[0].tool_name == "echo"
    assert steps[1].tool_name == "word_count"


def test_execute_chain_fills_in_results(chain):
    steps = chain.parse_chain(TWO_TOOL_TEXT)
    steps = chain.execute_chain(steps)
    assert all(step.result is not None for step in steps)


def test_successful_steps_have_success_true(chain):
    steps = chain.parse_chain(TWO_TOOL_TEXT)
    steps = chain.execute_chain(steps)
    assert all(step.success is True for step in steps)


def test_unknown_tool_step_has_success_false(chain):
    text = '<tool_call>{"name": "no_such_tool", "args": {}}</tool_call>'
    steps = chain.parse_chain(text)
    steps = chain.execute_chain(steps)
    assert len(steps) == 1
    assert steps[0].success is False


def test_format_results_returns_string(chain):
    steps = chain.parse_chain(TWO_TOOL_TEXT)
    steps = chain.execute_chain(steps)
    result = chain.format_results(steps)
    assert isinstance(result, str)


def test_format_results_contains_tool_name(chain):
    steps = chain.parse_chain(TWO_TOOL_TEXT)
    steps = chain.execute_chain(steps)
    result = chain.format_results(steps)
    assert "echo" in result
    assert "word_count" in result


def test_run_returns_tuple_of_list_and_str(chain):
    steps, formatted = chain.run(TWO_TOOL_TEXT)
    assert isinstance(steps, list)
    assert isinstance(formatted, str)
