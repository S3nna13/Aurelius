"""Tests for ProgramOfThought."""
from __future__ import annotations

import pytest
from src.reasoning.program_of_thought import (
    ProgramOfThought,
    PoTConfig,
    PoTResult,
    POT_REGISTRY,
)


def test_extract_code_from_python_markdown_block():
    pot = ProgramOfThought()
    text = "Here is the code:\n```python\nprint(42)\n```"
    assert pot.extract_code(text) == "print(42)"


def test_extract_code_from_generic_markdown_block():
    pot = ProgramOfThought()
    text = "```\nprint('hello')\n```"
    assert pot.extract_code(text) == "print('hello')"


def test_extract_code_fallback_looks_like_code():
    pot = ProgramOfThought()
    code = "print(1 + 1)"
    assert pot.extract_code(code) == code


def test_execute_code_runs_print_statement():
    pot = ProgramOfThought()
    output, success = pot.execute_code("print(42)")
    assert success
    assert "42" in output


def test_execute_code_failure_on_bad_code():
    pot = ProgramOfThought()
    output, success = pot.execute_code("raise ValueError('boom')")
    assert not success


def test_execute_code_timeout():
    pot = ProgramOfThought(PoTConfig(timeout_s=0.1))
    output, success = pot.execute_code("import time; time.sleep(10)")
    assert not success
    assert output == "timeout"


def test_extract_answer_from_output_numeric():
    pot = ProgramOfThought()
    assert pot.extract_answer_from_output("The result is 3.14\n") == "3.14"


def test_extract_answer_from_output_answer_assignment():
    pot = ProgramOfThought()
    result = pot.extract_answer_from_output("answer = 99\n")
    assert result == "99"


def test_run_returns_pot_result():
    pot = ProgramOfThought()
    result = pot.run("what is 2+2?", lambda q: "```python\nprint(4)\n```")
    assert isinstance(result, PoTResult)
    assert result.success
    assert result.answer == "4"


def test_run_sets_code_field():
    pot = ProgramOfThought()
    result = pot.run("q", lambda q: "```python\nprint(7)\n```")
    assert result.code == "print(7)"


def test_registry_key():
    assert "default" in POT_REGISTRY
    assert POT_REGISTRY["default"] is ProgramOfThought
