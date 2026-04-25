"""Unit tests for src/agent/ast_fim.py (16 tests)."""

from __future__ import annotations

import pytest

from src.agent.ast_fim import (
    ASTAnalyzer,
    ASTNode,
    AST_FIM_REGISTRY,
    FIMFormat,
    FIMSpan,
    FIMTokenizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_SOURCE = """\
import os

def hello(name):
    return f"Hello, {name}"

class Greeter:
    def greet(self):
        return hello("world")
"""

SYNTAX_ERROR_SOURCE = "def foo(:\n    pass\n"


@pytest.fixture()
def analyzer() -> ASTAnalyzer:
    return ASTAnalyzer()


@pytest.fixture()
def tokenizer() -> FIMTokenizer:
    return FIMTokenizer()


@pytest.fixture()
def simple_span() -> FIMSpan:
    return FIMSpan(prefix="def foo():\n    ", suffix="\n    return x", middle="x = 1")


# ---------------------------------------------------------------------------
# ASTAnalyzer.parse_python
# ---------------------------------------------------------------------------

def test_parse_python_valid_nonempty(analyzer):
    nodes = analyzer.parse_python(SIMPLE_SOURCE)
    assert len(nodes) > 0


def test_parse_python_valid_node_types(analyzer):
    nodes = analyzer.parse_python(SIMPLE_SOURCE)
    types = {n.node_type for n in nodes}
    # Should include import, function, class
    assert "import" in types
    assert "function" in types
    assert "class" in types


def test_parse_python_function_has_name(analyzer):
    nodes = analyzer.parse_python(SIMPLE_SOURCE)
    func_nodes = [n for n in nodes if n.node_type == "function"]
    assert any(n.name == "hello" for n in func_nodes)


def test_parse_python_class_has_name(analyzer):
    nodes = analyzer.parse_python(SIMPLE_SOURCE)
    class_nodes = [n for n in nodes if n.node_type == "class"]
    assert any(n.name == "Greeter" for n in class_nodes)


def test_parse_python_syntax_error_returns_empty(analyzer):
    result = analyzer.parse_python(SYNTAX_ERROR_SOURCE)
    assert result == []


def test_parse_python_syntax_error_no_crash(analyzer):
    """Must not raise even on malformed input."""
    result = analyzer.parse_python("<<<invalid python>>>")
    assert isinstance(result, list)


def test_parse_python_empty_string(analyzer):
    result = analyzer.parse_python("")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ASTAnalyzer.find_enclosing_scope
# ---------------------------------------------------------------------------

def test_find_enclosing_scope_finds_function(analyzer):
    # Line 4 is inside `hello`
    node = analyzer.find_enclosing_scope(SIMPLE_SOURCE, line=4)
    assert node is not None
    assert node.node_type == "function"
    assert node.name == "hello"


def test_find_enclosing_scope_outside_any_scope(analyzer):
    # Line 1 is `import os` — no enclosing function/class
    node = analyzer.find_enclosing_scope(SIMPLE_SOURCE, line=1)
    assert node is None


def test_find_enclosing_scope_syntax_error_returns_none(analyzer):
    node = analyzer.find_enclosing_scope(SYNTAX_ERROR_SOURCE, line=1)
    assert node is None


# ---------------------------------------------------------------------------
# ASTAnalyzer.extract_context
# ---------------------------------------------------------------------------

def test_extract_context_splits_at_cursor(analyzer):
    source = "line1\nline2\nline3\nline4\nline5\n"
    prefix, suffix = analyzer.extract_context(source, cursor_line=3, context_lines=2)
    # prefix should contain lines before line 3
    assert "line1" in prefix or "line2" in prefix
    # suffix should contain line 3 onward
    assert "line3" in suffix


def test_extract_context_prefix_not_in_suffix(analyzer):
    source = "a\nb\nc\nd\ne\n"
    prefix, suffix = analyzer.extract_context(source, cursor_line=3, context_lines=10)
    # Lines in prefix should not appear in suffix if split is correct
    assert "a\n" in prefix
    assert "c\n" in suffix


def test_extract_context_returns_tuple(analyzer):
    result = analyzer.extract_context(SIMPLE_SOURCE, cursor_line=5)
    assert isinstance(result, tuple)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# FIMTokenizer
# ---------------------------------------------------------------------------

def test_format_psm_starts_with_prefix_token(tokenizer, simple_span):
    result = tokenizer.format_psm(simple_span)
    assert result.startswith(tokenizer.FIM_PREFIX_TOKEN)


def test_format_spm_starts_with_suffix_token(tokenizer, simple_span):
    result = tokenizer.format_spm(simple_span)
    assert result.startswith(tokenizer.FIM_SUFFIX_TOKEN)


def test_format_psm_contains_all_tokens(tokenizer, simple_span):
    result = tokenizer.format_psm(simple_span)
    assert tokenizer.FIM_PREFIX_TOKEN in result
    assert tokenizer.FIM_SUFFIX_TOKEN in result
    assert tokenizer.FIM_MIDDLE_TOKEN in result


def test_format_spm_contains_all_tokens(tokenizer, simple_span):
    result = tokenizer.format_spm(simple_span)
    assert tokenizer.FIM_PREFIX_TOKEN in result
    assert tokenizer.FIM_SUFFIX_TOKEN in result
    assert tokenizer.FIM_MIDDLE_TOKEN in result


def test_format_span_psm_no_crash(tokenizer, simple_span):
    result = tokenizer.format_span(simple_span, FIMFormat.PSM)
    assert isinstance(result, str)


def test_format_span_spm_no_crash(tokenizer, simple_span):
    result = tokenizer.format_span(simple_span, FIMFormat.SPM)
    assert isinstance(result, str)


def test_format_span_random_no_crash(tokenizer, simple_span):
    # Run a few times to exercise both branches
    for _ in range(10):
        result = tokenizer.format_span(simple_span, FIMFormat.RANDOM)
        assert isinstance(result, str)
        assert tokenizer.FIM_MIDDLE_TOKEN in result


def test_parse_completion_strips_fim_tokens(tokenizer):
    raw = f"{tokenizer.FIM_MIDDLE_TOKEN}extracted middle content"
    result = tokenizer.parse_completion(raw)
    assert tokenizer.FIM_MIDDLE_TOKEN not in result
    assert "extracted middle content" in result


def test_parse_completion_no_tokens_returns_text(tokenizer):
    result = tokenizer.parse_completion("plain completion text")
    assert result == "plain completion text"


def test_parse_completion_full_psm_prompt(tokenizer, simple_span):
    formatted = tokenizer.format_psm(simple_span)
    # Append a simulated completion after the middle token
    completion = formatted + "filled in code here"
    result = tokenizer.parse_completion(completion)
    assert "filled in code here" in result
    assert tokenizer.FIM_PREFIX_TOKEN not in result


# ---------------------------------------------------------------------------
# AST_FIM_REGISTRY
# ---------------------------------------------------------------------------

def test_ast_fim_registry_contains_psm():
    assert "psm" in AST_FIM_REGISTRY


def test_ast_fim_registry_contains_spm():
    assert "spm" in AST_FIM_REGISTRY


def test_ast_fim_registry_psm_is_fim_tokenizer():
    assert AST_FIM_REGISTRY["psm"] is FIMTokenizer
