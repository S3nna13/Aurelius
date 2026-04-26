"""Tests for src/search/code_search.py — 22 tests."""

from __future__ import annotations

import pytest

from src.search.code_search import (
    _MAX_FILENAME_LEN,
    _MAX_QUERY_LEN,
    _MAX_RESULTS,
    _MAX_SOURCE_LEN,
    CodeSearchIndex,
    extract_symbols,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SOURCE = '''\
"""Module docstring."""
import os
from pathlib import Path

MY_VAR = 42

class MyClass:
    """A class."""
    pass

def my_func(x):
    """A function."""
    return x + 1

async def async_func():
    pass
'''


def fresh_index() -> CodeSearchIndex:
    return CodeSearchIndex()


# ---------------------------------------------------------------------------
# add_file
# ---------------------------------------------------------------------------


def test_add_file_returns_symbol_count():
    idx = fresh_index()
    count = idx.add_file("mod.py", SIMPLE_SOURCE)
    assert count > 0


def test_add_file_filename_too_long_raises():
    idx = fresh_index()
    long_name = "a" * (_MAX_FILENAME_LEN + 1)
    with pytest.raises(ValueError, match="filename exceeds"):
        idx.add_file(long_name, "x = 1")


def test_add_file_source_too_long_raises():
    idx = fresh_index()
    big_source = "x = 1\n" * ((_MAX_SOURCE_LEN // 6) + 1)
    with pytest.raises(ValueError, match="source exceeds"):
        idx.add_file("big.py", big_source)


# ---------------------------------------------------------------------------
# extract_symbols
# ---------------------------------------------------------------------------


def test_extract_symbols_function():
    syms = extract_symbols("f.py", "def foo(): pass")
    kinds = {s.kind for s in syms}
    assert "function" in kinds
    names = {s.name for s in syms}
    assert "foo" in names


def test_extract_symbols_class():
    syms = extract_symbols("f.py", "class Bar: pass")
    assert any(s.kind == "class" and s.name == "Bar" for s in syms)


def test_extract_symbols_variable():
    syms = extract_symbols("f.py", "MY_CONST = 99")
    assert any(s.kind == "variable" and s.name == "MY_CONST" for s in syms)


def test_extract_symbols_import():
    syms = extract_symbols("f.py", "import os\nfrom sys import path")
    import_names = {s.name for s in syms if s.kind == "import"}
    assert "os" in import_names
    assert "path" in import_names


# ---------------------------------------------------------------------------
# search_symbols
# ---------------------------------------------------------------------------


def test_search_symbols_substring_match():
    idx = fresh_index()
    idx.add_file("mod.py", SIMPLE_SOURCE)
    results = idx.search_symbols("func")
    names = {s.name for s in results}
    assert "my_func" in names or "async_func" in names


def test_search_symbols_exact_ranked_first():
    idx = fresh_index()
    idx.add_file("mod.py", SIMPLE_SOURCE)
    # Add a file with an exact match and a prefix match
    idx.add_file("extra.py", "def foo(): pass\ndef foobar(): pass\n")
    results = idx.search_symbols("foo")
    assert results[0].name == "foo"


def test_search_symbols_prefix_ranked_before_contains():
    idx = fresh_index()
    idx.add_file("mod.py", "def prefix_thing(): pass\ndef has_prefix_inside(): pass\n")
    results = idx.search_symbols("prefix")
    # prefix_thing starts with "prefix"; has_prefix_inside contains it
    names = [s.name for s in results]
    assert names.index("prefix_thing") < names.index("has_prefix_inside")


def test_search_symbols_kind_filter():
    idx = fresh_index()
    idx.add_file("mod.py", SIMPLE_SOURCE)
    results = idx.search_symbols("my", kind="function")
    assert all(s.kind == "function" for s in results)


def test_search_symbols_query_too_long_raises():
    idx = fresh_index()
    idx.add_file("mod.py", "x = 1")
    with pytest.raises(ValueError, match="query exceeds"):
        idx.search_symbols("q" * (_MAX_QUERY_LEN + 1))


def test_search_symbols_top_k_too_large_raises():
    idx = fresh_index()
    idx.add_file("mod.py", "x = 1")
    with pytest.raises(ValueError, match="top_k exceeds"):
        idx.search_symbols("x", top_k=_MAX_RESULTS + 1)


def test_search_symbols_top_k_less_than_1_raises():
    idx = fresh_index()
    idx.add_file("mod.py", "x = 1")
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        idx.search_symbols("x", top_k=0)


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------


def test_remove_file_symbols_gone():
    idx = fresh_index()
    idx.add_file("mod.py", SIMPLE_SOURCE)
    idx.remove_file("mod.py")
    results = idx.search_symbols("my_func")
    assert all(s.filename != "mod.py" for s in results)


def test_remove_file_noop_for_missing():
    idx = fresh_index()
    # Should not raise
    idx.remove_file("nonexistent.py")


# ---------------------------------------------------------------------------
# text_search
# ---------------------------------------------------------------------------


def test_text_search_regex_finds_lines():
    idx = fresh_index()
    idx.add_file("mod.py", SIMPLE_SOURCE)
    results = idx.text_search(r"def my_func")
    assert any("my_func" in line for _, _, line in results)


def test_text_search_invalid_regex_raises():
    idx = fresh_index()
    idx.add_file("mod.py", "x = 1")
    with pytest.raises(ValueError, match="invalid regex"):
        idx.text_search(r"[unclosed")


def test_text_search_top_k_caps_results():
    idx = fresh_index()
    # Create source with many matching lines
    source = "\n".join(f"x_{i} = {i}" for i in range(50))
    idx.add_file("many.py", source)
    results = idx.text_search(r"x_\d+", top_k=5)
    assert len(results) <= 5


# ---------------------------------------------------------------------------
# list_files / __len__
# ---------------------------------------------------------------------------


def test_list_files_sorted():
    idx = fresh_index()
    idx.add_file("b.py", "x = 1")
    idx.add_file("a.py", "y = 2")
    assert idx.list_files() == ["a.py", "b.py"]


def test_len_correct():
    idx = fresh_index()
    assert len(idx) == 0
    idx.add_file("a.py", "x = 1")
    idx.add_file("b.py", "y = 2")
    assert len(idx) == 2


# ---------------------------------------------------------------------------
# Edge / adversarial
# ---------------------------------------------------------------------------


def test_syntax_error_source_succeeds_zero_symbols():
    idx = fresh_index()
    count = idx.add_file("bad.py", "def (broken syntax :")
    assert count == 0
    assert "bad.py" in idx.list_files()


def test_source_with_no_symbols_returns_empty():
    idx = fresh_index()
    count = idx.add_file("empty.py", "# just a comment\n")
    assert count == 0


def test_multi_file_cross_file_search():
    idx = fresh_index()
    idx.add_file("a.py", "def shared_name(): pass")
    idx.add_file("b.py", "def shared_name(): pass")
    results = idx.search_symbols("shared_name")
    filenames = {s.filename for s in results}
    assert "a.py" in filenames
    assert "b.py" in filenames


def test_docstring_stored_in_symbol():
    idx = fresh_index()
    source = 'def documented():\n    """My docstring."""\n    pass\n'
    idx.add_file("doc.py", source)
    results = idx.search_symbols("documented")
    assert results
    sym = results[0]
    assert "My docstring" in sym.docstring
