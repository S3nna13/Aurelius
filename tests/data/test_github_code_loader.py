"""Tests for src/data/github_code_loader.py.

All tests are pure Python — no network, no torch.
"""
from __future__ import annotations

import pytest

from src.data.github_code_loader import (
    CodeFile,
    CodeFunction,
    GitHubIssue,
    code_to_instruction,
    deduplicate_by_content,
    filter_by_language,
    filter_by_quality,
    issue_to_instruction,
    mock_codesearchnet_data,
    mock_github_issues,
    mock_the_stack_data,
    parse_codesearchnet_sample,
    parse_github_issue,
    parse_the_stack_sample,
)


# ---------------------------------------------------------------------------
# 1. mock_codesearchnet_data has correct fields
# ---------------------------------------------------------------------------

def test_mock_codesearchnet_data_fields():
    data = mock_codesearchnet_data(4)
    assert len(data) == 4
    required = {
        "repository_name", "func_name", "whole_func_string",
        "language", "func_documentation_string", "func_code_tokens",
    }
    for sample in data:
        assert required.issubset(sample.keys()), f"Missing fields in {sample.keys()}"


# ---------------------------------------------------------------------------
# 2. parse_codesearchnet_sample returns CodeFunction with code/docstring
# ---------------------------------------------------------------------------

def test_parse_codesearchnet_sample_returns_codefunction():
    raw = mock_codesearchnet_data(1)[0]
    fn = parse_codesearchnet_sample(raw)
    assert isinstance(fn, CodeFunction)
    assert fn.code != ""
    assert fn.docstring != ""
    assert fn.func_name == raw["func_name"]
    assert fn.repo == raw["repository_name"]
    assert fn.language == raw["language"]


# ---------------------------------------------------------------------------
# 3. mock_the_stack_data has content, lang, max_stars fields
# ---------------------------------------------------------------------------

def test_mock_the_stack_data_fields():
    data = mock_the_stack_data(4)
    assert len(data) == 4
    required = {"content", "lang", "max_stars_count", "max_stars_repo_name", "max_stars_repo_path"}
    for sample in data:
        assert required.issubset(sample.keys()), f"Missing fields in {sample.keys()}"


# ---------------------------------------------------------------------------
# 4. parse_the_stack_sample returns CodeFile
# ---------------------------------------------------------------------------

def test_parse_the_stack_sample_returns_codefile():
    raw = mock_the_stack_data(1)[0]
    cf = parse_the_stack_sample(raw)
    assert isinstance(cf, CodeFile)
    assert cf.content == raw["content"]
    assert cf.language == raw["lang"]
    assert cf.repo == raw["max_stars_repo_name"]
    assert cf.path == raw["max_stars_repo_path"]


# ---------------------------------------------------------------------------
# 5. parse_the_stack_sample stars come from max_stars_count
# ---------------------------------------------------------------------------

def test_parse_the_stack_sample_stars_from_max_stars_count():
    raw = mock_the_stack_data(2)[1]
    cf = parse_the_stack_sample(raw)
    assert cf.stars == raw["max_stars_count"]


# ---------------------------------------------------------------------------
# 6. mock_github_issues has correct fields
# ---------------------------------------------------------------------------

def test_mock_github_issues_fields():
    data = mock_github_issues(4)
    assert len(data) == 4
    required = {"number", "title", "body", "state", "labels", "comments", "created_at"}
    for sample in data:
        assert required.issubset(sample.keys()), f"Missing fields in {sample.keys()}"


# ---------------------------------------------------------------------------
# 7. parse_github_issue extracts labels from list of dicts
# ---------------------------------------------------------------------------

def test_parse_github_issue_extracts_labels():
    raw = {
        "number": 1,
        "title": "Test issue",
        "body": "Body text",
        "state": "open",
        "labels": [{"name": "bug"}, {"name": "help wanted"}],
        "comments": 2,
        "created_at": "2024-01-01T00:00:00Z",
        "user": {"login": "alice"},
    }
    issue = parse_github_issue(raw)
    assert isinstance(issue, GitHubIssue)
    assert issue.labels == ["bug", "help wanted"]


# ---------------------------------------------------------------------------
# 8. parse_github_issue state is "open" or "closed"
# ---------------------------------------------------------------------------

def test_parse_github_issue_state_valid():
    for state in ("open", "closed"):
        raw = {
            "number": 99,
            "title": "State test",
            "body": "",
            "state": state,
            "labels": [],
            "comments": 0,
            "created_at": "2024-06-01T00:00:00Z",
        }
        issue = parse_github_issue(raw)
        assert issue.state in ("open", "closed")
        assert issue.state == state


# ---------------------------------------------------------------------------
# 9. code_to_instruction returns dict with instruction/input/output
# ---------------------------------------------------------------------------

def test_code_to_instruction_structure():
    fn = CodeFunction(
        repo="owner/repo",
        func_name="add",
        code="def add(a, b):\n    return a + b",
        docstring="Add two numbers.",
        language="python",
    )
    result = code_to_instruction(fn)
    assert isinstance(result, dict)
    assert "instruction" in result
    assert "input" in result
    assert "output" in result
    assert "add" in result["instruction"]
    assert result["input"] == fn.docstring
    assert result["output"] == fn.code


# ---------------------------------------------------------------------------
# 10. issue_to_instruction returns dict with instruction/input/output
# ---------------------------------------------------------------------------

def test_issue_to_instruction_structure():
    issue = GitHubIssue(
        number=7,
        title="NullPointerException on startup",
        body="App crashes immediately.",
        state="closed",
        labels=["bug", "critical"],
        comments=5,
        created_at="2024-03-01T00:00:00Z",
        user="bob",
    )
    result = issue_to_instruction(issue)
    assert isinstance(result, dict)
    assert "instruction" in result
    assert "input" in result
    assert "output" in result
    assert issue.title in result["input"]
    assert issue.body in result["input"]
    assert issue.state in result["output"]
    assert str(issue.comments) in result["output"]


# ---------------------------------------------------------------------------
# 11. filter_by_language returns only matching language
# ---------------------------------------------------------------------------

def test_filter_by_language():
    files = [parse_the_stack_sample(r) for r in mock_the_stack_data(4)]
    python_files = filter_by_language(files, "Python")
    assert all(f.language.lower() == "python" for f in python_files)
    assert len(python_files) >= 1


def test_filter_by_language_case_insensitive():
    files = [parse_the_stack_sample(r) for r in mock_the_stack_data(4)]
    lower = filter_by_language(files, "python")
    upper = filter_by_language(files, "PYTHON")
    assert len(lower) == len(upper)


# ---------------------------------------------------------------------------
# 12. filter_by_quality removes low-star files
# ---------------------------------------------------------------------------

def test_filter_by_quality_removes_low_stars():
    raw_files = mock_the_stack_data(4)
    # Force one file to have low stars
    raw_files[0]["max_stars_count"] = 1
    raw_files[0]["alphanum_fraction"] = 0.8
    files = [parse_the_stack_sample(r) for r in raw_files]
    quality = filter_by_quality(files, min_stars=10, min_alphanum=0.5)
    star_counts = [f.stars for f in quality]
    assert all(s >= 10 for s in star_counts)
    # The low-star file should not appear
    assert 1 not in star_counts


def test_filter_by_quality_removes_low_alphanum():
    raw_files = mock_the_stack_data(2)
    raw_files[0]["max_stars_count"] = 100
    raw_files[0]["alphanum_fraction"] = 0.1  # too low
    raw_files[1]["max_stars_count"] = 100
    raw_files[1]["alphanum_fraction"] = 0.8
    files = [parse_the_stack_sample(r) for r in raw_files]
    quality = filter_by_quality(files, min_stars=10, min_alphanum=0.5)
    assert len(quality) == 1
    assert quality[0].alphanum_fraction == 0.8


# ---------------------------------------------------------------------------
# 13. deduplicate_by_content removes duplicates
# ---------------------------------------------------------------------------

def test_deduplicate_by_content_removes_duplicates():
    fn1 = CodeFunction(repo="r", func_name="foo", code="def foo(): pass",
                       docstring="", language="python")
    fn2 = CodeFunction(repo="r2", func_name="foo2", code="def foo(): pass",
                       docstring="", language="python")  # same code
    fn3 = CodeFunction(repo="r3", func_name="bar", code="def bar(): return 1",
                       docstring="", language="python")
    result = deduplicate_by_content([fn1, fn2, fn3])
    assert len(result) == 2
    codes = [f.code for f in result]
    assert "def foo(): pass" in codes
    assert "def bar(): return 1" in codes


# ---------------------------------------------------------------------------
# 14. deduplicate_by_content preserves order of first occurrences
# ---------------------------------------------------------------------------

def test_deduplicate_by_content_preserves_order():
    fns = []
    for i in range(5):
        fns.append(CodeFunction(
            repo="r", func_name=f"fn_{i}", code=f"def fn_{i}(): return {i}",
            docstring="", language="python",
        ))
    # Add duplicates of fn_1 and fn_3
    fns.append(CodeFunction(repo="r", func_name="fn_1_dup",
                            code="def fn_1(): return 1", docstring="", language="python"))
    fns.append(CodeFunction(repo="r", func_name="fn_3_dup",
                            code="def fn_3(): return 3", docstring="", language="python"))

    result = deduplicate_by_content(fns)
    assert len(result) == 5  # only 5 unique code strings
    assert [f.func_name for f in result] == ["fn_0", "fn_1", "fn_2", "fn_3", "fn_4"]
